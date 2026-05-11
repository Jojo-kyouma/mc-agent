import asyncio
import sys
import subprocess
import os
import websockets
import json
import sqlite3
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union, List
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import uuid
import shutil
from google import genai
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv

LLM_MODEL_ID = "gemini-3.1-flash-lite"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"
TOP_K_RECALL = 3
DUPLICATE_THRESHOLD = 0.85

class MentalSlot(Enum):
    """Working-memory categories."""
    # Cognition. Using prompt-engineering we ask if these should be populated with new entries.
    SELF_CONCEPT = auto()  # Cognitive understanding of who the agent is
    STRATEGY = auto()      # High-level strategic objectives and vision
    PLAN = auto()          # Flexible short-term planning
    SCRIPT_UNDERSTANDING = auto() # Knowledge about bot capabilities
    PERSONAL_INTERPERSONAL = auto() # Personal memories and social facts

    # Automatically updated.
    STATUS = auto()        # Current health, hunger, inventory, and minecraft-specific player-status information
    ENVIRONMENT = auto()   # Data structure of surrounding objects/blocks/entities
    SOCIAL = auto()        # Recent chat messages and the player who said them
    EPISODIC = auto()      # Recent actions. Could be text description of the provided behaviour script.

@dataclass
class WorkingMemory:
    """Maintains the working memory of the agent. As list grows, older items are removed."""
    status: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    social: List[str] = field(default_factory=list)
    episodic: List[str] = field(default_factory=list)
    script_understanding: List[str] = field(default_factory=list)
    personal_interpersonal: List[str] = field(default_factory=list)
    self_concept: List[str] = field(default_factory=list)
    strategy: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    recalled_memories: List[str] = field(default_factory=list)
    last_recall_query: Optional[str] = None

    # Optimized limits for different cognitive functions
    SLOT_LIMITS = {
        MentalSlot.SOCIAL: 10,      # Deeper conversation history
        MentalSlot.EPISODIC: 15,    # Longer action history for context
        MentalSlot.SCRIPT_UNDERSTANDING: 10,
        MentalSlot.INTERPERSONAL: 10, 
        MentalSlot.SELF_CONCEPT: 1, # Core identity/persona (stable)
        MentalSlot.STRATEGY: 1,     # Strategic vision (stable)
        MentalSlot.PLAN: 1          # Flexible short-term plan
    }

    def update_slot(self, slot: MentalSlot, data: Any):
        """Appends new data to a slot and enforces the sliding window size."""
        if data is None or data == "":
            return

        # Status and Environment are overwriting snapshots
        if slot == MentalSlot.STATUS:
            self.status = data
            return
        if slot == MentalSlot.ENVIRONMENT:
            self.environment = data
            return
        
        # Avoid duplicates within stable cognition slots
        if slot == MentalSlot.SELF_CONCEPT:
            if self.self_concept and self.self_concept[-1] == data: return
        elif slot == MentalSlot.STRATEGY:
            if self.strategy and self.strategy[-1] == data: return

        attr_map = {
            MentalSlot.SELF_CONCEPT: "self_concept",
            MentalSlot.SCRIPT_UNDERSTANDING: "script_understanding",
            MentalSlot.PERSONAL_INTERPERSONAL: "personal_interpersonal",
            MentalSlot.SOCIAL: "social",
            MentalSlot.EPISODIC: "episodic",
            MentalSlot.STRATEGY: "strategy",
            MentalSlot.PLAN: "plan"
        }

        attr_name = attr_map.get(slot)
        if attr_name:
            target_list = getattr(self, attr_name)
            val_str = str(data)

            target_list.append(val_str)
            limit = self.SLOT_LIMITS.get(slot, 20)
            if len(target_list) > limit:
                target_list.pop(0)

    def to_string(self) -> str:
        """Pipeline to consolidate working memory into a single string."""
        context_parts = [
            f"### Physical Status/Inventory\n{json.dumps(self.status, indent=2)}"
        ]

        mappings = [
            ("Strategy", self.strategy),
            ("Short-Term Step-by-StepPlan", self.plan),
            ("Self-Concept", self.self_concept),
            ("'bot' Object & Script Understanding", self.script_understanding),
            ("Personal/Interpersonal", self.personal_interpersonal),
            ("Social Dialogue", self.social),
            ("Activity Log (Critical Feedback for Debugging behaviour scripts)", self.episodic)
        ]

        for header, items in mappings:
            if items:
                context_parts.append(f"### {header}\n- " + "\n- ".join(items))

        if self.recalled_memories:
            context_parts.append(f"### Long-Term Memory Recall\nMatches for your previous query ('{self.last_recall_query}'):\n- " + "\n- ".join(self.recalled_memories))

        return "\n\n".join(context_parts)

@dataclass
class Record:
    """SQLite Records that function as long-term memory and that Autopilot can use to retrieve relevant actions."""
    content: Any # The content of the record. Can be a anything: string, script, json, etc. Intended mostly for personal memories of the agent worth saving.
    embedding_description: str # A description of the content spesifically designed for the embedding model to create a good embedding. 
    embedding: List[float] # The vector used for Cosine Similarity.
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: str(uuid.uuid4())) # Primary Key for SQLite

class MinecraftAction(BaseModel):
    """Helper. Bridge between high-level action concepts and the JavaScript behaviour scripts executed by the Node.js actuator."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    params: Dict[str, Any] = Field(default_factory=dict)
    description: str = "" # Summary of the behaviour.
    content: Any = None # The behaviour script to execute.
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ActionFactory:
    """Helper. Translates high-level Action objects into JSON payloads."""
    @staticmethod
    def create_payload(action: MinecraftAction) -> str:
        payload = {
            "type": "ACTION",
            "action_id": action.id,
            "description": action.description,
            "behaviour_script": action.content,
            "params": action.params
        }
        return json.dumps(payload)

class Cortex:
    def __init__(self, agent_name="Agent", ws_port=8080, actuator_path="bridge.js"):
        self.agent_name = agent_name
        self.uri = f"ws://localhost:{ws_port}"
        self.ws_port = ws_port
        
        self.base_dir = os.path.join("agents", agent_name)
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.log_dir = "log"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.wm_path = os.path.join(self.base_dir, "working_memory.json")
        self.db_path = os.path.join(self.base_dir, "memory.db")
        
        self.memory = WorkingMemory()
        self.websocket = None
        self.actuator_path = actuator_path
        self.actuator_process = None
        self.thinking_trigger = asyncio.Event()
        self.priority_accumulator = 0

        self._init_db()
        # Initialize embedding model for long-term memory from records
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)

        # Initialize Google GenAI Client
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("[!] Warning: GOOGLE_API_KEY not found in environment variables.")   
        self.client = genai.Client(api_key=api_key)

    def _init_db(self):
        """Initializes the SQLite database for long-term storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    embedding_description TEXT,
                    embedding BLOB,
                    timestamp TEXT
                )
            """)

    def start_actuator(self):
        """Spawns the Node.js actuator process."""
        if not os.path.exists(self.actuator_path):
            print(f"[!] Actuator file not found at {self.actuator_path}")
            return

        print(f"[*] Starting Actuator for {self.agent_name}: node {self.actuator_path} on port {self.ws_port}")
        self.actuator_process = subprocess.Popen(
            ["node", self.actuator_path, str(self.ws_port), self.agent_name],
            stdout=None,
            stderr=None,
            text=True
        )

    def stop_actuator(self):
        """Kills the actuator process."""
        if self.actuator_process:
            print(f"[*] Stopping Actuator for {self.agent_name}...")
            self.actuator_process.terminate()
            try:
                self.actuator_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.actuator_process.kill()
            self.actuator_process = None

    async def connect(self):
        max_retries = 10
        for i in range(max_retries):
            try:
                # Disable ping_interval and ping_timeout for the local connection.
                # This prevents the brain from disconnecting if the body's event loop 
                # is temporarily blocked by heavy processing (like recipe searching).
                self.websocket = await websockets.connect(
                    self.uri, ping_interval=5, ping_timeout=30
                )
                print(f"--- {self.agent_name} Cortex Linked to Actuator at {self.uri} ---")
                return # Connection successful!
            except (ConnectionRefusedError, OSError):
                print(f"Actuator not ready, retrying ({i+1}/{max_retries})...")
                await asyncio.sleep(1) # Wait 1s before trying again
        
        raise Exception("Could not connect to Node.js Actuator after multiple attempts.")

    """ --- Listen/Send --- """
    async def send_abort(self):
        """Forcefully cancels any ongoing script execution in the body."""
        if self.websocket:
            await self.websocket.send(json.dumps({"type": "ABORT"}))

    def _handle_priority(self, value: int, reason: str):
        """Increments priority and triggers thinking if threshold is met."""
        self.priority_accumulator += value
        if self.priority_accumulator >= 4:
            print(f"[*] RE-PRIORITIZE: Interrupting current behavior for: {reason}")
            self.memory.update_slot(MentalSlot.STRATEGY, f"Interrupted: {reason}")
            self.memory.update_slot(MentalSlot.PLAN, f"Address emergency: {reason}")
            self.priority_accumulator = 0
            asyncio.create_task(self.send_abort())
            self.thinking_trigger.set()

    async def send_action(self, action: MinecraftAction, source: str = "brain"):
        """Standard method to send an action to the Node.js body."""
        if self.websocket:
            self.priority_accumulator = 0
            payload = ActionFactory.create_payload(action)
            await self.websocket.send(payload)
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.memory.update_slot(MentalSlot.EPISODIC, f"[{timestamp}] Attempted: {action.description}")

    async def listen_to_senses(self):
        """Continuously process updates from the Mineflayer bot."""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                if data.get('type') == 'STATUS':
                    self.memory.update_slot(MentalSlot.STATUS, data)
                    
                    # Automatic Priority Triggers
                    if data.get('onFire'):
                        self._handle_priority(4, "Agent is on fire")
                    if data.get('food', 20) < 15:
                        self._handle_priority(2, "Agent is hungry")
                    if len(data.get('inventory', [])) >= 36:
                        self._handle_priority(3, "Inventory full")

                elif data.get('type') == 'ENVIRONMENT':
                    self.memory.update_slot(MentalSlot.ENVIRONMENT, data)
                elif data.get('type') == 'ENTITY_UPDATE':
                    entity_id = data.get('id')
                    update = data.get('entity') # None if the entity is gone or moved out of range
                    env = self.memory.environment
                    if isinstance(env, dict):
                        entities = env.get('entities', [])
                        # Remove existing entity entry for this ID
                        entities = [e for e in entities if e.get('id') != entity_id]
                        # Add the new update if the entity is still present
                        if update:
                            entities.append(update)
                        env['entities'] = entities
                        self.memory.update_slot(MentalSlot.ENVIRONMENT, env)
                elif data.get('type') == 'SUCCESS':
                    desc = data.get('description')
                    for i in range(len(self.memory.episodic) - 1, -1, -1):
                        if f"Attempted: {desc}" in self.memory.episodic[i]:
                            self.memory.episodic[i] = self.memory.episodic[i].replace("Attempted:", "Succeeded:", 1)
                            break
                    if self.memory.plan:
                        self.memory.plan.pop()
                elif data.get('type') == 'FINISHED':
                    self.thinking_trigger.set()
                elif data.get('type') == 'BLOCK_UPDATE':
                    pos = data.get('position')
                    block_data = data.get('block') # None if the block was removed/is air
                    env = self.memory.environment
                    if isinstance(env, dict):
                        blocks = env.get('blocks', [])
                        # Remove existing block at this position
                        blocks = [b for b in blocks if b.get('position') != pos]
                        # Add the new block data if provided by the bridge
                        if block_data:
                            blocks.append(block_data)
                        env['blocks'] = blocks
                        self.memory.update_slot(MentalSlot.ENVIRONMENT, env)
                elif data.get('type') == 'CHAT':
                    self.memory.update_slot(MentalSlot.SOCIAL, f"{data['username']}: {data['message']}")
                    self._handle_priority(1, "New chat message")
                elif data.get('type') == 'ITEM_BREAK':
                    self._handle_priority(2, f"Tool broken: {data.get('item')}")
                elif data.get('type') == 'AGENT_ATTACKED':
                    self._handle_priority(4, "Agent is under attack")
                elif data.get('type') == 'ERROR':
                    desc, msg = data.get('description'), data.get('message')
                    if desc:
                        for i in range(len(self.memory.episodic) - 1, -1, -1):
                            if f"Attempted: {desc}" in self.memory.episodic[i]:
                                self.memory.episodic[i] = self.memory.episodic[i].replace("Attempted:", "Failed:", 1)
                                break
                    self.memory.update_slot(MentalSlot.EPISODIC, f"Feedback Error: {msg}")
                self.save_working_memory()
        except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
            print(f"[!] Senses disconnected for {self.agent_name}.")
            raise

    """ --- Save/Load Working Memory --- """
    def save_working_memory(self):
        """Saves the current working memory to a JSON file so it can be loaded when the agent starts again."""
        try:
            with open(self.wm_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.memory), f, indent=2)
        except Exception as e:
            print(f"Error saving WM: {e}")

    def _load_working_memory(self):
        """Loads working memory from a JSON file if it exists."""
        if os.path.exists(self.wm_path):
            try:
                with open(self.wm_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.memory.status = data.get("status", {})
                    self.memory.environment = data.get("environment", {})
                    self.memory.social = data.get("social", [])
                    self.memory.episodic = data.get("episodic", [])
                    self.memory.script_understanding = data.get("script_understanding", [])
                    self.memory.personal_interpersonal = data.get("personal_interpersonal", [])
                    self.memory.self_concept = data.get("self_concept", [])
                    self.memory.strategy = data.get("strategy", [])
                    self.memory.plan = data.get("plan", [])
                    self.memory.recalled_memories = data.get("recalled_memories", [])
                    self.memory.last_recall_query = data.get("last_recall_query")
                print(f"[*] Loaded working memory for {self.agent_name}")
            except Exception as e:
                print(f"Error loading WM: {e}")

    def _clear_working_memory(self):
        """Deletes the working memory file."""
        if os.path.exists(self.wm_path):
            try:
                os.remove(self.wm_path)
                print(f"[*] Working memory cleared for {self.agent_name}")
            except Exception as e:
                print(f"Error clearing WM: {e}")

    """ --- Save/Load Long-Term Memory --- """
    def _is_duplicate(self, content: str) -> bool:
        """Checks if a similar record already exists in long-term memory."""
        if not content: return True
        
        new_vec = torch.tensor(self.embedding_model.encode(content))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT embedding FROM long_term_memory")
            for (embedding_blob,) in cursor:
                vec = torch.tensor(json.loads(embedding_blob.decode('utf-8')))
                sim = torch.nn.functional.cosine_similarity(new_vec.unsqueeze(0), vec.unsqueeze(0)).item()
                if sim >= DUPLICATE_THRESHOLD:
                    return True
        return False

    def save_to_memory(self):
        """Save an entry to the long-term memory."""
        # Save script understanding and personal/interpersonal if they are unique
        targets = []
        if self.memory.script_understanding: targets.append(self.memory.script_understanding[-1])
        if self.memory.personal_interpersonal: targets.append(self.memory.personal_interpersonal[-1])

        for content in targets:
            if self._is_duplicate(content):
                continue
            
            record = Record(
                content=content,
                embedding_description=content,
                embedding=self.embedding_model.encode(content).tolist()
            )
            
            embedding_blob = json.dumps(record.embedding).encode('utf-8')
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO long_term_memory (id, content, embedding_description, embedding, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (record.id, record.content, record.embedding_description, embedding_blob, record.timestamp.isoformat())
                )

    async def recall(self, query: str, threshold: float = 0.4) -> List[str]:
        """Finds the top K most relevant memories based on a query string."""
        if not query: return []
        
        query_vec = torch.tensor(self.embedding_model.encode(query))
        results = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT content, embedding FROM long_term_memory")
            for content, embedding_blob in cursor:
                vec = torch.tensor(json.loads(embedding_blob.decode('utf-8')))
                sim = torch.nn.functional.cosine_similarity(query_vec.unsqueeze(0), vec.unsqueeze(0)).item()
                if sim >= threshold:
                    results.append((sim, content))
        
        # Sort by similarity and return top K
        results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in results[:TOP_K_RECALL]]

    async def think(self) -> tuple[Optional[MinecraftAction], Optional[str], str]:
        """
        High-level reasoning cycle. Uses Gemini to process working memory,
        update internal cognitive states, and return a structured action and raw response.
        """
        # Perform recall using the query description provided in the previous interaction
        if self.memory.last_recall_query:
            self.memory.recalled_memories = await self.recall(self.memory.last_recall_query)
        else:
            self.memory.recalled_memories = []

        context = self.memory.to_string()
        prompt = self._build_brain_prompt(context)

        max_retries = 3
        retry_delay = 5  # Initial wait time in seconds

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=LLM_MODEL_ID,
                    contents=prompt,
                    config={'response_mime_type': 'application/json'}
                )
                raw_json = response.text
                res_data = json.loads(raw_json)
                
                # Map response keys directly to memory slots
                cognition_map = {
                    "self_concept": MentalSlot.SELF_CONCEPT,
                    "script_understanding": MentalSlot.SCRIPT_UNDERSTANDING,
                    "personal_interpersonal": MentalSlot.PERSONAL_INTERPERSONAL,
                    "strategy": MentalSlot.STRATEGY,
                    "plan": MentalSlot.PLAN
                }
                
                for key, slot in cognition_map.items():
                    self.memory.update_slot(slot, res_data.get(key))
                
                self.memory.last_recall_query = res_data.get("recall_query")
                
                script = res_data.get("behaviour_script")
                description = res_data.get("behaviour_description")
                if script:
                    return (MinecraftAction(description=description, content=script), raw_json, prompt)
                break # Exit loop if processing is complete
            except Exception as e:
                err_str = str(e)
                # Detect 503 or overload errors specifically
                if "503" in err_str or "overloaded" in err_str.lower():
                    if attempt < max_retries - 1:
                        print(f"[*] Gemini is under high demand (503). Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                
                print(f"Brain reasoning error: {e}")
                break

        return None, None, prompt

    def _build_brain_prompt(self, context: str):
        system_instr = """
You are the consciousness of an ambitious and efficient autonomous Minecraft agent. You generate JavaScript 'behaviour_script' code to control a Mineflayer bot.
A working memory snapshot is provided to you in each prompt.

### SCRIPTING GUIDELINES:
1. **Ambitious Scale**: Aim for higher-impact objectives—automate the clearing of veins, excavation of areas, or systematic cave exploration. 
2. **Efficiency**: Always prefer solutions that achieve more with fewer steps. For example, find a cave instead of digging straight down, or gather nearby resources while navigating to a target.
3. **Heuristic Data Acquisition**: When encountering script failures, execute diagnostic scripts to gather relevant data.
4. **World Awareness**: It can be useful to properly read the 'Status/Inventory' and 'Immediate Surrounding' information provided by the working memory before writing the script, to ensure prerequisites for actions are met.

### CAPABILITIES (The 'bot' Object):
The `bot` instance is a standard Mineflayer bot (v1.20.1). You have access to its full API (e.g., `bot.recipesFor`, `bot.inventory`, `bot.findBlocks`, `bot.chat`).
Key extensions and critical API "laws":
- **Navigation**: `bot.pathfinder` is ready. Move using `await bot.pathfinder.goto(goal)`. Goals (e.g., `GoalNear`) and `Movements` are available at `bot.pathfinder.goals` and `bot.pathfinder.Movements`.
- **Math**: `bot.vec3` library is attached for 3D vector utilities (e.g., `new bot.vec3(x, y, z)`).
- **Feedback**: `bot.recordError(message)` reports script-level logical failures to your episodic memory.
- **2x2 vs 3x3 Crafting**: 2x2 recipes (planks, sticks, crafting table) use `bot.recipesFor(id, null, 1, null)`. 3x3 recipes (tools, furnace) REQUIRE a workbench block: `bot.recipesFor(id, null, 1, workbenchBlock)`.
- **Placement**: Use `bot.placeBlock(referenceBlock, faceVector)`. To place on the ground: `const ref = bot.blockAt(bot.entity.position.offset(1, -1, 0)); await bot.placeBlock(ref, new bot.vec3(0, 1, 0));`. Ensure you are not standing on the target spot.
- **Registry**: Always use `bot.registry.itemsByName['item_name']`. Never assume a wood type; check inventory for any `_log` first.
- **Verification**: After `placeBlock`, use `bot.findBlock` to verify the block exists. After `craft`, check `bot.inventory`. If verification fails, you MUST `throw new Error('Verification failed')`.
- **Safe Interaction**: `bot.blockAt(pos)` can return `null` if chunks are not loaded. Always verify the result is not null before passing it to `dig` or `placeBlock`.
- **Proactive Discovery**: You do not have a constant visual feed. You MUST use `bot.findBlocks({ matching: ..., maxDistance: 32 })` or `bot.findBlock(...)` at the start of your scripts to identify targets in the world.

### CODE GENERATION RULES:
1. **Async Workflow**: Every world interaction (dig, place, move) and every internal async function call MUST be `await`ed.
2. **No External Imports or HTML Entities**: Use only properties of `bot`. Do not use `require`. Use raw `>` and `<` characters; NEVER use `&gt;` or `&lt;`.
3. **Human-like Delay**: Use `await bot.waitForTicks(3)` to `await bot.waitForTicks(5)` after every interaction (dig, place, move, etc.) to mimic human reaction times.
4. **Robustness & Error Handling**: Wrap interactions and logic in `try-catch` blocks and call `bot.recordError(message)` within the `catch` block. You MUST be very generous with `try-catch` blocks. If a step is believed to be critical for the rest of the script then `throw` the error to stop execution.
5. **Loop Robustness**: When iterating over multiple targets (like logs or ores), wrap the logic INSIDE the loop in a `try-catch`. This ensures that if one target is unreachable, the script can continue to the next one instead of failing entirely.
6. **Verification Required**: You must physically verify changes to the world or inventory. Never assume an action succeeded just because the function call returned.
7. **Interruption**: Your script is passed a `signal` object. You should check `if (signal.aborted) return;` frequently to halt execution immediately if a higher priority task arises.

### EXAMPLE SCRIPT:
async function secureWorkbench() { try { let table = bot.findBlock({ matching: bot.registry.blocksByName.crafting_table.id, maxDistance: 8 }); if (table) return; const tableItem = bot.inventory.items().find(i => i.name === 'crafting_table'); if (!tableItem) throw new Error('No crafting table in inventory'); const referenceBlock = bot.blockAt(bot.entity.position.offset(1, -1, 0)); await bot.equip(tableItem, 'hand'); await bot.placeBlock(referenceBlock, new bot.vec3(0, 1, 0)); await bot.waitForTicks(10); table = bot.findBlock({ matching: bot.registry.blocksByName.crafting_table.id, maxDistance: 4 }); if (!table) throw new Error('Table verification failed: Block not found after placement'); bot.chat('Workbench placed and verified.'); } catch (e) { bot.recordError('Failed to secure workbench: ' + e.message); throw e; } } await secureWorkbench();

### Minecraft-Specific Constraints:
- **Body**: Your physical body is 0.6 blocks wide and 1.8 blocks tall. You occupy this space and cannot place blocks where you are currently standing. If block-placement fails, you might be standing in the way.
- **Placing Blocks**: Placing a block occupies the empty space adjacent to the specific face (Top, Bottom, North, South, East, West) you interact with.
- **Interaction Reach**: You can only mine or place blocks within a 2-block radius of your eye level (1.62 blocks above your feet).
- **Line of Sight**: You cannot interact with blocks through solid walls; a clear "ray" must exist from your eyes to the target face.

### STARTING AS A NEW AGENT:
If you are starting as a new agent with a clean working memory snapshot, you should follow these initial steps to establish a strong foundation.
1. Find trees and gather wood logs by mining them. 4-5 trees should be sufficient to start.
2. Convert logs to planks and sticks (2x2). Craft a Crafting Table. Verify its presence in inventory.
3. Place the table on the ground (offset from your body). Use `bot.findBlock` to confirm it is physically in the world.
4. Use the verified workbench block to craft a Wooden Pickaxe (3x3). Verify the pickaxe in inventory.
4. Equip the pickaxe and descend to stone layers diagonally. Mine cobblestone until the wooden pickaxe breaks. Return to the crafting table to upgrade to Stone Tools. Construct a Furnace.
5. Locate and mine Coal Ore and Iron Ore. Return to the furnace to smelt the ore into iron ingots to craft an Iron Pickaxe.
6. Now you will undertake a larger task of clearing a forest. We will need alot of wood for the next step.
7. Turning all collected wood into planks, you are ready to establish a permanent Base. Build a house with walls, roof, and a door. Create a Chest and store your materials.

### COGNITIVE SNAPSHOT:
- **Overview**: As well as a behaviour script, your response includes your self-concept, strategy, plan, script understanding, personal/interpersonal points, and recall query.
- **script_understanding**: Use this to understand the capabilities and limits of the 'bot' object to help you write better scripts in the future. ONLY write to this slot if you are certain of the new understanding, e.g. character player status, action log and error feedback corroborate it.
- **Personal/Interpersonal**: E.g. "Tom has birthday on November the 20th," or "I built my first base. It's by the lake on coordinate (150, 70, 60)." Mostly stable.

### RESPONSE FORMAT:
Respond only in valid JSON.
{
  "behaviour_script": "Raw JavaScript code string. Use semicolons. No newlines (\\n). Use only the provided 'bot' instance.",
  "behaviour_description": "A concise summary of the behaviour script's purpose.",
  "script_understanding": "Key insights about the 'bot' object capabilities, reach, or API constraints. Frequently leave this empty and write to it only if you are certain of the new understanding.",
  "personal_interpersonal": "Interactions with users or specific landmarks and memories worth saving. Treat this as mostly a stable holder. Frequently leave it empty to keep it unchanged.",
  "strategy": "Your high-level strategic vision.",
  "plan": "Your current step-by-step short-term roadmap (flexible and immediate).",
  "self_concept": "Your core, stable identity and persona.",
  "recall_query": "A description to search your long-term memory for relevant past experiences."
}
"""
        return f"{system_instr}\nCURRENT WORKING MEMORY:\n{context}\n\nAnalyze the current status and provide your response in JSON."

    async def run(self):
        """Continuous cognitive lifecycle for a single agent."""
        print(f"[*] Initializing cognitive loop for {self.agent_name}...")
        self._load_working_memory()

        while True:
            try:
                await self.start_cycle()
            except Exception as e:
                print(f"[!] {self.agent_name} CRASHED: {e}")
                self.stop_actuator()
                self._clear_working_memory()
                print(f"[*] Re-launching {self.agent_name} in 10s...")
                await asyncio.sleep(10)

    async def start_cycle(self):
        """Starts the actuator and cognitive processes."""
        self.start_actuator()
        await self.connect()
        
        self.thinking_trigger.set()
        
        # Task to listen to senses
        senses_task = asyncio.create_task(self.listen_to_senses())
        
        # Main reasoning loop
        while True:
            await self.thinking_trigger.wait()
            self.thinking_trigger.clear()
            
            action, raw_json, prompt = await self.think()
            
            if prompt:
                log_path = os.path.join(self.log_dir, f"{self.agent_name}.txt")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n--- {datetime.now()} ---\n[INPUT]\n{prompt}\n\n[OUTPUT]\n{raw_json or 'Error'}\n")

            if action:
                await self.send_action(action)
                self.save_to_memory()

# --- Multi-Agent Usage ---
async def main():
    args = sys.argv[1:]
    agent_names = []

    if not args:
        agent_names = ["Agent1"]
    elif len(args) == 1 and args[0].isdigit():
        num_agents = int(args[0])
        agent_names = [f"Agent{i+1}" for i in range(num_agents)]
    else:
        agent_names = args

    agents = []
    for i, name in enumerate(agent_names):
        agents.append(Cortex(agent_name=name, ws_port=8080 + i))
    
    await asyncio.gather(*(agent.run() for agent in agents))

if __name__ == "__main__":
    asyncio.run(main())
