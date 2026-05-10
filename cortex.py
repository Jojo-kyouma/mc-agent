import asyncio
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
from google import genai
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv

LLM_MODEL_ID = "gemini-3.1-flash-lite"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"

class Priority(Enum):
    IDLE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

class MentalSlot(Enum):
    """Working-memory categories."""
    # Cognition. Using prompt-engineering we ask if these should be populated with new entries.
    SELF_CONCEPT = auto()  # Cognitive understanding of who the agent is
    STRATEGY = auto()      # High-level strategic objectives and vision
    PLAN = auto()          # Flexible short-term planning
    REFLECTION = auto()    # Higher-level insights

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
    reflection: List[str] = field(default_factory=list)
    self_concept: List[str] = field(default_factory=list)
    strategy: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    recalled_memory: Optional[str] = None
    last_recall_query: Optional[str] = None

    # Optimized limits for different cognitive functions
    SLOT_LIMITS = {
        MentalSlot.SOCIAL: 5,      # Deeper conversation history
        MentalSlot.EPISODIC: 15,    # Longer action history for context
        MentalSlot.REFLECTION: 5,   # Insights
        MentalSlot.SELF_CONCEPT: 2, # Core identity/persona (stable)
        MentalSlot.STRATEGY: 2,     # Strategic vision (stable)
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
            MentalSlot.REFLECTION: "reflection",
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
            f"### Physical Status\n{json.dumps(self.status, indent=2)}"
        ]

        mappings = [
            ("Strategy", self.strategy),
            ("Short-Term Step-by-StepPlan", self.plan),
            ("Self-Concept", self.self_concept),
            ("Reflections", self.reflection),
            ("Social Dialogue", self.social),
            ("Activity Log (Critical Feedback for Debugging behaviour scripts)", self.episodic)
        ]

        for header, items in mappings:
            if items:
                context_parts.append(f"### {header}\n- " + "\n- ".join(items))

        if self.recalled_memory:
            context_parts.append(f"### Long-Term Memory Recall\nMatch found for your previous query ('{self.last_recall_query}'):\n- {self.recalled_memory}")

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
    def __init__(self, uri="ws://localhost:8080", db_path="memory.db", actuator_path="bridge.js"):
        self.uri = uri
        self.wm_path = "working_memory.json"
        self.memory = WorkingMemory()
        self.websocket = None
        self.db_path = db_path
        self.actuator_path = actuator_path
        self.thinking_trigger = asyncio.Event()

        self.current_priority = Priority.IDLE
        self.chat_counter = 0
        self.error_counter = 0

        self._load_working_memory()
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

        print(f"[*] Starting Actuator: node {self.actuator_path}")
        self.actuator_process = subprocess.Popen(
            ["node", self.actuator_path],
            stdout=None,
            stderr=None,
            text=True
        )

    async def connect(self):
        max_retries = 5
        for i in range(max_retries):
            try:
                self.websocket = await websockets.connect(self.uri)
                print(f"--- Cortex Linked to Actuator at {self.uri} ---")
                return # Connection successful!
            except (ConnectionRefusedError, OSError):
                print(f"Actuator not ready, retrying ({i+1}/{max_retries})...")
                await asyncio.sleep(1) # Wait 1s before trying again
        
        raise Exception("Could not connect to Node.js Actuator after multiple attempts.")

    """ --- Listen/Send --- """
    async def send_action(self, action: MinecraftAction, source: str = "brain"):
        """Standard method to send an action to the Node.js body."""
        if self.websocket:
            payload = ActionFactory.create_payload(action)
            await self.websocket.send(payload)
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.memory.update_slot(MentalSlot.EPISODIC, f"[{timestamp}] Attempted: {action.description}")

    async def send_abort(self):
        """Sends an immediate abort signal to the actuator."""
        if self.websocket:
            await self.websocket.send(json.dumps({"type": "ABORT"}))

    def _check_interrupt(self, event_priority: Priority, reason: str):
        if event_priority > self.current_priority:
            print(f"[*] RE-PRIORITIZE: Interrupting {self.current_priority.name} for {event_priority.name} ({reason})")
            asyncio.create_task(self.send_abort())
            self.thinking_trigger.set()

    async def listen_to_senses(self):
        """Continuously process updates from the Mineflayer bot."""
        async for message in self.websocket:
            data = json.loads(message)
            if data.get('type') == 'STATUS':
                self.memory.update_slot(MentalSlot.STATUS, data)
                
                # CRITICAL: Health < 10
                if data.get('health', 20) < 10:
                    self._check_interrupt(Priority.CRITICAL, "Low health")
                
                # HIGH: Inventory Full
                inv = data.get('inventory', [])
                if len(inv) >= 36:
                    self._check_interrupt(Priority.HIGH, "Inventory full")

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
                self.current_priority = Priority.IDLE
            elif data.get('type') == 'FINISHED':
                self.thinking_trigger.set()
                self.current_priority = Priority.IDLE
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
                self.chat_counter += 1
                if self.chat_counter >= 3:
                    self._check_interrupt(Priority.MEDIUM, "Multiple chat messages")
            elif data.get('type') == 'ERROR':
                desc, msg = data.get('description'), data.get('message')
                if desc:
                    for i in range(len(self.memory.episodic) - 1, -1, -1):
                        if f"Attempted: {desc}" in self.memory.episodic[i]:
                            self.memory.episodic[i] = self.memory.episodic[i].replace("Attempted:", "Failed:", 1)
                            break
                self.error_counter += 1
                if self.error_counter >= 3:
                    self._check_interrupt(Priority.HIGH, "Frequent feedback errors")

                self.memory.update_slot(MentalSlot.EPISODIC, f"Feedback Error: {msg}")
            elif data.get('type') == 'ITEM_BREAK':
                self._check_interrupt(Priority.HIGH, "Equipment broken")
            self.save_working_memory()

    """ --- Save/Load Working Memory --- """
    def save_working_memory(self):
        """Saves the current working memory to a JSON file so it can be loaded when the agent starts again."""
        try:
            with open(self.wm_path, "w") as f:
                json.dump(asdict(self.memory), f, indent=2)
        except Exception as e:
            print(f"Error saving WM: {e}")

    def _load_working_memory(self):
        """Loads working memory from a JSON file if it exists."""
        if os.path.exists(self.wm_path):
            try:
                with open(self.wm_path, "r") as f:
                    data = json.load(f)
                    self.memory.status = data.get("status", {})
                    self.memory.environment = data.get("environment", {})
                    self.memory.social = data.get("social", [])
                    self.memory.episodic = data.get("episodic", [])
                    self.memory.reflection = data.get("reflection", [])
                    self.memory.self_concept = data.get("self_concept", [])
                    self.memory.strategy = data.get("strategy", [])
                    self.memory.plan = data.get("plan", [])
                    self.memory.recalled_memory = data.get("recalled_memory")
                    self.memory.last_recall_query = data.get("last_recall_query")
            except Exception as e:
                print(f"Error loading WM: {e}")

    """ --- Save/Load Long-Term Memory --- """
    def save_to_memory(self):
        """Save an entry to the long-term memory."""
        # For simplicity, currently we only save the latest reflection as a general memory.
        if not self.memory.reflection:
            return
        
        reflection_content = self.memory.reflection[-1]
        
        # Creating a general record. Currently, it represents just a reflection.
        record = Record(
            content=reflection_content,
            embedding_description=reflection_content, # Using the reflection itself to describe the memory
            embedding=self.embedding_model.encode(reflection_content).tolist()
        )
        
        embedding_blob = json.dumps(record.embedding).encode('utf-8')

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO long_term_memory (id, content, embedding_description, embedding, timestamp) VALUES (?, ?, ?, ?, ?)",
                (record.id, record.content, record.embedding_description, embedding_blob, record.timestamp.isoformat())
            )

    async def recall(self, query: str, threshold: float = 0.4) -> Optional[str]:
        """Finds the most relevant general memory based on a query string."""
        if not query: return None
        
        query_vec = torch.tensor(self.embedding_model.encode(query))
        best_content, max_sim = None, -1.0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT content, embedding FROM long_term_memory")
            for content, embedding_blob in cursor:
                vec = torch.tensor(json.loads(embedding_blob.decode('utf-8')))
                sim = torch.nn.functional.cosine_similarity(query_vec.unsqueeze(0), vec.unsqueeze(0)).item()
                if sim > max_sim:
                    max_sim, best_content = sim, content
        
        return best_content if max_sim >= threshold else None

    async def think(self) -> tuple[Optional[MinecraftAction], Optional[str], str]:
        """
        High-level reasoning cycle. Uses Gemini to process working memory,
        update internal cognitive states, and return a structured action and raw response.
        """
        # Perform recall using the query description provided in the previous interaction
        if self.memory.last_recall_query:
            self.memory.recalled_memory = await self.recall(self.memory.last_recall_query)
        else:
            self.memory.recalled_memory = None

        context = self.memory.to_string()
        prompt = self._build_brain_prompt(context)
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
                "reflection": MentalSlot.REFLECTION,
                "strategy": MentalSlot.STRATEGY,
                "plan": MentalSlot.PLAN
            }
            
            # Update internal priority tracking based on LLM's assessment
            priority_str = res_data.get("priority", "LOW").upper()
            self.current_priority = Priority.__members__.get(priority_str, Priority.LOW)
            self.chat_counter = 0
            self.error_counter = 0

            for key, slot in cognition_map.items():
                self.memory.update_slot(slot, res_data.get(key))
            
            self.memory.last_recall_query = res_data.get("recall_query")
            
            script = res_data.get("behaviour_script")
            description = res_data.get("behaviour_description")
            if script:
                return (MinecraftAction(
                    description=description,
                    content=script
                ), raw_json, prompt)
        except Exception as e:
            print(f"Brain reasoning error: {e}")
        return None, None, prompt

    def _build_brain_prompt(self, context: str):
        system_instr = """
You are the consciousness of an ambitious and efficient autonomous Minecraft agent. You generate JavaScript 'behaviour_script' code to control a Mineflayer bot. 

### OPERATIONAL PHILOSOPHY:
1. **Ambitious Scale**: Aim for high-impact objectives—automate the clearing of entire veins, excavation of areas, or systematic cave exploration. 
2. **Iterative Problem Solving**: If a block or item name is uncertain (e.g., is it 'planks' or 'oak_planks'?), your script MUST programmatically check `bot.registry.itemsByName` or iterate over likely candidates. Exhaust all logic paths internally.

### CAPABILITIES (The 'bot' Object):
The `bot` instance is a standard Mineflayer bot (v1.20.1). You have access to its full API (e.g., `bot.recipesFor`, `bot.inventory`, `bot.findBlocks`, `bot.chat`).
Key extensions and configurations include:
- **Navigation**: `bot.pathfinder` is ready. Move using `await bot.pathfinder.goto(goal)`. Goals (e.g., `GoalNear`) and `Movements` are available at `bot.pathfinder.goals` and `bot.pathfinder.Movements`.
- **Math**: `bot.vec3` library is attached for 3D vector utilities (e.g., `new bot.vec3(x, y, z)`).
- **Feedback**: `bot.recordError(message)` reports script-level logical failures to your episodic memory.
- **Interruption**: Your script is passed a `signal` object. Use `if (signal.aborted) return;` inside loops to check for interruptions.

### CODE GENERATION RULES:
1. **Async Workflow**: Every world interaction (dig, place, move) and every internal async function call MUST be `await`ed.
2. **No External Imports**: Use only the properties of `bot`. Do not use `require`.
3. **Human-like Delay**: Use `await bot.waitForTicks(3)` to `await bot.waitForTicks(5)` after every interaction (dig, place, move, etc.) to mimic human reaction times.
4. **Robustness & Feedback**: Wrap interactions in `try-catch` blocks. For critical dependencies (like crafting tools needed for harvesting), if the action fails, you should `throw` a new error after recording it so the entire script stops rather than proceeding with incorrect assumptions.

### EXAMPLE SCRIPT:
async function gather() { const logNames = ['oak_log', 'birch_log']; const targets = bot.findBlocks({ matching: (block) => logNames.includes(block.name), maxDistance: 64, count: 5 }); if (targets.length === 0) { bot.chat('No logs found.'); } else { for (const pos of targets) { try { await bot.pathfinder.goto(new bot.pathfinder.goals.GoalNear(pos.x, pos.y, pos.z, 2)); await bot.waitForTicks(3); try { const b = bot.blockAt(pos); if (!b || b.name.includes('air')) continue; await bot.dig(b); await bot.waitForTicks(5); } catch (e) { bot.recordError(`Dig failed: ${e.message}`); } } catch (e) { bot.recordError(`Move failed: ${e.message}`); } } bot.chat('Operation complete.'); } } await gather();

### Minecraft-Specific Constraints:
- **Body**: Your physical body is 0.6 blocks wide and 1.8 blocks tall. You occupy this space and cannot place blocks where you are currently standing.
- **Placing Blocks**: Placing a block ocScupies the empty space adjacent to the specific face (Top, Bottom, North, South, East, West) you interact with.
- **Interaction Reach**: You can only mine or place blocks within a 3-block radius of your eye level (1.62 blocks above your feet).
- **Line of Sight**: You cannot interact with blocks through solid walls; a clear "ray" must exist from your eyes to the target face.

### COGNITIVE SNAPSHOT:
As well as a behaviour script, your response includes your self-concept, strategic goal, short-term plan, reflection, and recall query.

### RESPONSE FORMAT:
Respond only in valid JSON.
Respond only in valid JSON. Use Priority levels: LOW, MEDIUM, HIGH, CRITICAL.
{
  "behaviour_script": "Raw JavaScript code string. Use semicolons. No newlines (\\n). Use only the provided 'bot' instance.",
  "behaviour_description": "A concise summary of the behaviour script's purpose.",
  "priority": "The Priority level of this script.",
  "reflection": "Highly open-minded novel stream of insight and reflection, or personal/interpersonal moments worth elaborating on.",
  "strategy": "Your high-level strategic vision.",
  "plan": "Your current step-by-step short-term roadmap (flexible and immediate).",
  "self_concept": "Your core, stable identity and persona.",
  "recall_query": "A description to search your long-term memory for relevant past experiences."
}
"""
        return f"{system_instr}\nCURRENT WORKING MEMORY:\n{context}\n\nAnalyze the current status and provide your response in JSON."

# --- Usage Loop ---
async def main():
    nexus = Cortex()
    nexus.start_actuator()
    await nexus.connect()

    asyncio.create_task(nexus.listen_to_senses())

    nexus.thinking_trigger.set()

    while True:
        # Strictly wait for a signal (Finished action or Error) before thinking again.
        await nexus.thinking_trigger.wait()
        
        nexus.thinking_trigger.clear()
        action, raw_json, prompt = await nexus.think()
        
        if prompt:
            with open("llm_debug.txt", "a", encoding="utf-8") as f:
                f.write(f"\n--- {datetime.now()} ---\n[INPUT]\n{prompt}\n\n[OUTPUT]\n{raw_json or 'Error or No Response'}\n")

        if action:
            await nexus.send_action(action)
            nexus.save_to_memory()

if __name__ == "__main__":
    asyncio.run(main())
