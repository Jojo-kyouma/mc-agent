import asyncio
import sys
import subprocess
import os
from transformers import data
import websockets
import json
import sqlite3
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
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
TOP_K_RECALL = 3
DUPLICATE_THRESHOLD = 0.85

# --- Data Structures ---
class MentalSlot(Enum):
    """Working-memory categories."""
    SELF_CONCEPT = auto()  # Cognitive understanding of who the agent is
    STRATEGY = auto()      # High-level strategic objectives
    PLAN = auto()          # Flexible short-term planning

    # Automatically updated.
    STATUS = auto()        # Current health, hunger, inventory, and minecraft-specific player-status information
    ENVIRONMENT = auto()   # Data structure of surrounding blocks
    SOCIAL = auto()        # Recent chat messages and the player who said them
    EPISODIC = auto()      # Recent actions and their outcomes (success, failure, info)

@dataclass
class WorkingMemory:
    """Maintains the working memory of the agent. As list grows, older items are removed."""
    status: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    social: List[str] = field(default_factory=list)
    episodic: List[str] = field(default_factory=list)
    self_concept: List[str] = field(default_factory=list)
    strategy: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    recalled_memories: List[str] = field(default_factory=list)
    last_recall_query: Optional[str] = None

    SLOT_LIMITS = {
        MentalSlot.SOCIAL: 5,
        MentalSlot.EPISODIC: 15,
        MentalSlot.SELF_CONCEPT: 1,
        MentalSlot.STRATEGY: 1,
        MentalSlot.PLAN: 1
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

        attr_map = {
            MentalSlot.SELF_CONCEPT: "self_concept",
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
            f"### Physical Status/Inventory\n{json.dumps(self.status, indent=2)}",
            f"### Environment\n{json.dumps(self.environment, indent=2)}"
        ]

        mappings = [
            ("Strategy", self.strategy),
            ("Short-Term Step-by-StepPlan", self.plan),
            ("Self-Concept", self.self_concept),
            ("Social Dialogue", self.social),
            ("Activity Log (With Feedback)", self.episodic)
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
    content: Any
    embedding_description: str
    embedding: List[float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

class MinecraftAction(BaseModel):
    """Helper. Bridge between high-level action concepts and the JavaScript behaviour scripts executed by the Node.js actuator."""
    description: str = ""
    content: Any = None

class ActionFactory:
    """Helper. Translates high-level Action objects into JSON payloads."""
    @staticmethod
    def create_payload(action: MinecraftAction) -> str:
        payload = {
            "type": "ACTION",
            "description": action.description,
            "behaviour_script": action.content
        }
        return json.dumps(payload)

# --- Runtime ---
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

    async def connect(self):
        max_retries = 10
        for i in range(max_retries):
            try:
                self.websocket = await websockets.connect(
                    self.uri, ping_interval=20, ping_timeout=60
                )
                print(f"--- {self.agent_name} Cortex Linked to Actuator at {self.uri} ---")
                return
            except (ConnectionRefusedError, OSError):
                print(f"Actuator not ready, retrying ({i+1}/{max_retries})...")
                await asyncio.sleep(1)
        
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

    async def send_action(self, action: MinecraftAction):
        """Standard method to send an action to the Node.js body."""
        if self.websocket:
            self.priority_accumulator = 0
            payload = ActionFactory.create_payload(action)
            await self.websocket.send(payload)
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.memory.update_slot(MentalSlot.EPISODIC, f"[{timestamp}] Attempt: {action.description}")

    async def listen_to_senses(self):
        """Continuously process updates from the Mineflayer bot."""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                if not isinstance(data, dict):
                    continue
                if data.get('type') == 'STATUS':
                    data.pop('type', None)
                    self.memory.update_slot(MentalSlot.STATUS, data)  
                    if data.get('onFire'):
                        self._handle_priority(4, "Agent is on fire")
                    if data.get('food', 20) < 15:
                        self._handle_priority(2, "Agent is hungry")
                    if data.get('inventoryUsed', 0) >= 36:
                        self._handle_priority(3, "Inventory is full")

                elif data.get('type') == 'ENVIRONMENT':
                    data.pop('type', None)
                    self.memory.update_slot(MentalSlot.ENVIRONMENT, data)

                elif data.get('type') == 'INFO':
                    msg = data.get('message')
                    self.memory.update_slot(MentalSlot.EPISODIC, f"INFO: {msg}")

                elif data.get('type') == 'SUCCESS':
                    desc = data.get('description')
                    for i in range(len(self.memory.episodic) - 1, -1, -1):
                        if f"Attempt: {desc}" in self.memory.episodic[i]:
                            self.memory.episodic[i] = self.memory.episodic[i].replace("Attempt:", "SUCCESS:", 1)
                            break
                    if self.memory.plan:
                        self.memory.plan.pop()

                elif data.get('type') == 'FINISHED':
                    self.thinking_trigger.set()

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
                            if f"Attempt: {desc}" in self.memory.episodic[i]:
                                self.memory.episodic[i] = self.memory.episodic[i].replace("Attempt:", "FAILED:", 1)
                                break
                    self.memory.update_slot(MentalSlot.EPISODIC, f"ERROR: {msg}")
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
                    if not isinstance(data, dict):
                        return

                    def as_list(key):
                        val = data.get(key, [])
                        if isinstance(val, list): return val
                        return [val] if val else []

                    self.memory.status = data.get("status", {})
                    self.memory.environment = data.get("environment", {})
                    self.memory.social = as_list("social")
                    self.memory.episodic = as_list("episodic")
                    self.memory.self_concept = as_list("self_concept")
                    self.memory.strategy = as_list("strategy")
                    self.memory.plan = as_list("plan")
                    self.memory.recalled_memories = as_list("recalled_memories")
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
    def _find_duplicate_id(self, embedding_key: str) -> Optional[str]:
        """Checks if a similar record exists for the given search trigger and returns its ID."""
        if not embedding_key: return None
        
        new_vec = torch.tensor(self.embedding_model.encode(embedding_key))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id, embedding FROM long_term_memory")
            for rec_id, embedding_blob in cursor:
                vec = torch.tensor(json.loads(embedding_blob.decode('utf-8')))
                sim = torch.nn.functional.cosine_similarity(new_vec.unsqueeze(0), vec.unsqueeze(0)).item()
                if sim >= DUPLICATE_THRESHOLD:
                    return rec_id
        return None

    def save_to_memory(self, memory_data: Optional[Dict[str, str]]):
        """Save an entry to the long-term memory using an embedding anchor for better recall."""
        if not memory_data:
            return
        
        content = memory_data.get("to_save")
        embedding_key = memory_data.get("embedding_key")
        
        if not content or not embedding_key:
            return

        dup_id = self._find_duplicate_id(embedding_key)
        record = Record(
            content=content,
            embedding_description=embedding_key,
            embedding=self.embedding_model.encode(embedding_key).tolist()
        )
        embedding_blob = json.dumps(record.embedding).encode('utf-8')
        with sqlite3.connect(self.db_path) as conn:
            if dup_id:
                conn.execute(
                    "UPDATE long_term_memory SET content = ?, embedding_description = ?, embedding = ?, timestamp = ? WHERE id = ?",
                    (record.content, record.embedding_description, embedding_blob, record.timestamp.isoformat(), dup_id)
                )
            else:
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
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in results[:TOP_K_RECALL]]

    async def think(self) -> tuple[Optional[MinecraftAction], Optional[str], str, Optional[dict]]:
        """
        High-level reasoning cycle. Uses Gemini to process working memory,
        update internal cognitive states, and return a structured action and raw response.
        """
        if self.memory.last_recall_query:
            self.memory.recalled_memories = await self.recall(self.memory.last_recall_query)
        else:
            self.memory.recalled_memories = []

        context = self.memory.to_string()
        prompt = self._build_brain_prompt(context)

        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=LLM_MODEL_ID,
                    contents=prompt,
                    config={'response_mime_type': 'application/json'}
                )
                raw_json = response.text
                res_data = json.loads(raw_json)
                
                # Update Cognition
                cognition = res_data.get("cognition", {})
                self.memory.update_slot(MentalSlot.SELF_CONCEPT, cognition.get("self_concept"))
                self.memory.update_slot(MentalSlot.STRATEGY, cognition.get("strategy"))
                self.memory.update_slot(MentalSlot.PLAN, cognition.get("plan"))
                
                # Update Memory search state
                memory_data = res_data.get("memory", {})
                self.memory.last_recall_query = memory_data.get("recall_query")
                
                self.save_working_memory()
                
                # Extract behavior
                behaviour = res_data.get("behaviour", {})
                script = behaviour.get("script")
                description = behaviour.get("description")
                
                if script:
                    return (MinecraftAction(description=description, content=script), raw_json, prompt, memory_data)
                break
            except Exception as e:
                err_str = str(e)
                # Detect 503 or overload errors specifically
                if "503" in err_str or "overloaded" in err_str.lower():
                    if attempt < max_retries - 1:
                        print(f"[*] Gemini is under high demand (503). Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                
                print(f"Brain reasoning error: {e}")
                break

        return None, None, prompt, None

    def _build_brain_prompt(self, context: str):
        system_instr = """
You are an autonomous Minecraft Agent using a limited set of Mineflayer and related libraries.

### GUIDELINES:
- **Ambition**: Aim for high-level tasks, e.g. "clear a forest", "build a house", or "excavate a mine". If Working Memory is insufficient, start small and build up.
- **Resposive**: Frequently use `if (signal.aborted) return;` to make Agent responsive to new priorities.
- **Feedback**: When you see ERROR in Activity Log, use `bot.recordInfo(message)` to return diagnostic findings to your episodic memory.
- **Asynchronous**: Use `await` for all interactions with the bot to ensure proper sequencing and responsiveness.
- **Working Memory Awareness**: Pay especial attention to tags ERROR and INFO in Activity log and information under Environment and Inventory when evaluating outcome of your actions.
- **Problem Solving**: If you encounter an error, be highly open-minded when trying to solve it, within the API constraints. 

### SCRIPT INTERFACE REFERENCE:
  ONLY use the following APIs. Under no circumstances ever should you attempt to use APIs not listed here.
- `new Vec3(x, y, z)`: `.add/minus(v)`, `.scaled(n)`, `.unit()`, `.distanceTo/Squared(v)`, `.floored()`
- `mcData`: Knowledge base (e.g. `mcData.blocksByName['oak_log'].id`).
- Available IDs: '_log', '_planks', '_pickaxe', '_axe', '_shovel', '_sword', '_door', '_button', '_pressure_plate', 'cooked_', '_ingot', 'diamond', 'coal', 'cobblestone', 'dirt', 'sand', 'gravel', 'flint_and_steel', 'bucket', 'torch', 'crafting_table', 'furnace', 'chest', 'redstone_dust', 'lever', 'piston'. IDs listed in Inventory can also be used.
- `GoalNear`: `await bot.pathfinder.goto(new GoalNear(x, y, z, range))`.
- `bot`: `inventory.items()`, `await equip(item, slot)`, `findBlock({matching, maxDistance})`, `await attack(entity)`, `chat(msg)`, `lookAt(vec)`, `findIds(_planks)`, `await digSafe(b)`, `await placeBlockSafe(r, f)`, `await activateBlockSafe(b)`.
- **Block Interaction**: Use `await bot.digSafe(block)`, `await bot.placeBlockSafe(refBlock, faceVector)`, and `await bot.activateBlockSafe(block)`.
- **Crafting**: 3x3 recipes REQUIRE a `craftingTableBlock`. 
  Example: `const recipe = bot.recipesFor(id, null, 1, table)[0]; if (recipe) await bot.craft(recipe, 1, table);`
- **Interaction Safety**: `bot.digSafe`, `bot.placeBlockSafe`, and `bot.activateBlockSafe` include distance and visibility checks, and automatic look-at. You must be within 4.5 blocks and have a clear line of sight to the target block. If these conditions are not met, an error will be thrown.
  
### RESPONSE FORMAT (JSON):
{
  "behaviour": {
    "script": "JS code. No literal newlines.",
    "description": "Detailed description of the script that can be used for debugging."
  },
  "cognition": {
    "self_concept": "Core persona.",
    "strategy": "Strategic vision.",
    "plan": "Step-by-step roadmap. 1. 2. 3. format"
  },
  "memory": {
    "to_save": "Important facts or findings to persist in long-term memory. E.g. someone's birthday, location of a village, solution to a problem you solved.",
    "embedding_key": "A search term (e.g. 'how to fish') that should trigger this memory in the future.",
    "recall_query": "Search term to use in your LTM search during the NEXT cycle."
  }
}
"""
        return f"{system_instr}\nCURRENT WORKING MEMORY:\n{context}\n\nAnalyze status and provide JSON response."

    async def run(self):
        """Single cognitive lifecycle for a single agent."""
        print(f"[*] Initializing cognitive loop for {self.agent_name}...")
        self._load_working_memory()
        
        try:
            await self.start_cycle()
        except Exception as e:
            print(f"[!] {self.agent_name} execution stopped: {e}")

    async def start_cycle(self):
        """Starts the actuator and cognitive processes."""
        self.start_actuator()
        await self.connect()
        
        listener_task = asyncio.create_task(self.listen_to_senses())
        self.thinking_trigger.set()
        
        # Main reasoning loop
        async def reasoning_loop():
            while True:
                await self.thinking_trigger.wait()
                self.thinking_trigger.clear()
                
                action, raw_json, prompt, memory_data = await self.think()
                
                if prompt:
                    log_path = os.path.join(self.log_dir, f"{self.agent_name}.txt")
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"\n--- {datetime.now()} ---\n[INPUT]\n{prompt}\n\n[OUTPUT]\n{raw_json or 'Error'}\n")

                if action:
                    await self.send_action(action)
                    self.save_to_memory(memory_data)

        try:
            await asyncio.gather(listener_task, reasoning_loop())
        finally:
            listener_task.cancel()

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

"""
NOTE:
You can gradually reintroduce Autopilot after understanding the project much better.
Conscious behaviour is already being witnessed.
"""
