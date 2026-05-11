# Minecraft Autonomous AI Agent

This project implements an autonomous Minecraft agent powered by Google Gemini. It uses a Node.js "bridge" to interface with the game via Mineflayer and a Python "cortex" for high-level reasoning, long-term memory, and cognitive processing.



## Prerequisites

1.  **Minecraft Java Edition**: Version **1.20.1** is required.
2.  **Node.js**: Version 18 or newer.
3.  **Python**: Version 3.11 or newer.
4.  **Google Gemini API Key**: Required for the agent's reasoning capabilities. 

## Installation

### 1. Node.js Dependencies
Navigate to the project root and install the required packages:
```bash
npm install
```

### 2. Python Dependencies
It is recommended to use a virtual environment. You can install the dependencies using `pip` or `uv`:
```bash
pip install .
# OR if using uv
uv sync
```

## Getting Started

1.  **Launch Minecraft**: Open Minecraft version **1.20.1**, start a Singleplayer world.
2.  **Open to LAN**: Press `Esc`, click **Open to LAN**, and note the port number provided in the chat.
3.  **Configure Bridge**: Open `bridge.js` and update `MINECRAFT_PORT` on line 7 with the port from the previous step.
4.  **Set API Key**: Create a `.env` file in the root directory and add your Google API key:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```
5.  **Run the Program**: Start the agent orchestrator:
    ```bash
    python cortex.py [Names or Agent Count]
    ```
    The Python script will automatically spawn the Node.js bridge and begin the cognitive cycle.

## Subjects touched upon

### Functionalist Memory Systems
In alignment with functionalist philosophy, this project treats consciousness and agency as products of informational processing rather than biological substrate. The architecture focuses on the management of two distinct memory layers:
* **Volatile Working Memory:** Real-time environmental data acquisition and immediate spatial awareness (via `scanEnv`).
* **Persistent Long-Term Memory:** State serialization (JSON-based) that ensures a continuous identity across sessions.
* **Significance:** Most LLM implementations are "stateless." This project prototypes an agent that maintains a coherent narrative and state history, solving the problem of informational continuity in autonomous systems.

### Embodied Agency in Simulated Environments
This project serves as a low-cost testbed for **Embodied AI**. While traditional AI exists in a vacuum of text, this agent must navigate the "friction" of a physical world (gravity, collisions, and pathfinding limits).
* **Reasoning-to-Action Translation:** The core challenge addressed is the middleware layer—converting high-level intent (e.g., "secure resources") into precise motor commands (e.g., vector-based movement and block manipulation).
* **Robotics Prototyping:** By simulating the constraints of a humanoid body, the project develops logic for error recovery and adaptation that is directly applicable to real-world humanoid robotics and industrial automation.

### Social Experiment
Project is quite resource-efficient. It should be possible to instantiate 100 bots in the minecraft world, encourage them to be social, and watch them build a civilisation together, all that with only a i7, linux and a headless minecraft server.
