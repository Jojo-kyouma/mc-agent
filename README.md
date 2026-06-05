## Discontinued
During the development of this project, a few lessons were learned about autonomous agents:
* What seems like it "might" work will not work, unless you have Von Neumann intuition. Next iteration of this project will attempt a more complete replication of human cognition, or other ingenious approach.
* Sequences are hard and fragile. If gemini action is a sequence, verification and behaviour trees might make it robust, but it is a shot in the dark. I dont have faith in sequences.
* Hierarchical learning is more robust. Every action is atomic, and we reuse successful previous scripts from long_term memory.
* Autonomy is hard. Agent doesnt have a drive, a will to attain aestetically pleasing and elegant solutions. Only reasons. This lack of drive is troublesome because his outcome, like building a ugly house might pass to the agent as great. Adding personality entries to working memory did give him indication of personal drive. He expressed disgust with lack of progress according to working memory action log. That single thing is promising, but overall I witness little autonomy and am skeptical.

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

### Agentic AI
The minecraft environment can be used to explore limitations/opportunities of Agentic AI. Agentic AI might primarily interact with APIs, and that is precisely what this agent does.
