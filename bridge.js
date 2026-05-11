const mineflayer = require('mineflayer');
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
const WebSocket = require('ws');
const vec3 = require('vec3');

// --- CONFIGURATION ---
const MINECRAFT_PORT = 60802; // Change this to the port shown when you "Open to LAN"

// Parse CLI args: node bridge.js [ws_port] [bot_username]
const WS_PORT = parseInt(process.argv[2]) || 8080;
const BOT_USERNAME = process.argv[3] || 'Agent';

const ENVIRONMENT_RADIUS = 2; // Radius for the horizontal scan (e.g., 2 creates a 5x5 area)
const BLOCK_UPDATE_RADIUS = ENVIRONMENT_RADIUS + 4; // Interaction distance (Bot can reach blocks ~4 blocks away)

const bot = mineflayer.createBot({ 
    host: 'localhost',
    port: MINECRAFT_PORT,
    username: BOT_USERNAME,
    auth: 'offline',
    version: '1.20.1'
});
bot.loadPlugin(pathfinder);

const getItemName = (entity) => {
    if (entity.name !== 'item') return entity.name || entity.username || 'unknown';
    try {
        const itemData = entity.metadata[8];
        if (itemData && itemData.itemId !== undefined) {
            const item = bot.registry.items[itemData.itemId];
            return item ? item.name : 'dropped_item';
        }
    } catch (e) {}
    return 'dropped_item';
};

bot.once('inject_allowed', () => {
    if (bot.pathfinder) {
        const defaultMovements = new Movements(bot);
        defaultMovements.allowDig = false;
        defaultMovements.allow1by1towers = false;
        defaultMovements.allowSprinting = false; // Mimic regular human walking
        defaultMovements.scafoldingBlocks = [];
        bot.pathfinder.setMovements(defaultMovements);

        bot.pathfinder.goals = goals;
        bot.pathfinder.Movements = Movements;
        bot.vec3 = vec3;

        console.log("Pathfinder namespaces successfully attached to bot.");

        bot.pathfinder.goto = async (goal) => {
            return new Promise((resolve, reject) => {
                let completed = false;

                const cleanup = (error = null) => {
                    if (completed) return;
                    completed = true;

                    bot.removeListener('goal_reached', onGoalReached);
                    bot.removeListener('path_update', onPathUpdate);
                    bot.removeListener('path_stop', onPathStop);
                    bot.removeListener('death', onDeath);
                    bot.removeListener('kicked', onKicked);
                    bot.removeListener('end', onEnd);

                    bot.clearControlStates();

                    if (error) {
                        bot.pathfinder.setGoal(null);
                        reject(new Error(error));
                    } else {
                        resolve();
                    }
                };

                const onGoalReached = () => cleanup();
                const onPathStop = () => cleanup('Pathfinding stopped.');
                const onPathUpdate = (res) => {
                    if (res.status === 'noPath') cleanup('No path found to goal.');
                    else if (res.status === 'timeout') cleanup('Pathfinding search timed out.');
                    else if (res.status === 'stuck') cleanup('Bot is stuck while pathfinding.');
                };

                const onDeath = () => cleanup('Bot died while pathfinding.');
                const onKicked = (reason) => cleanup(`Bot was kicked: ${reason}`);
                const onEnd = () => cleanup('Bot disconnected.');

                bot.on('goal_reached', onGoalReached);
                bot.on('path_update', onPathUpdate);
                bot.on('path_stop', onPathStop);
                bot.on('death', onDeath);
                bot.on('kicked', onKicked);
                bot.on('end', onEnd);

                bot.pathfinder.setGoal(goal);
            });
        };
    }
});

const wss = new WebSocket.Server({ port: WS_PORT });

// Mineflayer bot error handling
bot.on('error', err => console.error(`[Mineflayer Bot Error]: ${err}`));
bot.on('kicked', reason => console.log(`[Mineflayer Bot Kicked]: ${reason}`));
bot.on('end', reason => console.log(`[Mineflayer Bot Disconnected]: ${reason}`));
wss.on('error', err => console.error(`[WebSocket Server Error]: ${err}`));

let lastScanPos = null;
let currentAbortController = null;

wss.on('connection', (ws) => {
    lastScanPos = null;

    const stopPhysicalActions = () => {
        bot.pathfinder.setGoal(null);
        if (bot.targetDigBlock) bot.stopDigging();
    };

    const abortCurrentScript = () => {
        if (currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
        }
        stopPhysicalActions();
    };

    console.log("Cortex connected.");

    ws.on('message', async (message) => {
        try {
            const data = JSON.parse(message);
            if (data.type === 'ABORT') {
                abortCurrentScript();
            } else if (data.type === 'ACTION' && data.behaviour_script) {

                console.log(`[*] Executing: ${data.description}`);
                
                abortCurrentScript();
                currentAbortController = new AbortController();
                const { signal } = currentAbortController;

                // Allow scripts to report internal errors to episodic memory
                bot.recordError = (msg) => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: 'ERROR', message: msg, description: data.description }));
                    }
                };

                try {
                    const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
                    // Pass signal to the behavior script so it can check signal.aborted
                    const execute = new AsyncFunction('bot', 'vec3', 'GoalNear', 'Movements', 'goals', 'signal', data.behaviour_script);
                    
                    const scriptPromise = execute(bot, vec3, goals.GoalNear, Movements, goals, signal);
                    const abortPromise = new Promise((_, reject) => {
                        signal.addEventListener('abort', () => reject(new Error('Script aborted')), { once: true });
                    });

                    await Promise.race([scriptPromise, abortPromise]);

                    if (!signal.aborted) {
                        ws.send(JSON.stringify({ type: 'SUCCESS', description: data.description }));
                    }
                } catch (scriptErr) {
                    if (scriptErr.message !== 'Script aborted') {
                        console.error(`[Script Error]: ${scriptErr}`);
                        ws.send(JSON.stringify({ type: 'ERROR', message: scriptErr.message, description: data.description }));
                    }
                } finally {
                    if (!signal.aborted) {
                        ws.send(JSON.stringify({ type: 'FINISHED', description: data.description }));
                    }
                }
            }
        } catch (err) {
            console.log("Error parsing command message:", err);
        }
    });
    
    const onBlockUpdate = (oldBlock, newBlock) => {
        if (ws.readyState !== WebSocket.OPEN || !bot.entity) return;
        if (newBlock.position.distanceTo(bot.entity.position) > BLOCK_UPDATE_RADIUS) return;

        const position = newBlock.position;
        const isAir = ['air', 'cave_air', 'void_air'].includes(newBlock.name);
        
        const payload = { type: 'BLOCK_UPDATE', position };
        if (!isAir) payload.block = { name: newBlock.name, position };

        ws.send(JSON.stringify(payload));
    };

    /*
    bot.on('blockUpdate', onBlockUpdate);
    */

    const sendStatus = () => {
        const status = {
            type: 'STATUS',
            health: bot.health,
            food: bot.food,
            saturation: bot.foodSaturation,
            inventory: bot.inventory.items().map(item => ({
                name: item.name,
                count: item.count,
                slot: item.slot
            })),
            onFire: (bot.entity.metadata[0] & 0x01) !== 0,
            heldItem: bot.heldItem ? { name: bot.heldItem.name, count: bot.heldItem.count } : null,
            position: bot.entity.position
        };
        ws.send(JSON.stringify(status));
    };

    bot.on('itemBreak', (item) => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ITEM_BREAK', item: item.name }));
        }
    });

    bot.on('entityHurt', (entity) => {
        if (entity === bot.entity && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'AGENT_ATTACKED' }));
        }
    });

    bot.on('health', sendStatus);
    bot.on('playerCollect', sendStatus);

    const onChat = (username, message) => {
        if (username === bot.username) return;
        let processedMessage = message;
        if (processedMessage.endsWith(']') && !processedMessage.startsWith('[')) {
            processedMessage = processedMessage.slice(0, -1);
        }
        ws.send(JSON.stringify({ type: 'CHAT', username: username, message: processedMessage }));
    };
    bot.on('chat', onChat);

    const sendEnvironment = (radius, force = false) => {
        if (ws.readyState !== WebSocket.OPEN || !bot.entity) return;

        // Ensure radius defaults to ENVIRONMENT_RADIUS if not a positive number
        // This handles cases where radius is undefined, 0, or an event object.
        const actualRadius = (typeof radius === 'number' && radius > 0) 
            ? radius 
            : ENVIRONMENT_RADIUS;
        const currentPos = bot.entity.position;

        if (!force && lastScanPos) {
            const dx = Math.abs(currentPos.x - lastScanPos.x);
            const dz = Math.abs(currentPos.z - lastScanPos.z);
            const dy = currentPos.y - lastScanPos.y;
            // Only scan if the bot has left the previous horizontal area or the vertical slice
            if (dx <= actualRadius && dz <= actualRadius && dy >= -1 && dy <= 2) return;
        }

        const botPos = currentPos.floored();
        const blocks = [];

        // Scan 4 layers vertically within the defined horizontal radius
        for (let y = -1; y <= 2; y++) {
            for (let x = -actualRadius; x <= actualRadius; x++) {
                for (let z = -actualRadius; z <= actualRadius; z++) {
                    const block = bot.blockAt(botPos.offset(x, y, z));
                    if (block && block.name !== 'air' && block.name !== 'cave_air' && block.name !== 'void_air') {
                        blocks.push({ 
                            name: block.name, 
                            position: block.position 
                        });
                    }
                }
            }
        }
        
        const entities = Object.values(bot.entities)
            .filter(e => e !== bot.entity && e.position.distanceTo(bot.entity.position) < 16)
            .map(e => ({
                id: e.id,
                name: getItemName(e),
                dist: Math.round(e.position.distanceTo(bot.entity.position)),
                position: e.position
            }))
            .sort((a, b) => a.dist - b.dist)
            .slice(0, 5);

        // Prevent sending empty environments caused by unloaded chunks unless forced.
        // Standard Minecraft worlds always have blocks (floor) nearby.
        if (!force && blocks.length === 0 && entities.length === 0) return;

        lastScanPos = currentPos.clone();
        ws.send(JSON.stringify({ type: 'ENVIRONMENT', entities, blocks }));
    };

    /* 
    bot.on('move', sendEnvironment);
    bot.on('spawn', sendEnvironment);
    */

    const onEntityUpdate = (entity) => {
        if (ws.readyState !== WebSocket.OPEN || !bot.entity || entity === bot.entity) return;
        
        const isGone = !bot.entities[entity.id];
        if (!isGone && entity.position.distanceTo(bot.entity.position) > BLOCK_UPDATE_RADIUS) return;

        const payload = { type: 'ENTITY_UPDATE', id: entity.id };
        if (!isGone) {
            payload.entity = {
                id: entity.id,
                name: getItemName(entity),
                dist: Math.round(entity.position.distanceTo(bot.entity.position)),
                position: entity.position
            };
        }
        ws.send(JSON.stringify(payload));
    };

    /*
    bot.on('entitySpawn', onEntityUpdate);
    bot.on('entityGone', onEntityUpdate);
    */

    ws.on('close', () => {
        console.log("Cortex disconnected from WebSocket.");
        // Cleanup listeners to prevent memory leaks and redundant processing
        bot.removeListener('blockUpdate', onBlockUpdate);
        bot.removeListener('entitySpawn', onEntityUpdate);
        bot.removeListener('entityGone', onEntityUpdate);
        bot.removeListener('health', sendStatus);
        bot.removeListener('playerCollect', sendStatus);
        bot.removeListener('chat', onChat);
        bot.removeListener('move', sendEnvironment);
        bot.removeListener('spawn', sendEnvironment);
    });

});

// Resource cleanup on exit
process.on('SIGTERM', () => {
    console.log('Stopping actuator...');
    bot.quit();
    wss.close();
    process.exit(0);
});