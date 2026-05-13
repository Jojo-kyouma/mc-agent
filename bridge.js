const mineflayer = require('mineflayer');
const mcData = require('minecraft-data')('1.20.1');
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
const WebSocket = require('ws');
const vec3 = require('vec3');

// --- CONFIGURATION ---
const MINECRAFT_PORT = 55580; // Change this to the port shown when you "Open to LAN"
const MC_VERSION = '1.20.1';

// Parse CLI args: node bridge.js [ws_port] [bot_username]
const WS_PORT = parseInt(process.argv[2]) || 8080;
const BOT_USERNAME = process.argv[3] || 'Agent';

const ENVIRONMENT_RADIUS = 3; // Radius for the horizontal scan (e.g., 2 creates a 5x5 area)
const BLOCK_UPDATE_RADIUS = ENVIRONMENT_RADIUS + 4; // Interaction distance (Bot can reach blocks ~4 blocks away)

const bot = mineflayer.createBot({ 
    host: 'localhost',
    port: MINECRAFT_PORT,
    username: BOT_USERNAME,
    auth: 'offline',
    version: '1.20.1'
});
bot.loadPlugin(pathfinder);

const wss = new WebSocket.Server({ port: WS_PORT });
let lastScanPos = null;
let currentAbortController = null;

wss.on('connection', (ws) => {
    const _stopPhysicalActions = () => {
        bot.pathfinder.setGoal(null);
        if (bot.targetDigBlock) bot.stopDigging();
    };
    const abortCurrentScript = () => {
        if (currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
        }
        _stopPhysicalActions();
    };

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

                // --- API for behavior scripts ---
                bot.recordError = (msg) => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: 'ERROR', message: msg, description: data.description }));
                    }
                };
                bot.recordSuccess = () => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: 'SUCCESS', description: data.description }));
                    }
                };
                bot.recordInfo = (msg) => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: 'INFO', message: msg, description: data.description }));
                    }
                };
                bot.placeBlockSafe = async (referenceBlock, faceVector) => {
                    if (!referenceBlock) {
                        bot.recordError("Cannot place block: referenceBlock is null.");
                        return;
                    }
                    const targetPos = referenceBlock.position.add(faceVector);
                    
                    if (bot.entity.position.distanceSquared(targetPos) < 1.5) {
                        bot.recordInfo(`Moving to clear target space at ${targetPos.floored()}`);
                        
                        const awayPos = bot.entity.position.add(new Vec3(-1, 0, 1)); 
                        const goal = new GoalNear(awayPos.x, awayPos.y, awayPos.z, 1);
                        
                        await bot.pathfinder.goto(goal);
                    }
                    if (signal.aborted) return;
                    
                    try {
                        await bot.placeBlock(referenceBlock, faceVector);
                    } catch (e) {
                        bot.recordError(`Placement failed: ${e.message}`);
                    }
                };
                bot.findIds = (query) => {
                    return Object.values(mcData.items)
                        .filter(i => i.name.includes(query))
                        .map(i => i.id);
                };

                // --- Execute the behavior script ---
                try {
                    const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
                    const wrappedScript = `
                        try {
                            ${data.behaviour_script}
                            bot.recordSuccess();
                        } catch (e) {
                            bot.recordError(e.message);
                            throw e;
                        }
                    `;
                    const execute = new AsyncFunction('bot', 'vec3', 'Vec3', 'mcData', 'GoalNear', 'signal', wrappedScript);
                    const scriptPromise = execute(bot, vec3, vec3, mcData, goals.GoalNear, signal);
                    
                    const abortPromise = new Promise((_, reject) => {
                        signal.addEventListener('abort', () => reject(new Error('Script aborted')), { once: true });
                    });

                    await Promise.race([scriptPromise, abortPromise]);

                } catch (scriptErr) {
                    if (scriptErr.message !== 'Script aborted') {
                        console.error(`[Script Error]: ${scriptErr}`);
                        ws.send(JSON.stringify({ type: 'ERROR', message: scriptErr.message }));
                    }
                } finally {
                    if (!signal.aborted) {
                        ws.send(JSON.stringify({ type: 'FINISHED' }));
                    }
                }
            }
        } catch (err) {
            console.log("Error parsing command message:", err);
        }
    });

    // --- Event Listeners ---
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

    const sendStatus = () => {
        const inventoryMap = bot.inventory.items().reduce((acc, item) => {
            acc[item.name] = (acc[item.name] || 0) + item.count;
            return acc;
        }, {});

        const status = {
            health: Math.round(bot.health),
            food: bot.food,
            saturation: bot.foodSaturation,
            inventory: inventoryMap, 
            onFire: (bot.entity.metadata[0] & 0x01) !== 0,
            heldItem: bot.heldItem ? { name: bot.heldItem.name, count: bot.heldItem.count } : null,
            position: bot.entity.position.floored()
        };
        ws.send(JSON.stringify(status));
    };
    bot.on('health', sendStatus);
    bot.on('updateSlot', sendStatus);

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

        const actualRadius = (typeof radius === 'number' && radius > 0) 
            ? radius 
            : ENVIRONMENT_RADIUS;
        const currentPos = bot.entity.position;

        if (!force && lastScanPos) {
            const dx = Math.abs(currentPos.x - lastScanPos.x);
            const dz = Math.abs(currentPos.z - lastScanPos.z);
            const dy = currentPos.y - lastScanPos.y;
            if (dx <= actualRadius && dz <= actualRadius && dy >= -1 && dy <= 2) return;
        }

        const botPos = currentPos.floored(); 
        const uniqueBlocks = new Set();

        for (let y = -1; y <= 2; y++) {
            for (let x = -actualRadius; x <= actualRadius; x++) {
                for (let z = -actualRadius; z <= actualRadius; z++) {
                    const block = bot.blockAt(botPos.offset(x, y, z));
                    if (block && !['air', 'cave_air', 'void_air'].includes(block.name)) {
                        uniqueBlocks.add(block.name);
                    }
                }
            }
        }

        lastScanPos = currentPos.clone();
        const blockList = [...uniqueBlocks];

        ws.send(JSON.stringify({ 
            type: 'ENVIRONMENT', 
            blocks: blockList
        }));
    };
    bot.on('move', sendEnvironment);

    const onBlockUpdate = (oldBlock, newBlock) => {
        if (ws.readyState !== WebSocket.OPEN || !bot.entity) return;
        if (newBlock.position.distanceTo(bot.entity.position) > BLOCK_UPDATE_RADIUS) return;
        sendEnvironment(ENVIRONMENT_RADIUS, true);
    };
    bot.on('blockUpdate', onBlockUpdate);

    // Ensure world is loaded and entity is initialized before sending initial sync 
    (async () => {
        console.log("Cortex linked. Waiting for world data to load...");
        while (!bot.entity || !bot.entity.position || !bot.blockAt(bot.entity.position)) {
            if (ws.readyState !== WebSocket.OPEN) return;
            await bot.waitForTicks(5);
        }
        sendStatus();
        sendEnvironment(ENVIRONMENT_RADIUS, true);
    })();

    ws.on('close', () => {
        console.log("Cortex disconnected from WebSocket.");

        bot.removeListener('chat', onChat);
        bot.removeListener('blockUpdate', onBlockUpdate);
    });

});