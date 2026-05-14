const mineflayer = require('mineflayer');
const mcData = require('minecraft-data')('1.20.1');
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
const WebSocket = require('ws');
const Vec3 = require('vec3');

// --- CONFIGURATION ---
const MINECRAFT_PORT = 51674; // Change this to the port shown when you "Open to LAN"
const MC_VERSION = '1.20.1';

// Parse CLI args: node bridge.js [ws_port] [bot_username]
const WS_PORT = parseInt(process.argv[2]) || 8080;
const BOT_USERNAME = process.argv[3] || 'Agent';

const ENVIRONMENT_RADIUS = 10; // Radius for the horizontal scan (e.g., 2 creates a 5x5 area)
const BLOCK_UPDATE_RADIUS = ENVIRONMENT_RADIUS + 4; // Interaction distance (Bot can reach blocks ~4 blocks away)

const AIR_BLOCKS = new Set(['air', 'cave_air', 'void_air']);

const getItemName = (entity) => {
    if (entity.type === 'player') return entity.username;
    if (entity.name) return entity.name;
    return entity.type;
};

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
let nearbyBlocksCount = new Map(); // name -> count
let syncTimeout = null;
let currentAbortController = null;

wss.on('connection', (ws) => {
    const abortCurrentScript = () => {
        if (currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
        }
        bot.pathfinder.setGoal(null);
        if (bot.targetDigBlock) {
            bot.stopDigging();
        }
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

                // --- Contextual API & Safety Layer ---
                const validate = (target, action) => {
                    if (!target || !target.position) return;
                    const dist = bot.entity.position.distanceTo(target.position);
                    if (dist > 4.5) throw new Error(`${action}: Target too far (${dist.toFixed(1)}m). Move closer.`);
                    if (action === 'dig' && !bot.canSeeBlock(target)) throw new Error(`${action}: Target not visible. You need a clear line of sight to dig.`);
                };

                bot.digSafe = async (b) => { if (signal.aborted) throw new Error('Script aborted'); await bot.lookAt(b.position.offset(0.5, 0.5, 0.5)); validate(b, 'dig'); return await bot.dig(b); };
                bot.activateBlockSafe = async (b, ...a) => { if (signal.aborted) throw new Error('Script aborted'); await bot.lookAt(b.position.offset(0.5, 0.5, 0.5)); validate(b, 'activateBlock'); return await bot.activateBlock(b, ...a); };
                bot.placeBlockSafe = async (ref, face) => {
                    if (signal.aborted) throw new Error('Script aborted');
                    validate(ref, 'placeBlock');
                    const targetPos = ref.position.add(face);
                    if (bot.entity.position.distanceSquared(targetPos) < 2.25) {
                        const away = bot.entity.position.offset(-1, 0, 1);
                        await bot.pathfinder.goto(new goals.GoalNear(away.x, away.y, away.z, 1));
                    }
                    if (signal.aborted) throw new Error('Script aborted');
                    return await bot.placeBlock(ref, face);
                };

                bot.recordError = (m) => ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify({ type: 'ERROR', message: m, description: data.description }));
                bot.recordSuccess = () => ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify({ type: 'SUCCESS', description: data.description }));
                bot.recordInfo = (m) => ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify({ type: 'INFO', message: m, description: data.description }));

                bot.findIds = (query) => {
                    return Object.values(mcData.items)
                        .filter(i => i.name.includes(query))
                        .map(i => i.id);
                };

                // --- Execute the behavior script ---
                try {
                    const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
                    const execute = new AsyncFunction('bot', 'Vec3', 'mcData', 'GoalNear', 'signal', data.behaviour_script);
                    const scriptPromise = execute(bot, Vec3, mcData, goals.GoalNear, signal);
                    
                    const abortPromise = new Promise((resolve, reject) => {
                        signal.addEventListener('abort', () => reject(new Error('Script aborted')), { once: true });
                    });

                    await Promise.race([scriptPromise, abortPromise]);

                } catch (scriptErr) {
                    const isAbort = signal.aborted || 
                                    scriptErr.message === 'Script aborted' || 
                                    scriptErr.message === 'Digging aborted' || 
                                    scriptErr.message === 'Goal cancelled';

                    if (!isAbort) {
                        console.error(`[Script Error]: ${scriptErr}`);
                        ws.send(JSON.stringify({ type: 'ERROR', message: scriptErr.message }));
                    }
                } finally {
                    delete bot.digSafe;
                    delete bot.activateBlockSafe;
                    delete bot.placeBlockSafe;
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
            type: 'STATUS',
            health: Math.round(bot.health),
            food: bot.food,
            saturation: bot.foodSaturation,
            inventory: inventoryMap,
            inventoryUsed: bot.inventory.items().length,
            onFire: (bot.entity.metadata[0] & 0x01) !== 0,
            heldItem: bot.heldItem ? { name: bot.heldItem.name, count: bot.heldItem.count } : null,
            position: bot.entity.position.floored()
        };
        ws.send(JSON.stringify(status));
    };

    bot.on('health', sendStatus);
    bot.on('updateSlot', sendStatus);

    const onChat = (username, message) => {
        if (username === bot.username || ws.readyState !== WebSocket.OPEN) return;
        let processedMessage = message;
        if (processedMessage.endsWith(']') && !processedMessage.startsWith('[')) {
            processedMessage = processedMessage.slice(0, -1);
        }
        ws.send(JSON.stringify({ type: 'CHAT', username: username, message: processedMessage }));
    };
    bot.on('chat', onChat);

    const syncEnvironment = () => {
        if (ws.readyState !== WebSocket.OPEN || !bot.entity) return;
        if (syncTimeout) return;

        syncTimeout = setTimeout(() => {
            const entities = [...new Set(Object.values(bot.entities)
                .filter(e => e !== bot.entity && e.position.distanceSquared(bot.entity.position) < ENVIRONMENT_RADIUS**2)
                .map(e => getItemName(e))
                .filter(name => name)
            )].slice(0, 5);

            ws.send(JSON.stringify({ 
                type: 'ENVIRONMENT', 
                blocks: [...nearbyBlocksCount.keys()],
                entities: entities
            }));
            syncTimeout = null;
        }, 50);
    };

    const performFullScan = async () => {
        if (ws.readyState !== WebSocket.OPEN || !bot.entity) return;
        
        nearbyBlocksCount.clear();
        const currentPos = bot.entity.position;
        const botPos = currentPos.floored(); 
        const cursor = new Vec3(0, 0, 0);
        lastScanPos = currentPos.clone();

        for (let y = -1; y <= 2; y++) {
            cursor.y = botPos.y + y;
            for (let x = -ENVIRONMENT_RADIUS; x <= ENVIRONMENT_RADIUS; x++) {
                cursor.x = botPos.x + x;
                for (let z = -ENVIRONMENT_RADIUS; z <= ENVIRONMENT_RADIUS; z++) {
                    cursor.z = botPos.z + z;
                    const block = bot.blockAt(cursor);
                    if (block && !AIR_BLOCKS.has(block.name)) {
                        nearbyBlocksCount.set(block.name, (nearbyBlocksCount.get(block.name) || 0) + 1);
                    }
                }
            }
            await new Promise(resolve => setImmediate(resolve));
        }
        syncEnvironment();
    };

    bot.on('move', () => {
        if (!bot.entity) return;
        if (lastScanPos) {
            const dist = bot.entity.position.distanceTo(lastScanPos);
            if (dist < 1.5) return;
        }
        performFullScan();
    });

    const onBlockUpdate = (oldBlock, newBlock) => {
        if (ws.readyState !== WebSocket.OPEN || !bot.entity) return;
        const dist = newBlock.position.distanceTo(bot.entity.position);
        if (dist > BLOCK_UPDATE_RADIUS) return;

        // Incremental Update Logic
        if (oldBlock && nearbyBlocksCount.has(oldBlock.name)) {
            const count = nearbyBlocksCount.get(oldBlock.name) - 1;
            if (count <= 0) nearbyBlocksCount.delete(oldBlock.name);
            else nearbyBlocksCount.set(oldBlock.name, count);
        }

        if (newBlock && !AIR_BLOCKS.has(newBlock.name)) {
            nearbyBlocksCount.set(newBlock.name, (nearbyBlocksCount.get(newBlock.name) || 0) + 1);
        }

        syncEnvironment();
    };
    bot.on('blockUpdate', onBlockUpdate);

    // Efficient Entity Tracking
    const onEntityUpdate = (entity) => {
        if (entity === bot.entity) return;
        const distSq = entity.position.distanceSquared(bot.entity.position);
        if (distSq < (ENVIRONMENT_RADIUS + 2)**2) {
            syncEnvironment();
        }
    };
    bot.on('entitySpawn', onEntityUpdate);
    bot.on('entityGone', onEntityUpdate);
    bot.on('entityMoved', onEntityUpdate);

    // Ensure world is loaded and entity is initialized before sending initial sync 
    (async () => {
        console.log("Cortex linked. Waiting for world data to load...");
        while (!bot.entity || !bot.entity.position || !bot.blockAt(bot.entity.position)) {
            if (ws.readyState !== WebSocket.OPEN) return;
            await bot.waitForTicks(5);
        }
        sendStatus();
        performFullScan();
    })();

    ws.on('close', () => {
        console.log("Cortex disconnected from WebSocket.");

        bot.removeListener('chat', onChat);
        bot.removeListener('blockUpdate', onBlockUpdate);
        bot.removeListener('health', sendStatus);
        bot.removeListener('updateSlot', sendStatus);
        bot.removeListener('entitySpawn', onEntityUpdate);
        bot.removeListener('entityGone', onEntityUpdate);
        bot.removeListener('entityMoved', onEntityUpdate);
    });

});