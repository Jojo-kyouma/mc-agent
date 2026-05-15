const mineflayer = require('mineflayer');
const mcData = require('minecraft-data')('1.20.1');
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
const WebSocket = require('ws');
const Vec3 = require('vec3');
const { rawPlaceBlock } = require('./mc-utils.js'); // Who would believe that Mineflayer is insufficient? What a shame.

// --- CONFIGURATION ---
const MINECRAFT_PORT = 51238; // Change this to the port shown when you "Open to LAN"
const MC_VERSION = '1.20.1';

// Parse CLI args: node bridge.js [ws_port] [bot_username]
const WS_PORT = parseInt(process.argv[2]) || 8080;
const BOT_USERNAME = process.argv[3] || 'Agent';

const ENVIRONMENT_RADIUS = 20; // Radius for the horizontal scan (e.g., 2 creates a 5x5 area)
const BLOCK_UPDATE_RADIUS = ENVIRONMENT_RADIUS + 4; // Interaction distance (Bot can reach blocks ~4 blocks away)

const AIR_BLOCKS = new Set(['air', 'cave_air', 'void_air']);

const getItemName = (entity) => {
    if (entity.type === 'player') return entity.username;
    if (entity.name === 'item' && entity.metadata && entity.metadata[8]) {
        const itemMetadata = entity.metadata[8];
        const item = mcData.items[itemMetadata.itemId];
        if (item) return `${item.name}`;
    }
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
    let numTries = 0;
    ws.on('message', async (message) => {
        try {
            const data = JSON.parse(message);
            if (data.type === 'ABORT') {
                abortCurrentScript();
            } else if (data.type === 'ACTION' && data.behaviour_script) {
                console.log(`\n[*] ACT ${numTries}: ${data.description}`);
                console.log(`[*] SCRIPT:\n${data.behaviour_script.replaceAll(';', ';\n')}`);
                numTries++;

                abortCurrentScript();
                currentAbortController = new AbortController();
                const { signal } = currentAbortController;

                // --- Action API & Safety Layer ---
                const sleep = (ms) => new Promise(res => setTimeout(res, ms)); // Used by gotoSafe()
                bot.digSafe = async (b) => { 
                    return await bot.dig(b); 
                };
                bot.activateBlockSafe = async (b, ...a) => { 
                    return await bot.activateBlock(b, ...a); 
                };
                bot.craftSafe = async (recipe, count, b) => {
                    if (!recipe) throw new Error('craftSafe: No recipe provided. Check if you have enough materials.');
                    const resultId = recipe.result.id;
                    const oldAmount = bot.inventory.count(resultId);

                    await bot.craft(recipe, count, b);
                    
                    const newAmount = bot.inventory.count(resultId);
                    if (newAmount <= oldAmount) {
                        throw new Error(`craftSafe: Verification failed. Inventory count for item ID ${resultId} did not increase.`);
                    }
                };
                bot.gotoSafe = async (goal) => {
                    if (!goal || typeof goal.isValid !== 'function') {
                        throw new Error('gotoSafe: Invalid goal. You MUST use "new GoalNear(x, y, z, range)". Raw Vec3 is not allowed.');
                    }
                    let lastPos = bot.entity.position.clone();
                    let lastMoveTime = Date.now();
                    let finished = false;
                    const moveTask = bot.pathfinder.goto(goal).catch(() => {});
                    moveTask.finally(() => { finished = true; });

                    while (!finished) {
                        await sleep(500);
                        if (signal?.aborted) {
                            bot.pathfinder.setGoal(null);
                            throw new Error('Script aborted');
                        }
                        const distanceMoved = bot.entity.position.distanceTo(lastPos);
                        if (distanceMoved > 0.3) {
                            lastPos = bot.entity.position.clone();
                            lastMoveTime = Date.now();
                        } else if (Date.now() - lastMoveTime > 3500) {
                            bot.pathfinder.setGoal(null);
                            throw new Error('Movement stuck: Progress stalled in leaf/block.');
                        }
                    }
                    return await moveTask;
                };
                bot.placeBlockSafe = async (ref, face) => {
                    if (!ref || !ref.position) {
                        throw new Error('placeBlockSafe: referenceBlock is null or missing position. Did findBlock return null? Did you forget to "await" a previous action? Note: craftSafe does NOT return a block object.');
                    }
                    const targetPos = ref.position.add(face);
                    const botFeet = bot.entity.position.floored();
                    const botHead = bot.entity.position.offset(0, 1, 0).floored();

                    if (targetPos.equals(botFeet) || targetPos.equals(botHead)) {
                        throw new Error(`placeBlockSafe: Cannot place block. You are standing at the target position ${targetPos}. Move away first.`);
                    }

                    await rawPlaceBlock(bot, ref, face);
                    await bot.waitForTicks(2);

                    const placedBlock = bot.blockAt(targetPos);
                    if (!placedBlock || AIR_BLOCKS.has(placedBlock.name)) {
                        throw new Error(`placeBlockSafe: Target position ${targetPos} is ${placedBlock ? placedBlock.name : 'empty'}. referenceBlock must be a block.`);
                    }
                };
                bot.findIds = (query) => {
                    return Object.values(mcData.items)
                        .filter(i => i.name.includes(query))
                        .map(i => i.id);
                };
                bot.recordError = (m) => { throw new Error(m); };
                bot.recordSuccess = () => {};

                // --- Execute the behavior script ---
                try {
                    const handler = {
                        get(target, prop) {
                            const val = Reflect.get(target, prop);
                            if (typeof val === 'function') {
                                return (...args) => {
                                    if (signal.aborted) throw new Error('Script aborted');
                                    return val.apply(target, args);
                                };
                            }
                            if (val && typeof val === 'object' && prop !== 'inventory' && prop !== 'entities') {
                                return new Proxy(val, handler);
                            }
                            return val;
                        }
                    };
                    const botProxy = new Proxy(bot, handler);

                    const AsyncFunction = Object.getPrototypeOf(async function () { }).constructor;
                    const execute = new AsyncFunction('bot', 'Vec3', 'mcData', 'GoalNear', data.behaviour_script);
                    const scriptPromise = execute(botProxy, Vec3, mcData, goals.GoalNear);
                    
                    const abortPromise = new Promise((resolve, reject) => {
                        signal.addEventListener('abort', () => reject(new Error('Script aborted')), { once: true });
                    });

                    await Promise.race([scriptPromise, abortPromise]);

                    ws.send(JSON.stringify({ type: 'SUCCESS' }));
                } catch (scriptErr) {
                    const isAbort = signal.aborted || 
                                    scriptErr.message === 'Script aborted' || 
                                    scriptErr.message === 'Digging aborted' || 
                                    scriptErr.message === 'Goal cancelled';

                    if (!isAbort) {
                        const errorMsg = scriptErr instanceof Error ? scriptErr.message : String(scriptErr);
                        console.error(`[ERROR]: ${errorMsg}`);
                        ws.send(JSON.stringify({ type: 'ERROR', message: errorMsg }));
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
        const eyePos = bot.entity.position.offset(0, bot.entity.eyeHeight, 0);
        const status = {
            type: 'STATUS',
            health: Math.round(bot.health),
            food: bot.food,
            saturation: bot.foodSaturation,
            inventory: inventoryMap,
            inventoryUsed: bot.inventory.items().length,
            onFire: (bot.entity.metadata[0] & 0x01) !== 0,
            heldItem: bot.heldItem ? { name: bot.heldItem.name, count: bot.heldItem.count } : null,
            eyePosition: { 
                x: Number(eyePos.x.toFixed(2)), 
                y: Number(eyePos.y.toFixed(2)), 
                z: Number(eyePos.z.toFixed(2)) 
            }
        };
        ws.send(JSON.stringify(status));
    };
    bot.on('health', sendStatus);
    bot.on('playerUpdated', sendStatus);
    const onChat = (username, message) => {
        if (ws.readyState !== WebSocket.OPEN) return; // Only filter if WebSocket is not open
        // Removed: username === bot.username, so bot hears itself.
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
});