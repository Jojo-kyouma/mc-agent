const mineflayer = require('mineflayer');
const mcData = require('minecraft-data')('1.20.1');
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
const WebSocket = require('ws');
const Vec3 = require('vec3');

// --- CONFIGURATION ---
const MINECRAFT_PORT = 51785; // Change this to the port shown when you "Open to LAN"
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
        if (item) return `${item.name} (dropped)`;
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
vec3 = Vec3;

const wss = new WebSocket.Server({ port: WS_PORT });
let lastScanPos = null;
let nearbyBlocksCount = new Map(); // name -> count
let syncTimeout = null;
let currentAbortController = null;
const scriptLibrary = new Map(); // name -> wrapper function

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
                const AIR_BLOCKS = new Set(['air', 'cave_air', 'void_air']);
                const UNSOLID_ANCHORS = new Set(['water', 'lava', 'tall_grass', 'grass', 'fire']);
                /**
                 * Validates player reach distance to a target position vector.
                 * @throws {Error} if target is further than 4 blocks away.
                 */
                function validateReach(targetPos, actionName) {
                    const dist = bot.entity.position.distanceTo(targetPos);
                    if (dist > 4) {
                        throw new Error(`${actionName}: Target is too far away (${dist.toFixed(2)} blocks). Max reach is 4 blocks.`);
                    }
                }
                /**
                 * Validates that the bot's crosshair is aimed directly at the intended reference block and face.
                 * @throws {Error} if looking at the wrong block or face.
                 */
                function validateLineOfSight(expectedRefBlock, expectedFace) {
                    const cursor = bot.blockAtCursor(4); // Raycast line-of-sight up to 4 blocks away
                    if (!cursor) {
                        throw new Error('placeBlockSafe: Bot is not looking at any block or raycast failed.');
                    }
                    
                    if (!cursor.position.equals(expectedRefBlock.position)) {
                        throw new Error(`placeBlockSafe: Look-target mismatch. Expected to look at ${expectedRefBlock.name} at ${expectedRefBlock.position}, but currently looking at ${cursor.name} at ${cursor.position}.`);
                    }
                    
                    if (cursor.face !== bot.spiderFaceToNumber?.(expectedFace) && cursor.face !== expectedFace.y) {
                        // Fallback checks depending on how your specific environment handles face vector transformations
                        if (expectedFace.x === 0 && expectedFace.y === 1 && expectedFace.z === 0 && cursor.face !== 1) {
                            throw new Error(`placeBlockSafe: Face targeting mismatch. You must look at the TOP face (y=1) of the anchor block.`);
                        }
                    }
                }
                bot.craftSafe = async (recipe, count, tableBlock) => {
                    if (!recipe) throw new Error('craftSafe: No recipe provided.');
                    if (tableBlock) validateReach(tableBlock.position, 'craftSafe');

                    const resultId = recipe.result.id;
                    const oldAmount = bot.inventory.count(resultId);
                    
                    await bot.craft(recipe, count, tableBlock);
                    
                    if (bot.inventory.count(resultId) <= oldAmount) {
                        throw new Error(`craftSafe: Crafting failed. Inventory count for item ID ${resultId} did not increase.`);
                    }
                };

                bot.gotoSafe = async (goal) => {
                    if (!goal || typeof goal.isValid !== 'function') {
                        throw new Error('gotoSafe: Invalid goal. Must use a dynamic pathfinder Goal mapping configuration.');
                    }
                    // Mineflayer pathfinder naturally handles signal abort errors and timeout events internally
                    await bot.pathfinder.goto(goal);
                };

                bot.placeBlockSafe = async (ref, face) => {
                    if (!ref || !ref.position) throw new Error('placeBlockSafe: Reference block is completely missing or invalid.');
                    if (!face) throw new Error('placeBlockSafe: Face vector direction must be specified.');
                    
                    // 1. Distance Reach Validation
                    validateReach(ref.position, 'placeBlockSafe');

                    // 2. Inventory and Structural Soundness Validations
                    const heldItem = bot.heldItem;
                    if (!heldItem || AIR_BLOCKS.has(heldItem.name)) throw new Error('placeBlockSafe: Hand is empty.');
                    if (AIR_BLOCKS.has(ref.name) || UNSOLID_ANCHORS.has(ref.name)) {
                        throw new Error(`placeBlockSafe: Cannot anchor placement on unstable surface: ${ref.name}`);
                    }

                    // 3. Collision Box Prevention
                    const targetPos = ref.position.add(face);
                    const botFeet = bot.entity.position.floored();
                    const botHead = bot.entity.position.offset(0, 1, 0).floored();
                    if (targetPos.equals(botFeet) || targetPos.equals(botHead)) {
                        throw new Error(`placeBlockSafe: Cannot execute placement. You are currently standing inside target coordinate ${targetPos}. Step back.`);
                    }

                    // 4. Pre-Placement Raycast Line-of-Sight Verification
                    validateLineOfSight(ref, face);

                    // 5. Execution
                    await bot.placeBlock(ref, face);
                };
                bot.findIds = (query) => {
                    return Object.values(mcData.items)
                        .filter(i => i.name.includes(query))
                        .map(i => i.id);
                };
                bot.recordError = (msg) => {
                    throw new Error(msg);
                };
                /**
                 * Simple verification helper for sequential behavior logic.
                 * @param {Function} predicate - Async function returning boolean.
                 * @param {string} failureMessage - Message to log if false.
                 * @returns {Promise<boolean>}
                 */
                bot.verify = async (predicate, failureMessage) => {
                    const result = await predicate();
                    if (!result && failureMessage) {
                        console.log(`[*] Verification failed: ${failureMessage}`);
                    }
                    return result;
                };

                // --- Execute the behavior script ---
                try {
                    const AsyncFunction = Object.getPrototypeOf(async function () { }).constructor;
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

                    if (data.library) {
                        for (const [name, code] of Object.entries(data.library)) {
                            const fn = new AsyncFunction('bot', 'Vec3', 'vec3', 'mcData', 'GoalNear', 'goals', code);
                            scriptLibrary.set(name, (...args) => fn(botProxy, Vec3, vec3, mcData, goals.GoalNear, ...args));
                        }
                    }

                    const libNames = Array.from(scriptLibrary.keys());
                    const libFunctions = Array.from(scriptLibrary.values());

                    const execute = new AsyncFunction('bot', 'Vec3', 'vec3', 'mcData', 'GoalNear', 'goals', ...libNames, data.behaviour_script);
                    const scriptPromise = execute(botProxy, Vec3, vec3, mcData, goals.GoalNear, ...libFunctions);
                    
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
    const onItemBreak = (item) => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ITEM_BREAK', item: item.name }));
        }
    };
    const onEntityHurt = (entity) => {
        if (entity === bot.entity && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'AGENT_ATTACKED' }));
        }
    };
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
            worldPosition_eyePosition: { 
                x: Number(eyePos.x.toFixed(2)), 
                y: Number(eyePos.y.toFixed(2)), 
                z: Number(eyePos.z.toFixed(2)) 
            }
        };
        ws.send(JSON.stringify(status));
    };
    const onChat = (username, message) => {
        if (username === bot.username) return;
        if (ws.readyState !== WebSocket.OPEN) return;
        let processedMessage = message;
        if (processedMessage.endsWith(']') && !processedMessage.startsWith('[')) {
            processedMessage = processedMessage.slice(0, -1);
        }
        ws.send(JSON.stringify({ type: 'CHAT', username: username, message: processedMessage }));
    };
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
    const onMove = () => {
        if (!bot.entity) return;
        if (lastScanPos) {
            const dist = bot.entity.position.distanceTo(lastScanPos);
            if (dist < 1.5) return;
        }
        performFullScan();
    };
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
    const onEntityUpdate = (entity) => {
        if (entity === bot.entity) return;
        const distSq = entity.position.distanceSquared(bot.entity.position);
        if (distSq < (ENVIRONMENT_RADIUS + 2)**2) {
            syncEnvironment();
        }
    };
    const onPlayerCollect = (collector, collectedEntity) => {
        if (collector === bot.entity) {
            setImmediate(sendStatus);
        }
    };
    const onInventorySlotUpdate = (slot, oldItem, newItem) => {
        sendStatus();
    };
    const onHeldItemChanged = (heldItem) => {
        sendStatus();
    };
    // --- Event Listeners ---
    const activeListeners = [];
    const listenTo = (emitter, event, handler) => {
        emitter.on(event, handler);
        activeListeners.push({ emitter, event, handler });
    };
    listenTo(bot, 'itemBreak', onItemBreak);
    listenTo(bot, 'entityHurt', onEntityHurt);
    listenTo(bot, 'health', sendStatus);
    listenTo(bot, 'chat', onChat);
    listenTo(bot, 'move', onMove);
    listenTo(bot, 'blockUpdate', onBlockUpdate);
    listenTo(bot, 'entitySpawn', onEntityUpdate);
    listenTo(bot, 'entityGone', onEntityUpdate);
    listenTo(bot, 'entityMoved', onEntityUpdate);

    listenTo(bot, 'playerCollect', onPlayerCollect);
    listenTo(bot.inventory, 'updateSlot', onInventorySlotUpdate);
    listenTo(bot, 'heldItemChanged', onHeldItemChanged);

    // Ensure world is loaded and entity is initialized before sending initial sync
    const onSpawn = () => {
        (async () => {
            while (!bot.entity || !bot.entity.position || !bot.blockAt(bot.entity.position)) {
                if (ws.readyState !== WebSocket.OPEN) return;
                await new Promise(resolve => setTimeout(resolve, 500));
            }
            sendStatus();
            performFullScan();
        })();
    };
    listenTo(bot, 'spawn', onSpawn);

    ws.on('close', () => {
        while (activeListeners.length > 0) {
            const { emitter, event, handler } = activeListeners.pop();
            emitter.removeListener(event, handler);
        }
        if (syncTimeout) {
            clearTimeout(syncTimeout);
            syncTimeout = null;
        }
        abortCurrentScript();
    });
});