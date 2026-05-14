// mc-utils.js
const directionMap = {
    '0,-1,0': 0, // Down
    '0,1,0': 1,  // Up
    '0,0,-1': 2, // North
    '0,0,1': 3,  // South
    '-1,0,0': 4, // West
    '1,0,0': 5   // East
};

/**
 * Sends a block placement packet without the Mineflayer 5000ms timeout.
 */
async function rawPlaceBlock(bot, ref, face) {
    const direction = directionMap[`${face.x},${face.y},${face.z}`];
    
    // Optional: Head sync before placement
    await bot.lookAt(ref.position.offset(0.5, 0.5, 0.5), true);
    bot.swingArm('mainhand');

    bot._client.write('block_place', {
        hand: 0, 
        location: ref.position,
        direction: direction,
        cursorX: 0.5,
        cursorY: 0.5,
        cursorZ: 0.5,
        insideBlock: false
    });
}

module.exports = { rawPlaceBlock };