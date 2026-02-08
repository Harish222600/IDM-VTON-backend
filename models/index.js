const User = require('./User');
const Garment = require('./Garment');
const TryOnResult = require('./TryOnResult');
const Log = require('./Log');

module.exports = {
    User,
    Garment,
    TryOnResult,
    Log,
    SystemConfig: require('./SystemConfig')
};
