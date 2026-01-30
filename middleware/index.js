const { auth, optionalAuth } = require('./auth');
const adminAuth = require('./adminAuth');
const { uploadSingle, uploadMultiple } = require('./upload');
const { errorHandler, notFound } = require('./errorHandler');

module.exports = {
    auth,
    optionalAuth,
    adminAuth,
    uploadSingle,
    uploadMultiple,
    errorHandler,
    notFound
};
