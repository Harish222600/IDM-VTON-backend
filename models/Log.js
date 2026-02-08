const mongoose = require('mongoose');

const logSchema = new mongoose.Schema({
    action: {
        type: String,
        required: true,
        enum: [
            'user_register',
            'user_login',
            'user_logout',
            'password_reset_request',
            'password_reset_complete',
            'profile_update',
            'profile_image_upload',
            'account_delete',
            'tryon_request',
            'tryon_complete',
            'tryon_failed',
            'garment_create',
            'garment_update',
            'garment_delete',
            'user_block',
            'user_unblock',
            'admin_action',
            'user_create',
            'user_role_update',
            'system_config_update'
        ]
    },
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        default: null
    },
    targetId: {
        type: mongoose.Schema.Types.ObjectId,
        default: null
    },
    details: {
        type: mongoose.Schema.Types.Mixed,
        default: {}
    },
    ipAddress: {
        type: String,
        default: null
    },
    userAgent: {
        type: String,
        default: null
    }
}, {
    timestamps: true
});

// Index for efficient log querying
logSchema.index({ action: 1, createdAt: -1 });
logSchema.index({ userId: 1, createdAt: -1 });

// Static method to create a log entry
logSchema.statics.createLog = async function (action, userId, details = {}, req = null) {
    return await this.create({
        action,
        userId,
        details,
        ipAddress: req?.ip || req?.connection?.remoteAddress,
        userAgent: req?.get('User-Agent')
    });
};

module.exports = mongoose.model('Log', logSchema);
