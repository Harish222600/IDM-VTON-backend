const mongoose = require('mongoose');

const tryOnResultSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: [true, 'User ID is required'],
        index: true
    },
    garmentId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Garment',
        required: [true, 'Garment ID is required']
    },
    inputImageUrl: {
        type: String,
        required: [true, 'Input image URL is required']
    },
    outputImageUrl: {
        type: String,
        default: null
    },
    processingTime: {
        type: Number,
        default: null
    },
    status: {
        type: String,
        enum: ['pending', 'processing', 'completed', 'failed'],
        default: 'pending'
    },
    errorMessage: {
        type: String,
        default: null
    }
}, {
    timestamps: true
});

// Index for efficient querying user history
tryOnResultSchema.index({ userId: 1, createdAt: -1 });
tryOnResultSchema.index({ status: 1 });

module.exports = mongoose.model('TryOnResult', tryOnResultSchema);
