const { TryOnResult, Garment, Log } = require('../models');
const { storageService, aiService } = require('../services');

/**
 * @desc    Initiate virtual try-on
 * @route   POST /api/tryon
 * @access  Private
 */
const initiateTryOn = async (req, res, next) => {
    try {
        const { garmentId } = req.body;

        // Validate garment exists
        const garment = await Garment.findOne({ _id: garmentId, isActive: true });
        if (!garment) {
            return res.status(404).json({
                success: false,
                message: 'Garment not found'
            });
        }

        // Check if image uploaded
        if (!req.file) {
            return res.status(400).json({
                success: false,
                message: 'Please upload your photo'
            });
        }

        // Upload person image
        const { url: inputImageUrl, buffer: personBuffer } = await storageService.uploadTryOnImage(
            req.file.buffer,
            'tryon/input'
        );

        // Create try-on record with pending status
        const tryOnResult = await TryOnResult.create({
            userId: req.user._id,
            garmentId,
            inputImageUrl,
            status: 'processing'
        });

        // Log try-on request
        await Log.createLog('tryon_request', req.user._id, { garmentId, tryOnId: tryOnResult._id }, req);

        // Log the URLs being sent to AI Service (Debug for Invalid URL error)
        console.log('AI Service Inputs:', {
            inputImageUrl: inputImageUrl,
            garmentImageUrl: garment.imageUrl
        });

        // Create a descriptive string for the garment to help the AI model
        // e.g. "A red shirt", "blue jeans", "A floral dress"
        const garmentDescription = [
            garment.color,
            garment.category,
            garment.name
        ].filter(Boolean).join(' ') || "A generic garment";

        console.log(`Using garment description: "${garmentDescription}"`);

        // Determine category for AI service
        // OOTDiffusion needs: 'upper_body', 'lower_body', or 'dress'
        let category = 'upper_body';
        const lowerBodyKeywords = ['pants', 'trousers', 'shorts', 'skirt', 'jeans', 'lower'];
        const dressKeywords = ['dress', 'gown', 'saree', 'kurti'];

        const garmentCat = garment.category.toLowerCase();
        if (dressKeywords.some(k => garmentCat.includes(k))) {
            category = 'dress';
        } else if (lowerBodyKeywords.some(k => garmentCat.includes(k))) {
            category = 'lower_body';
        }

        // Perform AI try-on
        // Note: performTryOn expects URLs, not Buffers, as per the new @gradio/client implementation
        const result = await aiService.performTryOn(inputImageUrl, garment.imageUrl, garmentDescription, category);

        if (result.success) {
            // Upload output image
            const { url: outputImageUrl } = await storageService.uploadImage(
                result.imageBuffer,
                'tryon/output'
            );

            // Update try-on record
            tryOnResult.outputImageUrl = outputImageUrl;
            tryOnResult.status = 'completed';
            tryOnResult.processingTime = result.processingTime;
            await tryOnResult.save();

            // Log success
            await Log.createLog('tryon_complete', req.user._id, {
                tryOnId: tryOnResult._id,
                processingTime: result.processingTime
            }, req);

            res.status(200).json({
                success: true,
                message: 'Try-on completed successfully',
                data: {
                    id: tryOnResult._id,
                    inputImageUrl,
                    outputImageUrl,
                    garment: {
                        id: garment._id,
                        name: garment.name,
                        imageUrl: garment.imageUrl
                    },
                    processingTime: result.processingTime
                }
            });
        } else {
            console.error('AI Processing Error:', result.error); // Log the specific error

            // Update status to failed
            tryOnResult.status = 'failed';
            tryOnResult.errorMessage = result.error;
            tryOnResult.processingTime = result.processingTime;
            await tryOnResult.save();

            // Log failure
            await Log.createLog('tryon_failed', req.user._id, {
                tryOnId: tryOnResult._id,
                error: result.error
            }, req);

            res.status(500).json({
                success: false,
                message: 'Try-on processing failed',
                error: result.error, // Send error details to client
                debug: process.env.NODE_ENV === 'development' ? result.error : undefined
            });
        }
    } catch (error) {
        console.error('TryOn Controller Exception:', error);
        next(error);
    }
};

/**
 * @desc    Get try-on history
 * @route   GET /api/tryon/history
 * @access  Private
 */
const getHistory = async (req, res, next) => {
    try {
        const { page = 1, limit = 20 } = req.query;
        const skip = (parseInt(page) - 1) * parseInt(limit);

        const [results, total] = await Promise.all([
            TryOnResult.find({ userId: req.user._id })
                .sort({ createdAt: -1 })
                .skip(skip)
                .limit(parseInt(limit))
                .populate('garmentId', 'name imageUrl category'),
            TryOnResult.countDocuments({ userId: req.user._id })
        ]);

        res.status(200).json({
            success: true,
            data: results,
            pagination: {
                page: parseInt(page),
                limit: parseInt(limit),
                total,
                pages: Math.ceil(total / parseInt(limit))
            }
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Get single try-on result
 * @route   GET /api/tryon/:id
 * @access  Private
 */
const getTryOnResult = async (req, res, next) => {
    try {
        const result = await TryOnResult.findOne({
            _id: req.params.id,
            userId: req.user._id
        }).populate('garmentId', 'name imageUrl category gender');

        if (!result) {
            return res.status(404).json({
                success: false,
                message: 'Try-on result not found'
            });
        }

        res.status(200).json({
            success: true,
            data: result
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Delete try-on result
 * @route   DELETE /api/tryon/:id
 * @access  Private
 */
const deleteTryOnResult = async (req, res, next) => {
    try {
        const result = await TryOnResult.findOne({
            _id: req.params.id,
            userId: req.user._id
        });

        if (!result) {
            return res.status(404).json({
                success: false,
                message: 'Try-on result not found'
            });
        }

        // Delete images from storage
        if (result.inputImageUrl) {
            const inputPath = storageService.getPathFromUrl(result.inputImageUrl);
            if (inputPath) await storageService.deleteImage(inputPath);
        }
        if (result.outputImageUrl) {
            const outputPath = storageService.getPathFromUrl(result.outputImageUrl);
            if (outputPath) await storageService.deleteImage(outputPath);
        }

        await TryOnResult.findByIdAndDelete(req.params.id);

        res.status(200).json({
            success: true,
            message: 'Try-on result deleted'
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Clear all try-on history
 * @route   DELETE /api/tryon/history
 * @access  Private
 */
const clearHistory = async (req, res, next) => {
    try {
        // Get all user's try-on results
        const results = await TryOnResult.find({ userId: req.user._id });

        // Delete all images
        for (const result of results) {
            if (result.inputImageUrl) {
                const inputPath = storageService.getPathFromUrl(result.inputImageUrl);
                if (inputPath) await storageService.deleteImage(inputPath);
            }
            if (result.outputImageUrl) {
                const outputPath = storageService.getPathFromUrl(result.outputImageUrl);
                if (outputPath) await storageService.deleteImage(outputPath);
            }
        }

        // Delete all records
        await TryOnResult.deleteMany({ userId: req.user._id });

        res.status(200).json({
            success: true,
            message: 'Try-on history cleared'
        });
    } catch (error) {
        next(error);
    }
};

module.exports = {
    initiateTryOn,
    getHistory,
    getTryOnResult,
    deleteTryOnResult,
    clearHistory
};
