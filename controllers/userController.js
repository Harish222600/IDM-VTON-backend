const { User, Garment, TryOnResult, Log } = require('../models');
const { storageService } = require('../services');

/**
 * @desc    Get user profile
 * @route   GET /api/users/profile
 * @access  Private
 */
const getProfile = async (req, res, next) => {
    try {
        const user = await User.findById(req.user._id).populate('favorites', 'name imageUrl category');

        res.status(200).json({
            success: true,
            data: user
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Update user profile
 * @route   PUT /api/users/profile
 * @access  Private
 */
const updateProfile = async (req, res, next) => {
    try {
        const { name, bodyInfo } = req.body;

        const updateData = {};
        if (name) updateData.name = name;
        if (bodyInfo) {
            updateData.bodyInfo = {
                ...req.user.bodyInfo,
                ...bodyInfo
            };
        }

        const user = await User.findByIdAndUpdate(
            req.user._id,
            updateData,
            { new: true, runValidators: true }
        );

        // Log profile update
        await Log.createLog('profile_update', req.user._id, { updated: Object.keys(updateData) }, req);

        res.status(200).json({
            success: true,
            message: 'Profile updated successfully',
            data: user
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Upload profile image
 * @route   PUT /api/users/profile-image
 * @access  Private
 */
const uploadProfileImage = async (req, res, next) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                success: false,
                message: 'Please upload an image'
            });
        }

        // Delete old image if exists
        if (req.user.profileImage) {
            const oldPath = storageService.getPathFromUrl(req.user.profileImage);
            if (oldPath) {
                await storageService.deleteImage(oldPath);
            }
        }

        // Upload new image
        const { url } = await storageService.uploadImage(
            req.file.buffer,
            'profiles',
            req.file.originalname
        );

        // Update user
        const user = await User.findByIdAndUpdate(
            req.user._id,
            { profileImage: url },
            { new: true }
        );

        // Log profile image upload
        await Log.createLog('profile_image_upload', req.user._id, {}, req);

        res.status(200).json({
            success: true,
            message: 'Profile image updated',
            data: { profileImage: url }
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Change password
 * @route   PUT /api/users/password
 * @access  Private
 */
const changePassword = async (req, res, next) => {
    try {
        const { currentPassword, newPassword } = req.body;

        // Get user with password
        const user = await User.findById(req.user._id).select('+password');

        // Verify current password
        const isMatch = await user.comparePassword(currentPassword);
        if (!isMatch) {
            return res.status(400).json({
                success: false,
                message: 'Current password is incorrect'
            });
        }

        // Update password
        user.password = newPassword;
        await user.save();

        res.status(200).json({
            success: true,
            message: 'Password changed successfully'
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Delete user account
 * @route   DELETE /api/users/account
 * @access  Private
 */
const deleteAccount = async (req, res, next) => {
    try {
        const { password } = req.body;

        // Verify password
        const user = await User.findById(req.user._id).select('+password');
        const isMatch = await user.comparePassword(password);

        if (!isMatch) {
            return res.status(400).json({
                success: false,
                message: 'Password is incorrect'
            });
        }

        // Delete user's profile image
        if (user.profileImage) {
            const imagePath = storageService.getPathFromUrl(user.profileImage);
            if (imagePath) {
                await storageService.deleteImage(imagePath);
            }
        }

        // Log account deletion before deleting
        await Log.createLog('account_delete', req.user._id, { email: user.email }, req);

        // Delete user
        await User.findByIdAndDelete(req.user._id);

        res.status(200).json({
            success: true,
            message: 'Account deleted successfully'
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Get user favorites
 * @route   GET /api/users/favorites
 * @access  Private
 */
const getFavorites = async (req, res, next) => {
    try {
        const user = await User.findById(req.user._id)
            .populate('favorites', 'name category gender imageUrl color fabric');

        res.status(200).json({
            success: true,
            data: user.favorites
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Add garment to favorites
 * @route   POST /api/users/favorites/:id
 * @access  Private
 */
const addToFavorites = async (req, res, next) => {
    try {
        const garmentId = req.params.id;

        // Check if garment exists
        const garment = await Garment.findById(garmentId);
        if (!garment || !garment.isActive) {
            return res.status(404).json({
                success: false,
                message: 'Garment not found'
            });
        }

        // Add to favorites if not already added
        const user = await User.findByIdAndUpdate(
            req.user._id,
            { $addToSet: { favorites: garmentId } },
            { new: true }
        ).populate('favorites', 'name imageUrl category');

        res.status(200).json({
            success: true,
            message: 'Added to favorites',
            data: user.favorites
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Remove garment from favorites
 * @route   DELETE /api/users/favorites/:id
 * @access  Private
 */
const removeFromFavorites = async (req, res, next) => {
    try {
        const garmentId = req.params.id;

        const user = await User.findByIdAndUpdate(
            req.user._id,
            { $pull: { favorites: garmentId } },
            { new: true }
        ).populate('favorites', 'name imageUrl category');

        res.status(200).json({
            success: true,
            message: 'Removed from favorites',
            data: user.favorites
        });
    } catch (error) {
        next(error);
    }
};

module.exports = {
    getProfile,
    updateProfile,
    uploadProfileImage,
    changePassword,
    deleteAccount,
    getFavorites,
    addToFavorites,
    removeFromFavorites
};
