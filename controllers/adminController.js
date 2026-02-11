const { User, Garment, TryOnResult, Log } = require('../models');
const { storageService, AnalyticsService } = require('../services');

/**
 * @desc    Get all users
 * @route   GET /api/admin/users
 * @access  Admin
 */
const getUsers = async (req, res, next) => {
    try {
        const { page = 1, limit = 20, search, isBlocked } = req.query;
        const skip = (parseInt(page) - 1) * parseInt(limit);

        const query = { role: 'user' };
        if (search) {
            query.$or = [
                { name: new RegExp(search, 'i') },
                { email: new RegExp(search, 'i') }
            ];
        }
        if (isBlocked !== undefined) {
            query.isBlocked = isBlocked === 'true';
        }

        const [users, total] = await Promise.all([
            User.find(query)
                .sort({ createdAt: -1 })
                .skip(skip)
                .limit(parseInt(limit))
                .select('-password'),
            User.countDocuments(query)
        ]);

        res.status(200).json({
            success: true,
            data: users,
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
 * @desc    Block/Unblock user
 * @route   PUT /api/admin/users/:id/block
 * @access  Admin
 */
const toggleUserBlock = async (req, res, next) => {
    try {
        const user = await User.findById(req.params.id);

        if (!user) {
            return res.status(404).json({
                success: false,
                message: 'User not found'
            });
        }

        if (user.role === 'admin') {
            return res.status(400).json({
                success: false,
                message: 'Cannot block admin users'
            });
        }

        user.isBlocked = !user.isBlocked;
        await user.save();

        // Log action
        await Log.createLog(
            user.isBlocked ? 'user_block' : 'user_unblock',
            req.user._id,
            { targetUserId: user._id, email: user.email },
            req
        );

        res.status(200).json({
            success: true,
            message: user.isBlocked ? 'User blocked' : 'User unblocked',
            data: user
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Delete user
 * @route   DELETE /api/admin/users/:id
 * @access  Admin
 */
const deleteUser = async (req, res, next) => {
    try {
        const user = await User.findById(req.params.id);

        if (!user) {
            return res.status(404).json({
                success: false,
                message: 'User not found'
            });
        }

        if (user.role === 'admin') {
            return res.status(400).json({
                success: false,
                message: 'Cannot delete admin users'
            });
        }

        // Delete user's profile image
        if (user.profileImage) {
            const imagePath = storageService.getPathFromUrl(user.profileImage);
            if (imagePath) await storageService.deleteImage(imagePath);
        }

        // Log before deleting
        await Log.createLog('admin_action', req.user._id, {
            action: 'delete_user',
            targetUserId: user._id,
            email: user.email
        }, req);

        await User.findByIdAndDelete(req.params.id);

        res.status(200).json({
            success: true,
            message: 'User deleted successfully'
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Create new user
 * @route   POST /api/admin/users
 * @access  Admin
 */
const createUser = async (req, res, next) => {
    try {
        const { name, email, password, role } = req.body;

        // Check if user exists
        const existingUser = await User.findOne({ email: email.toLowerCase() });
        if (existingUser) {
            return res.status(400).json({
                success: false,
                message: 'Email already registered'
            });
        }

        const user = await User.create({
            name,
            email: email.toLowerCase(),
            password,
            role: role || 'user'
        });

        // Log creation
        await Log.createLog('user_create', req.user._id, { targetUserId: user._id, email: user.email }, req);

        res.status(201).json({
            success: true,
            message: 'User created successfully',
            data: user
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Update user role
 * @route   PUT /api/admin/users/:id/role
 * @access  Admin
 */
const updateUserRole = async (req, res, next) => {
    try {
        const { role } = req.body;
        const user = await User.findById(req.params.id);

        if (!user) {
            return res.status(404).json({
                success: false,
                message: 'User not found'
            });
        }

        if (user.role === 'admin' && role !== 'admin') {
            // Prevent removing the last admin? (Optional safety check, skipping for now to keep it simple)
        }

        const oldRole = user.role;
        user.role = role;
        await user.save();

        // Log role change
        await Log.createLog('user_role_update', req.user._id, {
            targetUserId: user._id,
            oldRole,
            newRole: role
        }, req);

        res.status(200).json({
            success: true,
            message: 'User role updated successfully',
            data: user
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Get user activity
 * @route   GET /api/admin/users/:id/activity
 * @access  Admin
 */
const getUserActivity = async (req, res, next) => {
    try {
        const userId = req.params.id;

        const [user, tryOns, logs] = await Promise.all([
            User.findById(userId).select('-password'),
            TryOnResult.find({ userId }).sort({ createdAt: -1 }).limit(10).populate('garmentId', 'name imageUrl category'),
            Log.find({ userId }).sort({ createdAt: -1 }).limit(20)
        ]);

        if (!user) {
            return res.status(404).json({
                success: false,
                message: 'User not found'
            });
        }

        res.status(200).json({
            success: true,
            data: {
                user,
                recentTryOns: tryOns,
                recentLogs: logs
            }
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Create garment
 * @route   POST /api/admin/garments
 * @access  Admin
 */
const createGarment = async (req, res, next) => {
    try {
        const { name, category, gender, fabric, color, description } = req.body;

        if (!req.file) {
            return res.status(400).json({
                success: false,
                message: 'Please upload a garment image'
            });
        }

        // Upload image
        const { url: imageUrl } = await storageService.uploadImage(
            req.file.buffer,
            'garments',
            req.file.originalname
        );

        const garment = await Garment.create({
            name,
            category,
            gender,
            fabric,
            color,
            description,
            imageUrl,
            createdBy: req.user._id
        });

        // Log garment creation
        await Log.createLog('garment_create', req.user._id, { garmentId: garment._id }, req);

        res.status(201).json({
            success: true,
            message: 'Garment created successfully',
            data: garment
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Update garment
 * @route   PUT /api/admin/garments/:id
 * @access  Admin
 */
const updateGarment = async (req, res, next) => {
    try {
        const { name, category, gender, fabric, color, description, isActive } = req.body;

        const garment = await Garment.findById(req.params.id);

        if (!garment) {
            return res.status(404).json({
                success: false,
                message: 'Garment not found'
            });
        }

        // Update fields
        if (name) garment.name = name;
        if (category) garment.category = category;
        if (gender) garment.gender = gender;
        if (fabric !== undefined) garment.fabric = fabric;
        if (color !== undefined) garment.color = color;
        if (description !== undefined) garment.description = description;
        if (isActive !== undefined) garment.isActive = isActive;

        // Handle image update
        if (req.file) {
            // Delete old image
            const oldPath = storageService.getPathFromUrl(garment.imageUrl);
            if (oldPath) await storageService.deleteImage(oldPath);

            // Upload new image
            const { url: imageUrl } = await storageService.uploadImage(
                req.file.buffer,
                'garments'
            );
            garment.imageUrl = imageUrl;
        }

        await garment.save();

        // Log update
        await Log.createLog('garment_update', req.user._id, { garmentId: garment._id }, req);

        res.status(200).json({
            success: true,
            message: 'Garment updated successfully',
            data: garment
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Delete garment
 * @route   DELETE /api/admin/garments/:id
 * @access  Admin
 */
const deleteGarment = async (req, res, next) => {
    try {
        const garment = await Garment.findById(req.params.id);

        if (!garment) {
            return res.status(404).json({
                success: false,
                message: 'Garment not found'
            });
        }

        // Delete image
        const imagePath = storageService.getPathFromUrl(garment.imageUrl);
        if (imagePath) await storageService.deleteImage(imagePath);

        // Log before deleting
        await Log.createLog('garment_delete', req.user._id, {
            garmentId: garment._id,
            name: garment.name
        }, req);

        await Garment.findByIdAndDelete(req.params.id);

        res.status(200).json({
            success: true,
            message: 'Garment deleted successfully'
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Get all garments (admin view - includes inactive)
 * @route   GET /api/admin/garments
 * @access  Admin
 */
const getAllGarments = async (req, res, next) => {
    try {
        const { page = 1, limit = 20, category, isActive, search } = req.query;
        const skip = (parseInt(page) - 1) * parseInt(limit);

        const query = {};
        if (category) query.category = category;
        if (isActive !== undefined) query.isActive = isActive === 'true';
        if (search) {
            query.$or = [
                { name: new RegExp(search, 'i') },
                { description: new RegExp(search, 'i') }
            ];
        }

        const [garments, total] = await Promise.all([
            Garment.find(query)
                .sort({ createdAt: -1 })
                .skip(skip)
                .limit(parseInt(limit))
                .populate('createdBy', 'name email'),
            Garment.countDocuments(query)
        ]);

        res.status(200).json({
            success: true,
            data: garments,
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
 * @desc    Get analytics dashboard
 * @route   GET /api/admin/analytics
 * @access  Admin
 */
const getAnalytics = async (req, res, next) => {
    try {
        const [systemStats, popularGarments, dailyStats, categoryDist] = await Promise.all([
            AnalyticsService.getSystemAnalytics(),
            AnalyticsService.getPopularGarments(5),
            AnalyticsService.getDailyTryOnStats(7),
            AnalyticsService.getCategoryDistribution()
        ]);

        res.status(200).json({
            success: true,
            data: {
                ...systemStats,
                popularGarments,
                dailyStats,
                categoryDistribution: categoryDist
            }
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Get system logs
 * @route   GET /api/admin/logs
 * @access  Admin
 */
const getLogs = async (req, res, next) => {
    try {
        const { page = 1, limit = 50, action, userId } = req.query;
        const skip = (parseInt(page) - 1) * parseInt(limit);

        const query = {};
        if (action) query.action = action;
        if (userId) query.userId = userId;

        const [logs, total] = await Promise.all([
            Log.find(query)
                .sort({ createdAt: -1 })
                .skip(skip)
                .limit(parseInt(limit))
                .populate('userId', 'name email'),
            Log.countDocuments(query)
        ]);

        res.status(200).json({
            success: true,
            data: logs,
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
 * @desc    Get system configuration
 * @route   GET /api/admin/config
 * @access  Admin
 */
const getSystemConfig = async (req, res, next) => {
    try {
        let config = await require('../models/SystemConfig').findOne({ key: 'main_config' });

        if (!config) {
            config = await require('../models/SystemConfig').create({
                key: 'main_config',
                activeModel: 'IDM-VTON'
            });
        }

        res.status(200).json({
            success: true,
            data: config
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Update system configuration
 * @route   PUT /api/admin/config
 * @access  Admin
 */
const updateSystemConfig = async (req, res, next) => {
    try {
        const { activeModel } = req.body;

        if (!['IDM-VTON', 'OOTDiffusion'].includes(activeModel)) {
            return res.status(400).json({
                success: false,
                message: 'Invalid model selection'
            });
        }

        let config = await require('../models/SystemConfig').findOne({ key: 'main_config' });

        if (!config) {
            config = await require('../models/SystemConfig').create({
                key: 'main_config',
                activeModel
            });
        } else {
            config.activeModel = activeModel;
            config.updatedBy = req.user._id;
            await config.save();
        }

        // Log configuration update
        await Log.createLog('system_config_update', req.user._id, {
            activeModel
        }, req);

        res.status(200).json({
            success: true,
            message: 'System configuration updated',
            data: config
        });
    } catch (error) {
        next(error);
    }
};


/**
 * @desc    Get garment usage stats
 * @route   GET /api/admin/analytics/garment-usage
 * @access  Admin
 */
const getGarmentUsage = async (req, res, next) => {
    try {
        const stats = await AnalyticsService.getGarmentUsageStats();
        res.status(200).json({
            success: true,
            data: stats
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Get user usage stats
 * @route   GET /api/admin/analytics/user-usage
 * @access  Admin
 */
const getUserUsage = async (req, res, next) => {
    try {
        const stats = await AnalyticsService.getUserUsageStats();
        res.status(200).json({
            success: true,
            data: stats
        });
    } catch (error) {
        next(error);
    }
};

module.exports = {
    getUsers,
    toggleUserBlock,
    deleteUser,
    getUserActivity,
    createUser,
    updateUserRole,
    createGarment,
    updateGarment,
    deleteGarment,
    getAllGarments,
    getAnalytics,
    getLogs,
    getSystemConfig,
    updateSystemConfig,
    getGarmentUsage,
    getUserUsage
};
