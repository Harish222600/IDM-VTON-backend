const { Garment } = require('../models');

/**
 * @desc    Get all garments with filters
 * @route   GET /api/garments
 * @access  Public
 */
const getGarments = async (req, res, next) => {
    try {
        const {
            category,
            gender,
            color,
            fabric,
            search,
            page = 1,
            limit = 20,
            sort = '-createdAt'
        } = req.query;

        // Build query
        const query = { isActive: true };

        if (category) query.category = category;
        if (gender) query.gender = gender;
        if (color) query.color = new RegExp(color, 'i');
        if (fabric) query.fabric = new RegExp(fabric, 'i');

        // Text search
        if (search) {
            query.$text = { $search: search };
        }

        // Pagination
        const skip = (parseInt(page) - 1) * parseInt(limit);

        // Execute query
        const [garments, total] = await Promise.all([
            Garment.find(query)
                .sort(sort)
                .skip(skip)
                .limit(parseInt(limit))
                .select('-createdBy'),
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
 * @desc    Get single garment
 * @route   GET /api/garments/:id
 * @access  Public
 */
const getGarment = async (req, res, next) => {
    try {
        const garment = await Garment.findOne({
            _id: req.params.id,
            isActive: true
        });

        if (!garment) {
            return res.status(404).json({
                success: false,
                message: 'Garment not found'
            });
        }

        res.status(200).json({
            success: true,
            data: garment
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Get all categories
 * @route   GET /api/garments/categories
 * @access  Public
 */
const getCategories = async (req, res, next) => {
    try {
        const categories = await Garment.aggregate([
            { $match: { isActive: true } },
            { $group: { _id: '$category', count: { $sum: 1 } } },
            { $sort: { count: -1 } }
        ]);

        res.status(200).json({
            success: true,
            data: categories.map(c => ({
                name: c._id,
                count: c.count
            }))
        });
    } catch (error) {
        next(error);
    }
};

/**
 * @desc    Get garment colors
 * @route   GET /api/garments/colors
 * @access  Public
 */
const getColors = async (req, res, next) => {
    try {
        const colors = await Garment.distinct('color', { isActive: true, color: { $ne: null } });

        res.status(200).json({
            success: true,
            data: colors.filter(c => c)
        });
    } catch (error) {
        next(error);
    }
};

module.exports = {
    getGarments,
    getGarment,
    getCategories,
    getColors
};
