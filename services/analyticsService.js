const { TryOnResult, Log, User, Garment } = require('../models');

/**
 * Analytics service for tracking system metrics
 */
class AnalyticsService {
    /**
     * Get overall system analytics
     */
    static async getSystemAnalytics() {
        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        const thisMonth = new Date(now.getFullYear(), now.getMonth(), 1);
        const lastMonth = new Date(now.getFullYear(), now.getMonth() - 1, 1);

        const [
            totalUsers,
            newUsersToday,
            newUsersThisMonth,
            totalGarments,
            activeGarments,
            totalTryOns,
            tryOnsToday,
            tryOnsThisMonth,
            successfulTryOns,
            failedTryOns,
            avgProcessingTime
        ] = await Promise.all([
            User.countDocuments({ role: 'user' }),
            User.countDocuments({ role: 'user', createdAt: { $gte: today } }),
            User.countDocuments({ role: 'user', createdAt: { $gte: thisMonth } }),
            Garment.countDocuments(),
            Garment.countDocuments({ isActive: true }),
            TryOnResult.countDocuments(),
            TryOnResult.countDocuments({ createdAt: { $gte: today } }),
            TryOnResult.countDocuments({ createdAt: { $gte: thisMonth } }),
            TryOnResult.countDocuments({ status: 'completed' }),
            TryOnResult.countDocuments({ status: 'failed' }),
            TryOnResult.aggregate([
                { $match: { status: 'completed', processingTime: { $exists: true } } },
                { $group: { _id: null, avg: { $avg: '$processingTime' } } }
            ])
        ]);

        const successRate = totalTryOns > 0
            ? ((successfulTryOns / totalTryOns) * 100).toFixed(2)
            : 0;

        return {
            users: {
                total: totalUsers,
                newToday: newUsersToday,
                newThisMonth: newUsersThisMonth
            },
            garments: {
                total: totalGarments,
                active: activeGarments
            },
            tryOns: {
                total: totalTryOns,
                today: tryOnsToday,
                thisMonth: tryOnsThisMonth,
                successful: successfulTryOns,
                failed: failedTryOns,
                successRate: parseFloat(successRate),
                avgProcessingTime: avgProcessingTime[0]?.avg || 0
            }
        };
    }

    /**
     * Get most popular garments
     */
    static async getPopularGarments(limit = 10) {
        const popular = await TryOnResult.aggregate([
            { $group: { _id: '$garmentId', count: { $sum: 1 } } },
            { $sort: { count: -1 } },
            { $limit: limit },
            {
                $lookup: {
                    from: 'garments',
                    localField: '_id',
                    foreignField: '_id',
                    as: 'garment'
                }
            },
            { $unwind: '$garment' },
            {
                $project: {
                    _id: '$garment._id',
                    name: '$garment.name',
                    category: '$garment.category',
                    imageUrl: '$garment.imageUrl',
                    tryOnCount: '$count'
                }
            }
        ]);

        return popular;
    }

    /**
     * Get daily try-on stats for the last N days
     */
    static async getDailyTryOnStats(days = 7) {
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - days);
        startDate.setHours(0, 0, 0, 0);

        const stats = await TryOnResult.aggregate([
            { $match: { createdAt: { $gte: startDate } } },
            {
                $group: {
                    _id: {
                        $dateToString: { format: '%Y-%m-%d', date: '$createdAt' }
                    },
                    total: { $sum: 1 },
                    successful: {
                        $sum: { $cond: [{ $eq: ['$status', 'completed'] }, 1, 0] }
                    },
                    failed: {
                        $sum: { $cond: [{ $eq: ['$status', 'failed'] }, 1, 0] }
                    }
                }
            },
            { $sort: { _id: 1 } }
        ]);

        return stats;
    }

    /**
     * Get user activity stats
     */
    static async getUserActivityStats(userId) {
        const [tryOnCount, favoritesCount, recentTryOns] = await Promise.all([
            TryOnResult.countDocuments({ userId }),
            User.findById(userId).select('favorites').then(u => u?.favorites?.length || 0),
            TryOnResult.find({ userId })
                .sort({ createdAt: -1 })
                .limit(5)
                .populate('garmentId', 'name imageUrl')
        ]);

        return {
            tryOnCount,
            favoritesCount,
            recentTryOns
        };
    }

    /**
     * Get category distribution
     */
    static async getCategoryDistribution() {
        const distribution = await Garment.aggregate([
            { $match: { isActive: true } },
            { $group: { _id: '$category', count: { $sum: 1 } } },
            { $sort: { count: -1 } }
        ]);

        return distribution;
    }
    /**
     * Get garment usage stats
     */
    static async getGarmentUsageStats(limit = 50) {
        const stats = await TryOnResult.aggregate([
            { $group: { _id: '$garmentId', count: { $sum: 1 } } },
            { $sort: { count: -1 } },
            { $limit: limit },
            {
                $lookup: {
                    from: 'garments',
                    localField: '_id',
                    foreignField: '_id',
                    as: 'garment'
                }
            },
            { $unwind: '$garment' },
            {
                $project: {
                    _id: '$garment._id',
                    name: '$garment.name',
                    category: '$garment.category',
                    imageUrl: '$garment.imageUrl',
                    tryOnCount: '$count',
                    gender: '$garment.gender'
                }
            }
        ]);
        return stats;
    }

    /**
     * Get user usage stats
     */
    static async getUserUsageStats(limit = 50) {
        const stats = await TryOnResult.aggregate([
            { $group: { _id: '$userId', count: { $sum: 1 } } },
            { $sort: { count: -1 } },
            { $limit: limit },
            {
                $lookup: {
                    from: 'users',
                    localField: '_id',
                    foreignField: '_id',
                    as: 'user'
                }
            },
            { $unwind: '$user' },
            {
                $project: {
                    _id: '$user._id',
                    name: '$user.name',
                    email: '$user.email',
                    tryOnCount: '$count',
                    joinedAt: '$user.createdAt'
                }
            }
        ]);
        return stats;
    }
}

module.exports = AnalyticsService;
