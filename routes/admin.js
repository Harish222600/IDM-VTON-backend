const express = require('express');
const router = express.Router();
const { body } = require('express-validator');
const { adminController } = require('../controllers');
const { auth, adminAuth, uploadSingle } = require('../middleware');

// Validation middleware
const validateGarment = [
    body('name')
        .trim()
        .notEmpty().withMessage('Name is required')
        .isLength({ min: 2, max: 100 }).withMessage('Name must be 2-100 characters'),
    body('category')
        .notEmpty().withMessage('Category is required')
        .isIn(['shirt', 'kurti', 'saree', 'dress', 'pants', 'jacket', 't-shirt', 'blouse', 'sweater', 'other'])
        .withMessage('Invalid category'),
    body('gender')
        .notEmpty().withMessage('Gender is required')
        .isIn(['male', 'female', 'unisex']).withMessage('Invalid gender'),
    body('fabric')
        .optional()
        .trim()
        .isLength({ max: 50 }).withMessage('Fabric max 50 characters'),
    body('color')
        .optional()
        .trim()
        .isLength({ max: 30 }).withMessage('Color max 30 characters'),
    body('description')
        .optional()
        .trim()
        .isLength({ max: 500 }).withMessage('Description max 500 characters')
];

const validateGarmentUpdate = [
    body('name')
        .optional()
        .trim()
        .isLength({ min: 2, max: 100 }).withMessage('Name must be 2-100 characters'),
    body('category')
        .optional()
        .isIn(['shirt', 'kurti', 'saree', 'dress', 'pants', 'jacket', 't-shirt', 'blouse', 'sweater', 'other'])
        .withMessage('Invalid category'),
    body('gender')
        .optional()
        .isIn(['male', 'female', 'unisex']).withMessage('Invalid gender'),
    body('isActive')
        .optional()
        .isBoolean().withMessage('isActive must be boolean')
];

// Validation error handler
const validate = (req, res, next) => {
    const { validationResult } = require('express-validator');
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        return res.status(400).json({
            success: false,
            message: 'Validation error',
            errors: errors.array().map(e => e.msg)
        });
    }
    next();
};

const validateUserCreate = [
    body('name').trim().notEmpty().withMessage('Name is required'),
    body('email').isEmail().withMessage('Valid email is required'),
    body('password').isLength({ min: 6 }).withMessage('Password must be at least 6 characters'),
    body('role').optional().isIn(['user', 'admin']).withMessage('Invalid role')
];

const validateUserRole = [
    body('role').isIn(['user', 'admin']).withMessage('Invalid role')
];

// All routes require auth + admin
router.use(auth, adminAuth);

// User management
router.get('/users', adminController.getUsers);
router.put('/users/:id/block', adminController.toggleUserBlock);
router.delete('/users/:id', adminController.deleteUser);
router.get('/users/:id/activity', adminController.getUserActivity);
router.post('/users', validateUserCreate, validate, adminController.createUser);
router.put('/users/:id/role', validateUserRole, validate, adminController.updateUserRole);

// Garment management
router.get('/garments', adminController.getAllGarments);
router.post('/garments', uploadSingle('image'), validateGarment, validate, adminController.createGarment);
router.put('/garments/:id', uploadSingle('image'), validateGarmentUpdate, validate, adminController.updateGarment);
router.delete('/garments/:id', adminController.deleteGarment);

// Analytics & Logs
router.get('/analytics', adminController.getAnalytics);
router.get('/analytics/garment-usage', adminController.getGarmentUsage);
router.get('/analytics/user-usage', adminController.getUserUsage);
router.get('/logs', adminController.getLogs);

// System Configuration
router.get('/config', adminController.getSystemConfig);
router.put('/config', adminController.updateSystemConfig);

module.exports = router;
