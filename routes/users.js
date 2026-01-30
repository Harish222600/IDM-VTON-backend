const express = require('express');
const router = express.Router();
const { body } = require('express-validator');
const { userController } = require('../controllers');
const { auth, uploadSingle } = require('../middleware');

// Validation middleware
const validateProfile = [
    body('name')
        .optional()
        .trim()
        .isLength({ min: 2, max: 50 }).withMessage('Name must be 2-50 characters'),
    body('bodyInfo.gender')
        .optional()
        .isIn(['male', 'female', 'other']).withMessage('Invalid gender'),
    body('bodyInfo.height')
        .optional()
        .isInt({ min: 50, max: 300 }).withMessage('Height must be 50-300 cm'),
    body('bodyInfo.bodyType')
        .optional()
        .isIn(['slim', 'regular', 'athletic', 'plus']).withMessage('Invalid body type')
];

const validatePassword = [
    body('currentPassword')
        .notEmpty().withMessage('Current password is required'),
    body('newPassword')
        .isLength({ min: 6 }).withMessage('New password must be at least 6 characters')
];

const validateDelete = [
    body('password')
        .notEmpty().withMessage('Password is required to delete account')
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

// All routes require auth
router.use(auth);

// Profile routes
router.get('/profile', userController.getProfile);
router.put('/profile', validateProfile, validate, userController.updateProfile);
router.put('/profile-image', uploadSingle('image'), userController.uploadProfileImage);
router.put('/password', validatePassword, validate, userController.changePassword);
router.delete('/account', validateDelete, validate, userController.deleteAccount);

// Favorites routes
router.get('/favorites', userController.getFavorites);
router.post('/favorites/:id', userController.addToFavorites);
router.delete('/favorites/:id', userController.removeFromFavorites);

module.exports = router;
