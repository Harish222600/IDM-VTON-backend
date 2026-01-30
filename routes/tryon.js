const express = require('express');
const router = express.Router();
const { body } = require('express-validator');
const { tryonController } = require('../controllers');
const { auth, uploadSingle } = require('../middleware');

// Validation
const validateTryOn = [
    body('garmentId')
        .notEmpty().withMessage('Garment ID is required')
        .isMongoId().withMessage('Invalid garment ID')
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

// Try-on routes
router.post('/', uploadSingle('image'), validateTryOn, validate, tryonController.initiateTryOn);
router.get('/history', tryonController.getHistory);
router.delete('/history', tryonController.clearHistory);
router.get('/:id', tryonController.getTryOnResult);
router.delete('/:id', tryonController.deleteTryOnResult);

module.exports = router;
