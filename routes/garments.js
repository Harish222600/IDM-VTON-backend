const express = require('express');
const router = express.Router();
const { garmentController } = require('../controllers');

// Public routes - no auth required
router.get('/', garmentController.getGarments);
router.get('/categories', garmentController.getCategories);
router.get('/colors', garmentController.getColors);
router.get('/:id', garmentController.getGarment);

module.exports = router;
