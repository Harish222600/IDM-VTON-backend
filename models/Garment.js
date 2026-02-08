const mongoose = require('mongoose');

const garmentSchema = new mongoose.Schema({
    name: {
        type: String,
        required: [true, 'Garment name is required'],
        trim: true,
        minlength: [2, 'Name must be at least 2 characters'],
        maxlength: [100, 'Name cannot exceed 100 characters']
    },
    category: {
        type: String,
        required: [true, 'Category is required'],
        enum: {
            values: ['shirt', 'kurti', 'saree', 'dress', 'pants', 'trousers', 'shorts', 'skirt', 'jeans', 'jacket', 't-shirt', 'blouse', 'sweater', 'other'],
            message: '{VALUE} is not a valid category'
        }
    },
    gender: {
        type: String,
        required: [true, 'Gender is required'],
        enum: {
            values: ['male', 'female', 'unisex'],
            message: '{VALUE} is not a valid gender option'
        }
    },
    fabric: {
        type: String,
        trim: true,
        maxlength: [50, 'Fabric type cannot exceed 50 characters']
    },
    color: {
        type: String,
        trim: true,
        maxlength: [30, 'Color cannot exceed 30 characters']
    },
    description: {
        type: String,
        trim: true,
        maxlength: [500, 'Description cannot exceed 500 characters']
    },
    imageUrl: {
        type: String,
        required: [true, 'Garment image URL is required']
    },
    isActive: {
        type: Boolean,
        default: true
    },
    createdBy: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    }
}, {
    timestamps: true
});

// Index for efficient searching and filtering
garmentSchema.index({ name: 'text', description: 'text' });
garmentSchema.index({ category: 1, gender: 1, isActive: 1 });
garmentSchema.index({ color: 1 });

module.exports = mongoose.model('Garment', garmentSchema);
