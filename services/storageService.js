const supabase = require('../config/supabase');
const sharp = require('sharp');
const { v4: uuidv4 } = require('uuid');

const BUCKET_NAME = process.env.SUPABASE_STORAGE_BUCKET || 'tryon-images';

/**
 * Upload image to Supabase Storage
 * @param {Buffer} buffer - Image buffer
 * @param {string} folder - Folder path (e.g., 'profiles', 'garments', 'tryon')
 * @param {string} originalName - Original filename
 * @returns {Promise<{url: string, path: string}>}
 */
const uploadImage = async (buffer, folder, originalName = 'image') => {
    try {
        // Generate unique filename
        const ext = 'webp'; // Convert all images to webp for optimization
        const filename = `${uuidv4()}.${ext}`;
        const filePath = `${folder}/${filename}`;

        // Optimize image with Sharp
        const optimizedBuffer = await sharp(buffer)
            .webp({ quality: 85 })
            .toBuffer();

        // Upload to Supabase
        const { data, error } = await supabase.storage
            .from(BUCKET_NAME)
            .upload(filePath, optimizedBuffer, {
                contentType: 'image/webp',
                upsert: false
            });

        if (error) {
            throw new Error(`Storage upload error: ${error.message}`);
        }

        // Get public URL
        const { data: urlData } = supabase.storage
            .from(BUCKET_NAME)
            .getPublicUrl(filePath);

        return {
            url: urlData.publicUrl,
            path: filePath
        };
    } catch (error) {
        console.error('Upload error:', error);
        throw error;
    }
};

/**
 * Upload image for try-on processing (resize to optimal dimensions)
 * @param {Buffer} buffer - Image buffer
 * @param {string} folder - Folder path
 * @param {Object} options - Resize options
 * @returns {Promise<{url: string, path: string, buffer: Buffer}>}
 */
const uploadTryOnImage = async (buffer, folder, options = {}) => {
    try {
        const { width = 768, height = 1024 } = options;

        // Generate unique filename
        const filename = `${uuidv4()}.webp`;
        const filePath = `${folder}/${filename}`;

        // Resize and optimize for try-on model
        const processedBuffer = await sharp(buffer)
            .resize(width, height, {
                fit: 'cover',
                position: 'center'
            })
            .webp({ quality: 90 })
            .toBuffer();

        // Upload to Supabase
        const { data, error } = await supabase.storage
            .from(BUCKET_NAME)
            .upload(filePath, processedBuffer, {
                contentType: 'image/webp',
                upsert: false
            });

        if (error) {
            throw new Error(`Storage upload error: ${error.message}`);
        }

        // Get public URL
        const { data: urlData } = supabase.storage
            .from(BUCKET_NAME)
            .getPublicUrl(filePath);

        return {
            url: urlData.publicUrl,
            path: filePath,
            buffer: processedBuffer
        };
    } catch (error) {
        console.error('Try-on upload error:', error);
        throw error;
    }
};

/**
 * Delete image from Supabase Storage
 * @param {string} filePath - File path in storage
 * @returns {Promise<boolean>}
 */
const deleteImage = async (filePath) => {
    try {
        const { error } = await supabase.storage
            .from(BUCKET_NAME)
            .remove([filePath]);

        if (error) {
            console.error('Delete error:', error);
            return false;
        }

        return true;
    } catch (error) {
        console.error('Delete error:', error);
        return false;
    }
};

/**
 * Extract file path from Supabase URL
 * @param {string} url - Full Supabase URL
 * @returns {string|null}
 */
const getPathFromUrl = (url) => {
    try {
        if (!url) return null;
        const match = url.match(/\/storage\/v1\/object\/public\/[^/]+\/(.+)$/);
        return match ? match[1] : null;
    } catch {
        return null;
    }
};

module.exports = {
    uploadImage,
    uploadTryOnImage,
    deleteImage,
    getPathFromUrl,
    BUCKET_NAME
};
