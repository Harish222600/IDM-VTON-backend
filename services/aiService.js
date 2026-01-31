const { Client, handle_file } = require('@gradio/client');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const os = require('os');

// We use the exact model space provided by the user
const SPACE_ID = 'yisol/IDM-VTON';
const MAX_RETRIES = 3;
const RETRY_DELAY = 5000;

/**
 * Sleep utility
 */
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Download image to temp file and return path
 */
const downloadToTempFile = async (url, prefix) => {
    const response = await axios.get(url, { responseType: 'arraybuffer' });
    const tempDir = os.tmpdir();
    const tempPath = path.join(tempDir, `${prefix}_${Date.now()}.jpg`);
    fs.writeFileSync(tempPath, response.data);
    return tempPath;
};

/**
 * Perform virtual try-on using Gradio Client for yisol/IDM-VTON
 * @param {string} personImageUrl - Public URL of person image
 * @param {string} garmentImageUrl - Public URL of garment image
 */
const performTryOn = async (personImageUrl, garmentImageUrl) => {
    const startTime = Date.now();
    console.log(`🚀 Starting Try-On with model: ${SPACE_ID}`);
    
    let personTempPath = null;
    let garmentTempPath = null;

    try {
        // 1. Initialize Client with HF token
        const hasToken = !!process.env.HUGGINGFACE_API_KEY;
        console.log(`🔑 HF Token configured: ${hasToken ? 'YES' : 'NO'} (${hasToken ? process.env.HUGGINGFACE_API_KEY.substring(0, 5) + '...' : ''})`);

        const client = await Client.connect(SPACE_ID, {
            hf_token: process.env.HUGGINGFACE_API_KEY
        });

        // 2. Download images to temp files for handle_file
        console.log('📥 Downloading images...');
        personTempPath = await downloadToTempFile(personImageUrl, 'person');
        garmentTempPath = await downloadToTempFile(garmentImageUrl, 'garment');
        console.log('✅ Images downloaded');

        // 3. Use handle_file to properly send images to Gradio
        // The ImageEditor expects a dict with 'background' as the image
        const personFile = await handle_file(personTempPath);
        const garmentFile = await handle_file(garmentTempPath);

        // 4. Prepare inputs matching the Gradio interface
        // Based on app.py: start_tryon(dict, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed)
        // The 'dict' is from gr.ImageEditor with {background, layers, composite}
        const imageEditorInput = {
            background: personFile,
            layers: [],
            composite: null
        };

        console.log('⏳ Sending request to Hugging Face Space (this may take 30-60s)...');

        // Call the /tryon endpoint with proper parameters
        const result = await client.predict("/tryon", {
            dict: imageEditorInput,        // ImageEditor input
            garm_img: garmentFile,         // Garment image
            garment_des: "A stylish garment", // Description
            is_checked: true,              // Use auto-masking
            is_checked_crop: false,        // Don't auto-crop
            denoise_steps: 30,             // Denoising steps
            seed: 42                       // Random seed
        });

        const processingTime = Date.now() - startTime;
        console.log(`✅ Try-on completed in ${processingTime}ms`);
        console.log('📊 Result structure:', JSON.stringify(result.data?.map(d => typeof d === 'object' ? Object.keys(d) : typeof d) || 'unknown'));

        // Result.data contains [output_image, masked_image]
        const outputImage = result.data[0];

        // Handle different output formats
        let outputBuffer;
        if (outputImage && outputImage.url) {
            // Remote URL from Gradio space
            console.log('📥 Downloading result from:', outputImage.url);
            const response = await axios.get(outputImage.url, { responseType: 'arraybuffer' });
            outputBuffer = Buffer.from(response.data);
        } else if (outputImage && outputImage.path) {
            // Local path (shouldn't happen for remote spaces)
            outputBuffer = fs.readFileSync(outputImage.path);
        } else if (typeof outputImage === 'string' && outputImage.startsWith('http')) {
            // Direct URL string
            const response = await axios.get(outputImage, { responseType: 'arraybuffer' });
            outputBuffer = Buffer.from(response.data);
        } else if (outputImage instanceof Blob) {
            outputBuffer = Buffer.from(await outputImage.arrayBuffer());
        } else {
            console.error('Unexpected output format:', outputImage);
            throw new Error('Unexpected output format from Gradio client');
        }

        return {
            success: true,
            imageBuffer: outputBuffer,
            processingTime
        };

    } catch (error) {
        console.error('❌ AI Service Error:', error);
        
        // Log more details for debugging
        if (error.message) {
            console.error('Error message:', error.message);
        }
        if (error.response) {
            console.error('Response status:', error.response.status);
            console.error('Response data:', error.response.data);
        }

        const processingTime = Date.now() - startTime;
        return {
            success: false,
            error: error.message || 'Try-on processing failed',
            processingTime
        };
    } finally {
        // Clean up temp files
        try {
            if (personTempPath && fs.existsSync(personTempPath)) {
                fs.unlinkSync(personTempPath);
            }
            if (garmentTempPath && fs.existsSync(garmentTempPath)) {
                fs.unlinkSync(garmentTempPath);
            }
        } catch (cleanupError) {
            console.warn('Failed to cleanup temp files:', cleanupError.message);
        }
    }
};

/**
 * Check if the AI model is available
 */
const checkModelStatus = async () => {
    try {
        // Simple check by connecting
        await Client.connect(SPACE_ID, { hf_token: process.env.HUGGINGFACE_API_KEY });
        return { available: true, status: 'Connected' };
    } catch (error) {
        return { available: false, error: error.message };
    }
};

module.exports = {
    performTryOn,
    checkModelStatus
};
