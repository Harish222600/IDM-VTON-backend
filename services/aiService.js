const { Client, handle_file } = require('@gradio/client');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const os = require('os');

// We use the exact model space provided by the user
const SPACE_ID = 'yisol/IDM-VTON';
const MAX_RETRIES = 3;
const RETRY_DELAY = 5000;

// Cache for the duplicated client
let cachedClient = null;

/**
 * Sleep utility
 */
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Fetch image as Blob from URL
 */
const fetchImageBlob = async (url) => {
    const response = await axios.get(url, { responseType: 'arraybuffer' });
    return new Blob([response.data], { type: 'image/jpeg' });
};

/**
 * Get or create duplicated client
 * Using duplicate() creates a private copy of the Space to avoid ZeroGPU quotas
 */
const getClient = async () => {
    if (cachedClient) {
        console.log('♻️ Using cached client...');
        return cachedClient;
    }

    const hasToken = !!process.env.HUGGINGFACE_API_KEY;
    console.log(`🔑 HF Token configured: ${hasToken ? 'YES' : 'NO'} (${hasToken ? process.env.HUGGINGFACE_API_KEY.substring(0, 5) + '...' : ''})`);

    if (!hasToken) {
        throw new Error('HUGGINGFACE_API_KEY is required for IDM-VTON');
    }

    // First try to connect normally (in case the space is available)
    try {
        console.log('🔗 Attempting to connect to Space...');
        cachedClient = await Client.connect(SPACE_ID, {
            hf_token: process.env.HUGGINGFACE_API_KEY
        });
        console.log('✅ Connected to Space successfully');
        return cachedClient;
    } catch (connectError) {
        console.log('⚠️ Direct connect failed, trying duplicate...', connectError.message);
    }

    // If connect fails, try to duplicate the space for private use
    console.log('� Duplicating Space for private use (this may take a few minutes on first run)...');
    cachedClient = await Client.duplicate(SPACE_ID, {
        hf_token: process.env.HUGGINGFACE_API_KEY,
        timeout: 300, // 5 minute timeout for the space to start
        private: true // Make it a private space
    });
    console.log('✅ Duplicated Space ready');
    return cachedClient;
};

/**
 * Perform virtual try-on using Gradio Client for yisol/IDM-VTON
 * @param {string} personImageUrl - Public URL of person image
 * @param {string} garmentImageUrl - Public URL of garment image
 */
const performTryOn = async (personImageUrl, garmentImageUrl) => {
    const startTime = Date.now();
    console.log(`🚀 Starting Try-On with model: ${SPACE_ID}`);

    try {
        // 1. Get client (cached or new)
        const client = await getClient();

        // 2. Fetch images as Blobs
        console.log('📥 Downloading images...');
        const personBlob = await fetchImageBlob(personImageUrl);
        const garmentBlob = await fetchImageBlob(garmentImageUrl);
        console.log('✅ Images downloaded');

        // 3. Prepare inputs matching the Gradio interface
        // Based on app.py: start_tryon(dict, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed)
        // The 'dict' is from gr.ImageEditor with {background, layers, composite}
        // For gr.ImageEditor, pass the blob directly wrapped with handle_file
        const personFile = handle_file(personBlob);
        const garmentFile = handle_file(garmentBlob);

        console.log('⏳ Sending request to Hugging Face Space (this may take 30-60s)...');

        // The ImageEditor component expects background as the main image
        const imageEditorInput = {
            background: personFile,
            layers: [],
            composite: null
        };

        // Call the /tryon endpoint with array parameters (positional args)
        // Based on: try_button.click(fn=start_tryon, inputs=[imgs, garm_img, prompt, is_checked, is_checked_crop, denoise_steps, seed])
        const result = await client.predict("/tryon", [
            imageEditorInput,           // imgs (ImageEditor dict)
            garmentFile,                 // garm_img
            "A stylish garment",        // prompt (garment description)
            true,                        // is_checked (use auto-generated mask)
            false,                       // is_checked_crop (use auto-crop)
            30,                          // denoise_steps
            42                           // seed
        ]);

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
            console.error('Unexpected output format:', JSON.stringify(outputImage));
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

        // Reset cached client on error so next call tries fresh
        cachedClient = null;

        const processingTime = Date.now() - startTime;
        return {
            success: false,
            error: error.message || 'Try-on processing failed',
            processingTime
        };
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
