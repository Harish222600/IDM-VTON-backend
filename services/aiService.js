const { Client } = require('@gradio/client');
const axios = require('axios');
const fs = require('fs');

// We use the exact model space provided by the user
const SPACE_ID = 'yisol/IDM-VTON';
const MAX_RETRIES = 3;
const RETRY_DELAY = 5000;

/**
 * Sleep utility
 */
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Fetch image as Blob from URL
 */
const fetchImageBlob = async (url) => {
    const response = await axios.get(url, { responseType: 'arraybuffer' });
    return new Blob([response.data]);
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
        // 1. Initialize Client
        // We use hf_token if available to avoid rate limits/queue
        let token = process.env.HUGGINGFACE_API_KEY;

        // Sanitize token: remove quotes and whitespace if present
        if (token) {
            token = token.replace(/["']/g, '').trim();
        }

        const hasToken = !!token;
        console.log(`🔑 HF Token configured: ${hasToken ? 'YES' : 'NO'} (${hasToken ? token.substring(0, 5) + '...' : ''})`);

        const options = {};
        if (hasToken) {
            options.hf_token = token;
        }

        const client = await Client.connect(SPACE_ID, options);

        // 2. Fetch images as Blobs (Gradio JS client handles Blobs best for 'handle_file')
        const personBlob = await fetchImageBlob(personImageUrl);
        const garmentBlob = await fetchImageBlob(garmentImageUrl);

        // 3. Prepare inputs matching the Python implementation
        // The space expects 'dict' for person image with background/layers/composite
        const imageEditorDict = {
            background: personBlob,
            layers: [],
            composite: null
        };

        // 4. Predict
        // Corresponding to:
        // result = client.predict(
        //     dict=image_editor_dict, 
        //     garm_img=garmentBlob, 
        //     garment_des="A shirt", 
        //     is_checked=true, 
        //     is_checked_crop=false, 
        //     denoise_steps=30, 
        //     seed=42
        // )

        console.log('⏳ Sending request to Hugging Face Space (this may take 30-60s)...');

        const result = await client.predict("/tryon", [
            imageEditorDict,    // dict
            garmentBlob,        // garm_img
            "A shirt",          // garment_des (generic description works)
            true,               // is_checked (auto-masking)
            false,              // is_checked_crop
            30,                 // denoise_steps
            42                  // seed
        ]);

        const processingTime = Date.now() - startTime;
        console.log(`✅ Try-on completed in ${processingTime}ms`);

        // The result is usually [output_image_info, masked_image_info]
        // In JS client, we get an object with 'data' and the data itself might be a URL or Blob
        // Let's inspect the structure. Usually result.data contains the outputs.

        const outputImage = result.data[0];

        // The output from gradio/client can be a URL or a Blob info depending on the version
        // We need to convert this back to a buffer for us to upload to Supabase

        let outputBuffer;
        if (outputImage && outputImage.url) {
            // It's a remote URL from the Gradio space
            const response = await axios.get(outputImage.url, { responseType: 'arraybuffer' });
            outputBuffer = Buffer.from(response.data);
        } else if (outputImage instanceof Blob) {
            outputBuffer = Buffer.from(await outputImage.arrayBuffer());
        } else {
            throw new Error('Unexpected output format from Gradio client');
        }

        return {
            success: true,
            imageBuffer: outputBuffer,
            processingTime
        };

    } catch (error) {
        console.error('❌ AI Service Error:', error);

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
        let token = process.env.HUGGINGFACE_API_KEY;
        if (token) {
            token = token.replace(/["']/g, '').trim();
        }
        const options = token ? { hf_token: token } : {};

        await Client.connect(SPACE_ID, options);
        return { available: true, status: 'Connected' };
    } catch (error) {
        return { available: false, error: error.message };
    }
};

module.exports = {
    performTryOn,
    checkModelStatus
};
