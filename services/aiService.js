const { Client } = require('@gradio/client');
const axios = require('axios');
const fs = require('fs');
const SystemConfig = require('../models/SystemConfig');

// Models
const IDM_VTON_SPACE = 'yisol/IDM-VTON';
const OOT_DIFFUSION_SPACE = 'levihsu/OOTDiffusion';

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
 * Perform virtual try-on
 * @param {string} personImageUrl 
 * @param {string} garmentImageUrl 
 * @param {string} garmentDescription 
 * @param {string} category - 'upper_body', 'lower_body', 'dress'
 */
const performTryOn = async (personImageUrl, garmentImageUrl, garmentDescription, category) => {
    const startTime = Date.now();

    try {
        // Get active model from config
        let config = await SystemConfig.findOne({ key: 'main_config' });
        const activeModel = config ? config.activeModel : 'IDM-VTON';

        console.log(`🚀 Starting Try-On with model: ${activeModel}`);

        // Use token from environment variables
        const token = process.env.HUGGINGFACE_API_KEY;
        const options = token ? { hf_token: token } : {};

        let client;
        let result;

        // 2. Fetch images
        const personBlob = await fetchImageBlob(personImageUrl);
        const garmentBlob = await fetchImageBlob(garmentImageUrl);

        if (activeModel === 'OOTDiffusion') {
            client = await Client.connect(OOT_DIFFUSION_SPACE, options);
            console.log(`🔌 Connected to ${OOT_DIFFUSION_SPACE}`);

            // Map category to OOTDiffusion inputs
            // Upper-body -> /process_hd
            // Lower-body / Dress -> /process_dc

            if (category === 'upper_body') {
                console.log('⚡ Using /process_hd for Upper-body');
                result = await client.predict("/process_hd", [
                    personBlob,      // vton_img
                    garmentBlob,     // garm_img
                    1,               // n_samples
                    20,              // n_steps (default)
                    2,               // image_scale (guidance)
                    -1               // seed (-1 for random)
                ]);
            } else {
                console.log('⚡ Using /process_dc for Lower-body/Dress');
                const ootdCategory = category === 'lower_body' ? 'Lower-body' : 'Dress';

                result = await client.predict("/process_dc", [
                    personBlob,      // vton_img
                    garmentBlob,     // garm_img
                    ootdCategory,    // category
                    1,               // n_samples
                    20,              // n_steps
                    2,               // image_scale
                    -1               // seed
                ]);
            }

        } else {
            // Default: IDM-VTON
            client = await Client.connect(IDM_VTON_SPACE, options);
            console.log(`Using Hardcoded Token for IDM-VTON...`);

            const imageEditorDict = {
                background: personBlob,
                layers: [],
                composite: null
            };

            console.log('⏳ Sending request to Hugging Face Space (IDM-VTON)...');
            console.log(`👕 Garment Description: "${garmentDescription}"`);

            result = await client.predict("/tryon", [
                imageEditorDict,    // dict
                garmentBlob,        // garm_img
                garmentDescription || "A shirt",
                true,               // is_checked (auto-masking)
                false,              // is_checked_crop
                30,                 // denoise_steps
                42                  // seed
            ]);
        }

        const processingTime = Date.now() - startTime;
        console.log(`✅ Try-on completed in ${processingTime}ms`);

        // Handle Result (Standardize output)
        // Check pattern of result.data
        const outputImage = result.data[0];

        // OOTDiffusion returns [{image: "url", caption: ""}] sometimes, or similar structure
        // Gradio Client normalizes this usually.
        // IDM-VTON returns result.data[0] as the image.

        let outputBuffer;
        let imageUrlToFetch;

        if (outputImage && outputImage.url) {
            imageUrlToFetch = outputImage.url;
        } else if (Array.isArray(outputImage) && outputImage[0] && outputImage[0].image && outputImage[0].image.url) {
            // OOTDiffusion gallery format sometimes
            imageUrlToFetch = outputImage[0].image.url;
        } else if (outputImage instanceof Blob) {
            outputBuffer = Buffer.from(await outputImage.arrayBuffer());
        }

        if (!outputBuffer && imageUrlToFetch) {
            const response = await axios.get(imageUrlToFetch, { responseType: 'arraybuffer' });
            outputBuffer = Buffer.from(response.data);
        }

        if (!outputBuffer) {
            // Fallback/Error check
            console.log("Debug Result Data:", JSON.stringify(result.data).substring(0, 200));
            if (activeModel === 'OOTDiffusion' && result.data[0] && Array.isArray(result.data[0])) {
                // Gallery format check [ [{image:..., caption:...}] ]
                const firstItem = result.data[0][0];
                if (firstItem && firstItem.image && firstItem.image.url) {
                    const response = await axios.get(firstItem.image.url, { responseType: 'arraybuffer' });
                    outputBuffer = Buffer.from(response.data);
                }
            }
        }

        if (!outputBuffer) {
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
        let config = await SystemConfig.findOne({ key: 'main_config' });
        const activeModel = config ? config.activeModel : 'IDM-VTON';
        const spaceId = activeModel === 'OOTDiffusion' ? OOT_DIFFUSION_SPACE : IDM_VTON_SPACE;

        // Simple check by connecting
        const token = process.env.HUGGINGFACE_API_KEY;
        const options = token ? { hf_token: token } : {};

        await Client.connect(spaceId, options);
        return { available: true, status: `Connected to ${activeModel}` };
    } catch (error) {
        return { available: false, error: error.message };
    }
};

module.exports = {
    performTryOn,
    checkModelStatus
};
