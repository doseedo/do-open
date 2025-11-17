/**
 * Hugging Face API Service
 * Handles authentication and API calls to Hugging Face models
 * Updated for 2025 - using official @huggingface/inference SDK
 */

import { HfInference } from '@huggingface/inference';

// Get HF configuration from environment variables
const HF_API_TOKEN = process.env.REACT_APP_HF_API_TOKEN;
// Whoami endpoint (updated to v2)
const HF_WHOAMI_URL = 'https://huggingface.co/api/whoami-v2';

// Initialize HF Inference client
let hf = null;

// Default models
const DEFAULT_MODELS = {
  textToMusic: process.env.REACT_APP_HF_TEXT_TO_MUSIC_MODEL || 'facebook/musicgen-small',
  melodyToMusic: process.env.REACT_APP_HF_AUDIO_MODEL || 'facebook/musicgen-melody',
  textToSpeech: process.env.REACT_APP_HF_VOICE_MODEL || 'suno/bark',
};

/**
 * Check if HF API is configured
 * @returns {boolean}
 */
export function isHFConfigured() {
  return !!HF_API_TOKEN && HF_API_TOKEN !== 'your_huggingface_token_here';
}

/**
 * Get HF Inference client (lazy initialization)
 * @returns {HfInference}
 */
function getClient() {
  if (!isHFConfigured()) {
    throw new Error('Hugging Face API token not configured. Please add REACT_APP_HF_API_TOKEN to .env file');
  }

  if (!hf) {
    hf = new HfInference(HF_API_TOKEN);
  }

  return hf;
}

/**
 * Get authentication headers for HF API
 * @returns {Object}
 */
function getAuthHeaders() {
  if (!isHFConfigured()) {
    throw new Error('Hugging Face API token not configured. Please add REACT_APP_HF_API_TOKEN to .env file');
  }

  return {
    'Authorization': `Bearer ${HF_API_TOKEN}`,
    'Content-Type': 'application/json'
  };
}

/**
 * Query a Hugging Face model (using official SDK)
 * @param {string} modelId - Model ID on Hugging Face
 * @param {Object} data - Request data
 * @param {Object} options - Additional options
 * @returns {Promise<any>}
 */
export async function queryModel(modelId, data, options = {}) {
  const client = getClient();

  try {
    // Use SDK's textGeneration for text models
    if (options.task === 'text-generation') {
      return await client.textGeneration({
        model: modelId,
        inputs: data.inputs,
        parameters: data.parameters
      });
    }

    // For other tasks, use general request method
    const response = await client.request({
      model: modelId,
      inputs: data.inputs,
      parameters: data.parameters
    });

    return response;
  } catch (error) {
    throw new Error(`HF API Error: ${error.message}`);
  }
}

/**
 * Generate music from text prompt
 * @param {string} prompt - Text description of music
 * @param {Object} options - Generation options
 * @returns {Promise<Blob>} - Audio blob
 */
export async function generateMusicFromText(prompt, options = {}) {
  const modelId = options.model || DEFAULT_MODELS.textToMusic;
  const client = getClient();

  console.log('🎵 Generating music from text via Hugging Face:', prompt);

  try {
    const blob = await client.textToAudio({
      model: modelId,
      inputs: prompt,
      parameters: options.parameters || {}
    });

    console.log('✅ Music generation complete');
    return blob;
  } catch (error) {
    console.error('❌ Music generation failed:', error);
    throw new Error(`Music generation failed: ${error.message}`);
  }
}

/**
 * Generate music from melody (audio conditioning)
 * @param {File|Blob} audioFile - Input melody audio
 * @param {string} prompt - Text description
 * @param {Object} options - Generation options
 * @returns {Promise<Blob>} - Audio blob
 */
export async function generateMusicFromMelody(audioFile, prompt = '', options = {}) {
  const modelId = options.model || DEFAULT_MODELS.melodyToMusic;

  console.log('🎼 Generating music from melody via Hugging Face');

  // Convert audio file to base64 or send as binary
  const formData = new FormData();
  formData.append('file', audioFile);
  if (prompt) {
    formData.append('inputs', prompt);
  }

  const audioBlob = await queryModel(modelId, formData);
  console.log('✅ Melody-conditioned generation complete');
  return audioBlob;
}

/**
 * Generate speech from text
 * @param {string} text - Text to speak
 * @param {Object} options - Voice options
 * @returns {Promise<Blob>} - Audio blob
 */
export async function generateSpeech(text, options = {}) {
  const modelId = options.model || DEFAULT_MODELS.textToSpeech;

  console.log('🗣️ Generating speech via Hugging Face:', text);

  const data = {
    inputs: text,
    parameters: options.parameters || {}
  };

  const audioBlob = await queryModel(modelId, data);
  console.log('✅ Speech generation complete');
  return audioBlob;
}

/**
 * Get model info
 * @param {string} modelId - Model ID
 * @returns {Promise<Object>} - Model information
 */
export async function getModelInfo(modelId) {
  const response = await fetch(`https://huggingface.co/api/models/${modelId}`, {
    headers: getAuthHeaders()
  });

  if (!response.ok) {
    throw new Error(`Failed to get model info: ${response.status}`);
  }

  return await response.json();
}

/**
 * Check if a model is loaded and ready
 * @param {string} modelId - Model ID
 * @returns {Promise<boolean>}
 */
export async function isModelReady(modelId) {
  try {
    const client = getClient();

    // Try a simple inference request
    await client.textGeneration({
      model: modelId,
      inputs: 'test',
      parameters: { max_new_tokens: 1 }
    });

    return true;
  } catch (error) {
    // If error message includes "loading", model is cold-starting
    if (error.message && error.message.includes('loading')) {
      return false;
    }
    // Other errors might mean model doesn't support this task, but is available
    console.warn('Model status check inconclusive:', error.message);
    return true;
  }
}

/**
 * Wait for model to load (if cold start)
 * @param {string} modelId - Model ID
 * @param {number} maxWaitTime - Max wait time in seconds
 * @returns {Promise<boolean>}
 */
export async function waitForModel(modelId, maxWaitTime = 120) {
  console.log(`⏳ Waiting for model ${modelId} to load...`);

  const startTime = Date.now();
  const maxWaitMs = maxWaitTime * 1000;

  while (Date.now() - startTime < maxWaitMs) {
    const ready = await isModelReady(modelId);
    if (ready) {
      console.log(`✅ Model ${modelId} is ready`);
      return true;
    }

    // Wait 2 seconds before checking again
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  console.warn(`⚠️ Model ${modelId} did not load within ${maxWaitTime}s`);
  return false;
}

export default {
  isHFConfigured,
  queryModel,
  generateMusicFromText,
  generateMusicFromMelody,
  generateSpeech,
  getModelInfo,
  isModelReady,
  waitForModel,
  DEFAULT_MODELS
};
