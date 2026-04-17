/**
 * Utility to test Hugging Face API connection
 * Run this to verify your HF token is working
 */

import * as hfAPI from '../services/huggingfaceAPI';

/**
 * Test HF API connection and authentication
 * @returns {Promise<Object>} - Test results
 */
export async function testHFConnection() {
  console.log('🧪 Testing Hugging Face API connection...');

  const results = {
    configured: false,
    authenticated: false,
    modelReady: false,
    error: null
  };

  try {
    // Check if configured
    results.configured = hfAPI.isHFConfigured();
    console.log('✓ Configuration check:', results.configured ? 'PASS' : 'FAIL');

    if (!results.configured) {
      results.error = 'HF API token not configured in .env file';
      return results;
    }

    // Try to query a lightweight model to test authentication
    console.log('Testing authentication with a small model...');
    const testModelId = 'gpt2'; // Small, fast model for testing

    try {
      await hfAPI.queryModel(testModelId, {
        inputs: 'test',
        parameters: { max_new_tokens: 5 }
      });
      results.authenticated = true;
      console.log('✓ Authentication check: PASS');
    } catch (error) {
      console.error('✗ Authentication check: FAIL', error.message);
      results.error = error.message;
      return results;
    }

    // Check if default music model is ready
    console.log('Checking if music generation model is ready...');
    const musicModel = hfAPI.DEFAULT_MODELS.textToMusic;
    results.modelReady = await hfAPI.isModelReady(musicModel);
    console.log('✓ Model status:', results.modelReady ? 'READY' : 'LOADING');

    if (!results.modelReady) {
      console.log('ℹ️ Model is cold-starting, it may take 1-2 minutes to become available');
    }

    console.log('✅ All tests passed!');
    return results;

  } catch (error) {
    console.error('❌ Test failed:', error);
    results.error = error.message;
    return results;
  }
}

/**
 * Test music generation with a simple prompt
 * @returns {Promise<Blob|null>}
 */
export async function testMusicGeneration() {
  console.log('🎵 Testing music generation...');

  try {
    if (!hfAPI.isHFConfigured()) {
      throw new Error('HF API not configured');
    }

    const audioBlob = await hfAPI.generateMusicFromText(
      'upbeat electronic dance music',
      { duration: 100 } // Short test generation
    );

    console.log('✅ Music generation test successful!');
    console.log('Audio blob size:', audioBlob.size, 'bytes');

    return audioBlob;

  } catch (error) {
    console.error('❌ Music generation test failed:', error);
    throw error;
  }
}

// Export a function that can be called from browser console
window.testHFConnection = testHFConnection;
window.testMusicGeneration = testMusicGeneration;

export default {
  testHFConnection,
  testMusicGeneration
};
