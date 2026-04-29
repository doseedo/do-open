/**
 * Browser-compatible test for DrumSampler Space
 * Can be imported and run from React components or browser console
 */

import drumSamplerAPI from '../services/drumSamplerAPI';

/**
 * Simple test to check if DrumSampler connection works
 * @returns {Promise<Object>} - Test results
 */
export async function testDrumSamplerConnection() {
  console.log('🧪 Testing DrumSampler Space connection...');

  const results = {
    connected: false,
    apiInfo: null,
    sampleGenerated: false,
    error: null
  };

  try {
    // Test 1: Get API info
    console.log('📡 Getting DrumSampler API info...');
    try {
      const apiInfo = await drumSamplerAPI.getDrumSamplerInfo();
      results.apiInfo = apiInfo;
      results.connected = true;
      console.log('✅ Connected to DrumSampler!');
      console.log('   Available endpoints:', Object.keys(apiInfo.named_endpoints || {}));
    } catch (error) {
      console.log('⚠️  Could not get API info, but will try generation:', error.message);
    }

    // Test 2: Generate a simple drum sample
    console.log('📡 Generating test drum sample...');
    const result = await drumSamplerAPI.generateDrumSample({
      prompt: 'kick drum'
    });

    if (result.success) {
      results.sampleGenerated = true;
      results.connected = true;
      console.log('✅ Sample generated successfully!');
      console.log('   Files:', result.files);
      return results;
    }

  } catch (error) {
    console.error('❌ Test failed:', error);
    results.error = error.message;
  }

  return results;
}

/**
 * Generate a drum sample and play it in the browser
 * @param {Object} params - Generation parameters
 * @returns {Promise<void>}
 */
export async function generateAndPlayDrum(params = {}) {
  try {
    console.log('🥁 Generating drum sample:', params.prompt || 'kick drum');

    // Generate the sample
    const blobs = await drumSamplerAPI.generateAndDownloadDrumSamples({
      prompt: params.prompt || 'kick drum',
      duration: params.duration || 1.0,
      temperature: params.temperature || 1.0
    });

    if (blobs.length === 0) {
      throw new Error('No audio files generated');
    }

    // Play the first audio file
    const blob = blobs[0];
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);

    audio.onloadedmetadata = () => {
      console.log('✅ Audio loaded, duration:', audio.duration, 'seconds');
    };

    audio.onended = () => {
      console.log('✅ Playback finished');
      URL.revokeObjectURL(url);
    };

    console.log('▶️  Playing drum sample...');
    await audio.play();

    return { success: true, blob, audio };

  } catch (error) {
    console.error('❌ Error generating/playing drum:', error);
    throw error;
  }
}

/**
 * Generate a drum kit (multiple samples)
 * @param {Array<string>} drumTypes - Array of drum names
 * @returns {Promise<Array<Object>>} - Array of {name, blob, url}
 */
export async function generateDrumKit(drumTypes = ['kick', 'snare', 'hi-hat', 'tom']) {
  console.log('🥁 Generating drum kit with:', drumTypes.join(', '));

  const kit = [];

  for (const drumType of drumTypes) {
    try {
      console.log(`   Generating ${drumType}...`);

      const blobs = await drumSamplerAPI.generateAndDownloadDrumSamples({
        prompt: `${drumType} drum`,
        duration: 1.0
      });

      if (blobs.length > 0) {
        const blob = blobs[0];
        const url = URL.createObjectURL(blob);

        kit.push({
          name: drumType,
          blob: blob,
          url: url,
          audio: new Audio(url)
        });

        console.log(`   ✅ ${drumType} generated`);
      }

    } catch (error) {
      console.error(`   ❌ Failed to generate ${drumType}:`, error.message);
    }
  }

  console.log(`✅ Drum kit complete! Generated ${kit.length}/${drumTypes.length} samples`);
  return kit;
}

/**
 * Download drum sample as a file
 * @param {Object} params - Generation parameters
 * @param {string} filename - Output filename
 */
export async function generateAndDownload(params, filename = 'drum_sample.wav') {
  try {
    console.log('⬇️  Generating and downloading:', filename);

    const blobs = await drumSamplerAPI.generateAndDownloadDrumSamples(params);

    if (blobs.length === 0) {
      throw new Error('No audio generated');
    }

    // Create download link
    const blob = blobs[0];
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log('✅ Download started:', filename);

  } catch (error) {
    console.error('❌ Download failed:', error);
    throw error;
  }
}

// Make functions available in browser console
if (typeof window !== 'undefined') {
  window.testDrumSampler = testDrumSamplerConnection;
  window.generateAndPlayDrum = generateAndPlayDrum;
  window.generateDrumKit = generateDrumKit;
  window.generateAndDownloadDrum = generateAndDownload;
}

export default {
  testDrumSamplerConnection,
  generateAndPlayDrum,
  generateDrumKit,
  generateAndDownload
};
