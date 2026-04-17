/**
 * DrumSampler API Service
 * Connects to local backend DrumSampler endpoints (replaces HuggingFace Space)
 */

/**
 * Generate drum samples from the local DrumSampler API
 * Uses the local backend workflow: randomize drums → render MIDI → get audio
 * @param {number} bpm - Target BPM for rendering
 * @returns {Promise<Object>} - Result with audio file URL and metadata
 */
export async function generateDrumSample(bpm = 120) {
  try {
    console.log(`🎲 Randomizing drum pattern at ${bpm} BPM...`);

    // Step 1: Randomize drums to get a random pattern
    const randomResponse = await fetch('/api/drum-sampler/randomize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    });

    if (!randomResponse.ok) {
      throw new Error(`Randomization failed: ${randomResponse.status}`);
    }

    const randomResult = await randomResponse.json();
    const drumMidi = randomResult.midiFile;
    const drumKit = randomResult.drumKit;

    console.log('✅ Random drum pattern selected:', drumMidi);

    // Step 2: Render the MIDI to audio with specified BPM
    console.log(`🎵 Rendering MIDI to audio at ${bpm} BPM...`);

    const formData = new FormData();
    formData.append('midiFile', drumMidi);
    formData.append('bpm', bpm.toString());

    const renderResponse = await fetch('/api/drum-sampler/render', {
      method: 'POST',
      body: formData
    });

    if (!renderResponse.ok) {
      throw new Error(`Render failed: ${renderResponse.status}`);
    }

    const renderResult = await renderResponse.json();
    
    console.log('✅ Drum audio rendered:', renderResult.fileName);

    return {
      success: true,
      midiFile: drumMidi,
      drumKit: drumKit,
      audioUrl: renderResult.audioUrl,
      fileName: renderResult.fileName,
      duration: renderResult.duration
    };

  } catch (error) {
    console.error('❌ Error generating drum sample:', error);
    throw new Error(`DrumSampler generation failed: ${error.message}`);
  }
}

/**
 * Download audio file from URL as a Blob
 * @param {string} url - URL to download from
 * @returns {Promise<Blob>}
 */
async function downloadAudioFile(url) {
  try {
    console.log('📥 Downloading audio file...');
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Download failed: ${response.status}`);
    }
    
    const blob = await response.blob();
    console.log('✅ Audio file downloaded, size:', blob.size);
    
    return blob;
    
  } catch (error) {
    console.error('❌ Error downloading audio:', error);
    throw error;
  }
}

/**
 * Generate and download drum samples in one call
 * This is the main function used by the frontend
 * @param {number} bpm - Target BPM for rendering
 * @returns {Promise<Object>} - { blob, audioUrl, midiFile, drumKit, fileName }
 */
export async function generateAndDownloadDrumSamples(bpm = 120) {
  try {
    const result = await generateDrumSample(bpm);

    if (!result.success) {
      throw new Error('Generation failed');
    }

    // Download the audio file as blob for preview
    const blob = await downloadAudioFile(result.audioUrl);

    console.log('✅ Downloaded drum sample:', result.midiFile);

    // Return both blob (for preview) and audioUrl (for persistence)
    return {
      blob: blob,
      audioUrl: result.audioUrl,  // Persistent URL
      midiFile: result.midiFile,
      drumKit: result.drumKit,
      fileName: result.fileName
    };

  } catch (error) {
    console.error('❌ Error in generateAndDownload:', error);
    throw error;
  }
}

// Export default object with all functions
export default {
  generateDrumSample,
  generateAndDownloadDrumSamples
};
