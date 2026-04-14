/**
 * Generation API Service
 * Handles all backend communication for audio generation
 */

// Use relative paths since React app and backend are on same server
// Leave empty to use relative URLs
const API_BASE_URL = '';

/**
 * Start a new generation task
 * @param {Object} params - Generation parameters
 * @param {File} audioFile - Optional audio conditioning file
 * @returns {Promise<Object>} - Response with task_id and expected_voices
 */
export async function startGeneration(params, audioFile = null) {
  const formData = new FormData();

  // Add audio/MIDI file if provided
  if (audioFile) {
    // Check if this is edited MIDI JSON (for regeneration)
    if (audioFile.type === 'application/json') {
      // Edited MIDI sent as JSON - backend will convert to MIDI and render
      formData.append('conditioningAudio', audioFile);
      console.log('🎹 Sending edited MIDI JSON for regeneration:', audioFile.name, 'size:', audioFile.size);
    } else if (params.inpaintMode) {
      // In inpaint mode, send the original input file
      if (audioFile.type === 'audio/midi') {
        formData.append('midiFile', audioFile);
        console.log('📤 Sending MIDI input file for inpainting:', audioFile.name);
      } else if (audioFile.type === 'audio/wav' || audioFile.type === 'audio/wave') {
        formData.append('conditioningAudio', audioFile);
        console.log('📤 Sending WAV conditioning input for inpainting:', audioFile.name);
      }
    } else {
      // For normal generation, append as usual
      formData.append('audioFile', audioFile);
      formData.append('conditioningAudio', audioFile);
      console.log('📤 Sending audio/MIDI file:', audioFile.name, 'type:', audioFile.type);
    }
  }

  // Add all other parameters as JSON
  const paramsJson = JSON.stringify(params);
  formData.append('params', paramsJson);

  console.log('📤 Starting generation with params:', params);
  console.log('📤 Audio file:', audioFile?.name || 'None');
  console.log('📤 Audio file type:', audioFile?.type || 'None');
  console.log('📤 Audio file size:', audioFile?.size || 'None', 'bytes');
  console.log('📤 Params JSON being sent to backend:', paramsJson);

  // Log inpainting specific params
  if (params.inpaintMode) {
    console.log('🎨 INPAINTING MODE ACTIVE');
    console.log('🎨 inpaintVoiceIndex:', params.inpaintVoiceIndex);
    console.log('🎨 inpaintStartTime:', params.inpaintStartTime);
    console.log('🎨 inpaintEndTime:', params.inpaintEndTime);
    console.log('🎨 monophonicMode:', params.monophonicMode);
  }

  // Log debug mode
  if (params.debugMode) {
    console.log('🐛 DEBUG MODE ENABLED');
    console.log('🐛 Trajectory logging will be saved to /mnt/msdd/generation_debug/');
    console.log('🐛 Process ID will be linked for feedback correlation');
  }

  const response = await fetch('/api/generate-do', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const result = await response.json();
  console.log('✅ Generation started:', result);
  return result;
}

/**
 * Poll for task status
 * @param {string} taskId - Task ID to poll
 * @returns {Promise<Object>} - Task status response
 */
export async function pollTaskStatus(taskId) {
  const response = await fetch(`/api/generate-do/task/${taskId}`);

  if (!response.ok) {
    throw new Error(`Failed to check task status: ${response.status}`);
  }

  const status = await response.json();
  return status;
}

/**
 * Poll for task completion with incremental updates
 * @param {string} taskId - Task ID to poll
 * @param {Function} onProgress - Callback for progress updates (optional)
 * @param {Function} onPartialResult - Callback for partial results (optional)
 * @param {number} maxAttempts - Maximum polling attempts (default: 1800 = 30 minutes)
 * @returns {Promise<Object>} - Final result with file_paths array
 */
export async function pollUntilComplete(
  taskId,
  onProgress = null,
  onPartialResult = null,
  maxAttempts = 1800
) {
  let attempts = 0;
  let lastCompletedCount = 0;
  const processedVoices = new Set(); // Track which voices we've already reported

  while (attempts < maxAttempts) {
    try {
      const status = await pollTaskStatus(taskId);
      console.log(`📊 Poll attempt ${attempts + 1}:`, status);

      // Check if task completed
      if (status.status === 'completed') {
        console.log('✅ Task completed! Result:', status.result);
        console.log('📦 Input files:', status.input_files);
        return {
          file_paths: Array.isArray(status.result) ? status.result : [],
          input_files: status.input_files || {},
          status: 'completed'
        };
      }

      // Check if task failed
      if (status.status === 'failed') {
        throw new Error(status.error || 'Task failed');
      }

      // Handle partial results (for monophonic/voice separation mode)
      if (onPartialResult && status.completed_voices && Array.isArray(status.completed_voices)) {
        // Check for new voices since last poll
        const newVoices = status.completed_voices.filter(voice => !processedVoices.has(voice));

        if (newVoices.length > 0) {
          console.log(`🎵 ${newVoices.length} new voice(s) completed:`, newVoices);

          // Mark voices as processed
          newVoices.forEach(voice => processedVoices.add(voice));
          lastCompletedCount = status.completed_voices.length;

          // Call partial result callback
          await onPartialResult({
            completed_voices: newVoices, // Only send new voices
            all_completed_voices: status.completed_voices, // All completed so far
            total_voices: status.total_voices,
            progress: status.progress,
            input_files: status.input_files || {}
          });
        }
      }

      // Call progress callback
      if (onProgress) {
        onProgress({
          attempts,
          progress: status.progress,
          completedVoices: lastCompletedCount,
          totalVoices: status.total_voices,
          status: status.status
        });
      }

      // Wait 1 second before next poll
      await new Promise(resolve => setTimeout(resolve, 1000));
      attempts++;

    } catch (error) {
      console.error('❌ Error polling task status:', error);
      throw error;
    }
  }

  throw new Error('Task timeout - generation took too long (30 minutes)');
}

/**
 * Generate drums (special endpoint)
 * @param {Object} params - Drum generation parameters
 * @returns {Promise<Object>} - Response with file paths
 */
export async function generateDrums(params) {
  const formData = new FormData();

  formData.append('tempo_override', params.tempo || 120);
  formData.append('pattern', params.pattern || 1); // 1, 2, or 4 bars

  if (params.sceneDurations) {
    formData.append('scene_durations', JSON.stringify(params.sceneDurations));
  }

  if (params.automationData) {
    formData.append('automation_data', JSON.stringify(params.automationData));
  }

  if (params.activeSamples) {
    formData.append('active_samples', JSON.stringify(params.activeSamples));
  }

  console.log('🥁 Generating orchestral drums with params:', params);

  const response = await fetch('/generate-drums', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }

  return await response.json();
}

/**
 * Generate risers (special endpoint)
 * @param {Object} params - Riser generation parameters
 * @returns {Promise<Object>} - Response with file paths
 */
export async function generateRisers(params) {
  const formData = new FormData();

  formData.append('tempo_override', params.tempo || 120);

  if (params.sceneDurations) {
    formData.append('scene_durations', JSON.stringify(params.sceneDurations));
  }

  if (params.automationData) {
    formData.append('automation_data', JSON.stringify(params.automationData));
  }

  console.log('🎚️ Generating risers with params:', params);

  const response = await fetch('/generate-risers', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }

  return await response.json();
}

/**
 * Separate stems from audio
 * @param {string} audioUrl - URL of audio to separate
 * @returns {Promise<Object>} - Response with separated stem URLs
 */
export async function separateStems(audioUrl) {
  const response = await fetch('/separate-stems', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ audioUrl })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

/**
 * List available MIDI files
 * @returns {Promise<Array>} - Array of MIDI file names
 */
export async function listMidiFiles() {
  const response = await fetch('/api/list-midi-files');

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Get MIDI file info
 * @param {string} filename - MIDI filename
 * @returns {Promise<Object>} - MIDI file metadata
 */
export async function getMidiInfo(filename) {
  const response = await fetch(`/api/get-midi-info/${encodeURIComponent(filename)}`);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Get MIDI file content
 * @param {string} filename - MIDI filename
 * @returns {Promise<Blob>} - MIDI file as blob
 */
export async function getMidiFile(filename) {
  const response = await fetch(`/api/get-midi-file/${encodeURIComponent(filename)}`);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.blob();
}

/**
 * Convert audio to MIDI using Basic Pitch
 * @param {File} audioFile - Audio file to convert
 * @param {number} bpm - Current timeline BPM (default: 120)
 * @param {boolean} detailedMode - Use detailed processing with slowdown (default: false)
 * @returns {Promise<Object>} - MIDI data and metadata
 */
export async function convertAudioToMidi(audioFile, bpm = 120, detailedMode = false) {
  const formData = new FormData();
  formData.append('audioFile', audioFile);
  formData.append('bpm', bpm.toString());
  formData.append('detailedMode', detailedMode.toString());

  console.log(`🎤 Converting audio to MIDI at ${bpm} BPM (${detailedMode ? 'detailed' : 'fast'} mode):`, audioFile.name);

  const response = await fetch('/api/audio-to-midi', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  const result = await response.json();
  console.log('✅ MIDI extraction complete:', result.metadata);
  return result;
}

/**
 * Generate audio with ACE-Step
 * @param {Object} params - ACE-Step parameters
 * @param {File} inputFile - Optional reference audio file (WAV, MP3, or MIDI)
 * @returns {Promise<Object>} - Response with task_id
 */
export async function generateACEStep(params, inputFile = null) {
  const formData = new FormData();

  // Add ACE-Step specific parameters
  formData.append('steps', params.aceSteps || 100);
  formData.append('prompt', params.acePrompt || '');
  formData.append('lyrics', params.aceLyrics || '');
  formData.append('key', params.aceKey || 'C');
  formData.append('seed', params.seed ?? 0);

  // Add detailed mode flag
  formData.append('detailed_mode', params.aceDetailedMode ?? false);

  // Add noise level (using denoising strength slider value directly)
  // The slider value (t0) is sent directly as noise_level
  const noiseLevel = params.t0 ?? 0.8;
  formData.append('noise_level', noiseLevel);

  // Add reference audio file if provided
  if (inputFile) {
    formData.append('ref_audio', inputFile);
    console.log('📤 Sending reference audio for ACE-Step:', inputFile.name, 'type:', inputFile.type);
  }

  // Add MIDI lyric map if provided
  if (params.midiLyricMap) {
    formData.append('midi_lyric_map', JSON.stringify(params.midiLyricMap));
    console.log('🎹 Sending MIDI lyric map for ACE-Step');
  }

  console.log('🎵 Generating with ACE-Step:', {
    steps: params.aceSteps || 100,
    prompt: params.acePrompt || '',
    lyrics: params.aceLyrics || '',
    key: params.aceKey || 'C',
    seed: params.seed ?? 0,
    detailed_mode: params.aceDetailedMode ?? false,
    noise_level: noiseLevel,
    t0_slider_value: params.t0 ?? 0.8,
    hasRefAudio: !!inputFile
  });

  const response = await fetch('/api/generate-ace-step', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }

  const result = await response.json();
  console.log('✅ ACE-Step generation started:', result);
  return result;
}

/**
 * Poll for ACE-Step task status
 * @param {string} taskId - Task ID to poll
 * @returns {Promise<Object>} - Task status response
 */
export async function pollACEStepTaskStatus(taskId) {
  const response = await fetch(`/api/generate-ace-step/task/${taskId}`);

  if (!response.ok) {
    throw new Error(`Failed to check ACE-Step task status: ${response.status}`);
  }

  const status = await response.json();
  return status;
}

/**
 * Poll for ACE-Step task completion
 * @param {string} taskId - Task ID to poll
 * @param {Function} onProgress - Callback for progress updates (optional)
 * @param {number} maxAttempts - Maximum polling attempts (default: 1800 = 30 minutes)
 * @returns {Promise<Object>} - Final result with file_paths array
 */
export async function pollACEStepUntilComplete(
  taskId,
  onProgress = null,
  maxAttempts = 1800
) {
  let attempts = 0;

  while (attempts < maxAttempts) {
    try {
      const status = await pollACEStepTaskStatus(taskId);
      console.log(`📊 ACE-Step poll attempt ${attempts + 1}:`, status);

      // Check if task completed
      if (status.status === 'completed' || status.state === 'SUCCESS') {
        console.log('✅ ACE-Step task completed! Result:', status.result);
        return {
          file_paths: Array.isArray(status.result?.file_paths)
            ? status.result.file_paths
            : (Array.isArray(status.result) ? status.result : []),
          input_files: status.result?.input_files || {},
          status: 'completed'
        };
      }

      // Check if task failed
      if (status.status === 'failed' || status.state === 'FAILURE') {
        throw new Error(status.error || status.result || 'Task failed');
      }

      // Call progress callback
      if (onProgress) {
        onProgress({
          attempts,
          status: status.status || status.state
        });
      }

      // Wait 1 second before next poll
      await new Promise(resolve => setTimeout(resolve, 1000));
      attempts++;

    } catch (error) {
      console.error('❌ Error polling ACE-Step task status:', error);
      throw error;
    }
  }

  throw new Error('ACE-Step task timeout - generation took too long (30 minutes)');
}

/**
 * Clarify audio using the InstrumentClarifier model
 * Improves timbre quality for generated instrumental tracks
 * @param {string} audioUrl - URL of audio to clarify
 * @param {string} instrumentGroup - Instrument group (e.g., 'brass', 'strings')
 * @param {string} instrumentSubgroup - Instrument subgroup (e.g., 'trumpet', 'violin')
 * @returns {Promise<Object>} - Response with task_id for polling
 */
export async function clarifyAudio(audioUrl, instrumentGroup, instrumentSubgroup) {
  const formData = new FormData();
  formData.append('audioUrl', audioUrl);
  formData.append('instrumentGroup', instrumentGroup);
  formData.append('instrumentSubgroup', instrumentSubgroup);

  console.log('🎺 Starting audio clarification:', {
    audioUrl,
    instrumentGroup,
    instrumentSubgroup
  });

  const response = await fetch('/clarify-audio', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }

  return await response.json();
}

/**
 * Poll for clarify audio task status
 * @param {string} taskId - Task ID to poll
 * @returns {Promise<Object>} - Task status response
 */
export async function pollClarifyStatus(taskId) {
  const response = await fetch(`/clarify-audio/status/${taskId}`);

  if (!response.ok) {
    throw new Error(`Failed to check clarify status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Apply an audio FX to a track (e.g., trumpet mute)
 * @param {string} audioUrl - URL of audio to process
 * @param {string} fxType - FX type (e.g., 'trumpet_mute')
 * @returns {Promise<Object>} - Response with task_id for polling
 */
export async function applyFx(audioUrl, fxType) {
  const formData = new FormData();
  formData.append('audioUrl', audioUrl);
  formData.append('fxType', fxType);

  console.log('🎛️ Starting apply-fx:', { audioUrl, fxType });

  const response = await fetch('/api/apply-fx', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }

  return await response.json();
}

/**
 * Poll for apply-fx task status
 * @param {string} taskId - Task ID to poll
 * @returns {Promise<Object>} - Task status response
 */
export async function pollFxStatus(taskId) {
  const response = await fetch(`/api/apply-fx/status/${taskId}`);

  if (!response.ok) {
    throw new Error(`Failed to check FX status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Download track with FX applied (reverb, EQ, VST plugins)
 * @param {Object} trackData - Track data including audioUrl and FX settings
 * @param {Object} busData - Bus data with bus-level FX settings
 * @param {Object} masterFX - Master FX settings
 * @param {Object} rc20FX - RC-20 Retro Color plugin settings (optional)
 * @param {Object} speccraftFX - SpecCraft plugin settings (optional)
 * @returns {Promise<Blob>} - Processed audio file as blob
 */
export async function downloadTrackWithFX(trackData, busData, masterFX, rc20FX = {}, speccraftFX = {}) {
  const formData = new FormData();

  try {
    // Fetch the original audio file
    console.log('📥 Fetching original audio from:', trackData.audioUrl);
    const audioResponse = await fetch(trackData.audioUrl);
    if (!audioResponse.ok) {
      throw new Error(`Failed to fetch audio: ${audioResponse.status}`);
    }
    const audioBlob = await audioResponse.blob();
    formData.append('audioFile', audioBlob, trackData.name || 'track.wav');

    // Add track-level FX settings
    const trackFX = {
      gain: trackData.gain || 1.0,
      // Add more track FX here as they're implemented
    };
    formData.append('trackFX', JSON.stringify(trackFX));

    // Add bus-level FX settings
    const busFX = {
      gain: busData?.gain || 1.0,
      // Add more bus FX here as they're implemented
    };
    formData.append('busFX', JSON.stringify(busFX));

    // Add master-level FX settings
    const masterFXSettings = {
      gain: masterFX?.masterGain || 0.8,
      reverb: masterFX?.showReverb ? {
        enabled: true,
        mix: masterFX?.reverbMix || 0.3
      } : { enabled: false },
      eq: masterFX?.showEQ ? {
        enabled: true,
        bands: masterFX?.eqBands || {}
      } : { enabled: false }
    };
    formData.append('masterFX', JSON.stringify(masterFXSettings));

    // Add RC-20 plugin settings
    formData.append('rc20FX', JSON.stringify(rc20FX));

    // Add SpecCraft plugin settings
    formData.append('speccraftFX', JSON.stringify(speccraftFX));

    console.log('📤 Sending track for FX processing...');
    console.log('   Track FX:', trackFX);
    console.log('   Bus FX:', busFX);
    console.log('   Master FX:', masterFXSettings);
    console.log('   RC-20 FX:', rc20FX);
    console.log('   SpecCraft FX:', speccraftFX);

    // Send to backend endpoint
    const response = await fetch('/api/download-with-fx', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    // Return the processed audio blob
    const processedBlob = await response.blob();
    console.log('✅ Track processed with FX, size:', processedBlob.size, 'bytes');
    return processedBlob;

  } catch (error) {
    console.error('❌ Error downloading track with FX:', error);
    throw error;
  }
}

// ============================================================================
// Stemphonic API
// ============================================================================

export async function listStemphonicCheckpoints() {
  const r = await fetch('/api/generate-stemphonic/checkpoints');
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function switchStemphonicCheckpoint(checkpointId) {
  const r = await fetch('/api/generate-stemphonic/checkpoint', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ checkpoint: checkpointId }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

export async function listStemphonicTimbres() {
  const r = await fetch('/api/generate-stemphonic/timbres');
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function generateStemphonic(params, midiFile = null, refAudioFile = null) {
  const { trackEvent, PRODUCT_EVENTS } = await import('../lib/telemetry');
  const t0 = performance.now();
  trackEvent(PRODUCT_EVENTS.GENERATION_STARTED, {
    route: '/api/generate-stemphonic',
    // Params are enum-y + small — safe to log. coerceProps drops any
    // binary or oversized fields defensively.
    params,
  });
  const fd = new FormData();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') fd.append(k, String(v));
  }
  if (midiFile) fd.append('midiFile', midiFile);
  if (refAudioFile) fd.append('refAudio', refAudioFile);
  try {
    const r = await fetch('/api/generate-stemphonic', { method: 'POST', body: fd });
    if (!r.ok) {
      const body = await r.text();
      // Gate surfaces 429 on quota + 503 on kill-switch — both are
      // intentional gates, not crashes. Record as `generation.failed`
      // with structured reasons so dashboards can split them.
      let reason = `http_${r.status}`;
      if (r.status === 429) reason = 'quota_exhausted';
      else if (r.status === 503) reason = 'service_unavailable';
      else if (r.status === 415) {
        // MIME allowlist bounce — record the specialized event too.
        try {
          const parsed = JSON.parse(body);
          trackEvent(PRODUCT_EVENTS.UPLOAD_REJECTED_NON_LATENT, {
            mime_type: parsed?.got_content_type || 'unknown',
            route: '/api/generate-stemphonic',
          });
        } catch (_) { /* fall through */ }
        reason = 'mime_allowlist_415';
      }
      trackEvent(PRODUCT_EVENTS.GENERATION_FAILED, {
        route: '/api/generate-stemphonic',
        reason,
        duration_ms: Math.round(performance.now() - t0),
      });
      throw new Error(`HTTP ${r.status}: ${body}`);
    }
    const out = await r.json();
    trackEvent(PRODUCT_EVENTS.GENERATION_SUCCEEDED, {
      route: '/api/generate-stemphonic',
      duration_ms: Math.round(performance.now() - t0),
    });
    return out;
  } catch (err) {
    // Only record network/throw failures here — structured HTTP
    // failures above are already recorded with their own reason.
    if (!/^HTTP \d/.test(err?.message || '')) {
      trackEvent(PRODUCT_EVENTS.GENERATION_FAILED, {
        route: '/api/generate-stemphonic',
        reason: 'network_error',
        duration_ms: Math.round(performance.now() - t0),
      });
    }
    throw err;
  }
}

export async function pollStemphonicUntilComplete(
  taskId,
  onProgress = null,
  maxAttempts = 600,
) {
  let attempts = 0;
  while (attempts < maxAttempts) {
    const r = await fetch(`/api/generate-stemphonic/task/${taskId}`);
    if (!r.ok) throw new Error(`Failed to poll stemphonic: ${r.status}`);
    const status = await r.json();
    if (onProgress) onProgress({ attempts, status: status.status });
    if (status.status === 'completed') return status.result;
    if (status.status === 'failed') {
      throw new Error(status.error || 'Stemphonic generation failed');
    }
    attempts += 1;
    await new Promise((res) => setTimeout(res, 1500));
  }
  throw new Error(`Stemphonic task timed out after ${maxAttempts} polls`);
}
