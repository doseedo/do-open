/**
 * Video Processing API Service
 * Handles video upload, scene detection, and audio extraction
 */

/**
 * Upload video for scene detection
 * @param {File} videoFile - Video file to upload
 * @param {boolean} useWhisper - Whether to use Whisper for transcription
 * @returns {Promise<Object>} - Response with video_id and task_id
 */
export async function uploadVideo(videoFile, useWhisper = false) {
  const formData = new FormData();
  formData.append('file', videoFile);
  formData.append('use_whisper', useWhisper);

  console.log(`📤 Uploading video for scene detection (Whisper: ${useWhisper ? 'enabled' : 'disabled'})...`);

  const response = await fetch('https://doseedo.com/uploadvideo/', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();

  if (!data.video_id || !data.task_id) {
    throw new Error('Missing video_id or task_id in response');
  }

  console.log('✅ Video uploaded:', data.video_id);
  return data;
}

/**
 * Poll for video processing task status
 * @param {string} taskId - Task ID to poll
 * @returns {Promise<Object>} - Task status response
 */
export async function pollVideoTaskStatus(taskId) {
  const response = await fetch(`https://doseedo.com/task-status/${taskId}`);

  if (!response.ok) {
    throw new Error(`Failed to check task status: ${response.status}`);
  }

  const data = await response.json();
  return data;
}

/**
 * Poll until video processing completes
 * @param {string} taskId - Task ID to poll
 * @param {Function} onProgress - Callback for progress updates (optional)
 * @param {number} pollInterval - Polling interval in ms (default: 3000)
 * @param {number} maxAttempts - Maximum polling attempts (default: 600 = 30 minutes)
 * @returns {Promise<Object>} - Final result with scene_changes
 */
export async function pollVideoUntilComplete(
  taskId,
  onProgress = null,
  pollInterval = 3000,
  maxAttempts = 600
) {
  let attempts = 0;

  while (attempts < maxAttempts) {
    try {
      const data = await pollVideoTaskStatus(taskId);
      console.log(`📊 Video processing poll attempt ${attempts + 1}:`, data.status);

      // Check if task completed
      if (data.status === 'SUCCESS') {
        console.log('✅ Video processing completed!');
        return {
          scene_changes: data.result?.scene_changes || [],
          video_duration: data.result?.duration || null,
          audio_url: data.result?.audio_url || null,  // Audio extracted by backend
          status: 'SUCCESS'
        };
      }

      // Check if task failed
      if (data.status === 'FAILURE') {
        throw new Error(data.error || 'Video processing failed');
      }

      // Call progress callback
      if (onProgress) {
        onProgress({
          attempts,
          status: data.status
        });
      }

      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, pollInterval));
      attempts++;

    } catch (error) {
      console.error('❌ Error polling video task status:', error);
      throw error;
    }
  }

  throw new Error('Video processing timeout (30 minutes)');
}

/**
 * Extract audio from video file using Web Audio API
 * @param {File} videoFile - Video file
 * @returns {Promise<Blob>} - Audio blob (WAV format)
 */
export async function extractAudioFromVideo(videoFile) {
  return new Promise((resolve, reject) => {
    const videoElement = document.createElement('video');
    videoElement.preload = 'metadata';

    videoElement.onloadedmetadata = () => {
      videoElement.currentTime = 0;
    };

    videoElement.onerror = () => {
      reject(new Error('Failed to load video'));
    };

    videoElement.onloadeddata = async () => {
      try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaElementSource(videoElement);
        const destination = audioContext.createMediaStreamDestination();

        source.connect(destination);

        const mediaRecorder = new MediaRecorder(destination.stream);
        const chunks = [];

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            chunks.push(e.data);
          }
        };

        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(chunks, { type: 'audio/webm' });
          resolve(audioBlob);
        };

        mediaRecorder.start();
        videoElement.play();

        videoElement.onended = () => {
          mediaRecorder.stop();
          audioContext.close();
        };
      } catch (error) {
        reject(error);
      }
    };

    videoElement.src = URL.createObjectURL(videoFile);
  });
}

/**
 * Collapse short scene changes (merge scenes closer than threshold)
 * @param {Array<number>} sceneChanges - Array of scene change timestamps
 * @param {number} threshold - Minimum duration between scenes (seconds)
 * @returns {Array<number>} - Collapsed scene changes
 */
export function collapseSceneChanges(sceneChanges, threshold = 3) {
  if (!sceneChanges || sceneChanges.length === 0) return [];

  const out = [];
  let start = sceneChanges[0];
  let end = start;

  for (let i = 1; i < sceneChanges.length; i++) {
    const t = sceneChanges[i];
    if (t - end < threshold) {
      end = t;
    } else {
      out.push(start);
      if (end !== start) out.push(end);
      start = end = t;
    }
  }

  if (start === end) {
    out.push(start);
  } else {
    out.push(start);
    out.push(end);
  }

  return out;
}

/**
 * Compute optimal tempo for each scene to align with musical bars
 * Matches backend algorithm in genfrominterface.py compute_best_tempos()
 * @param {Array<number>} sceneChanges - Array of scene change timestamps
 * @returns {Array<number>} - Optimal tempo (BPM) for each scene
 */
export function computeBestTempos(sceneChanges) {
  const MIN_TEMPO = 70;
  const MAX_TEMPO = 160;
  const MAX_TEMPO_JUMP = 20;

  const tempos = [];

  for (let i = 0; i < sceneChanges.length - 1; i++) {
    const duration = sceneChanges[i + 1] - sceneChanges[i];
    let best = null;
    let bestScore = Infinity;

    for (let bpm = MIN_TEMPO; bpm <= MAX_TEMPO; bpm++) {
      const secondsPerBeat = 60 / bpm;
      const beats = duration / secondsPerBeat;

      // Prefer durations that land near exact beats
      const residual = Math.abs(Math.round(beats) - beats);
      const fullBars = beats / 4;
      const barResidual = Math.abs(Math.round(fullBars) - fullBars);

      let score = residual + (barResidual * 0.5);

      // Penalize large tempo jumps
      if (tempos.length > 0) {
        const jump = Math.abs(bpm - tempos[tempos.length - 1]);
        if (jump > MAX_TEMPO_JUMP) {
          score += (jump - MAX_TEMPO_JUMP) * 0.3;
        }
      }

      if (score < bestScore) {
        best = bpm;
        bestScore = score;
      }
    }

    tempos.push(best);
  }

  return tempos;
}

/**
 * Convert scene changes to scene durations
 * @param {Array<number>} sceneChanges - Array of scene change timestamps
 * @param {number} totalDuration - Total video duration
 * @returns {Array<number>} - Array of scene durations
 */
export function sceneToDurations(sceneChanges, totalDuration) {
  if (!sceneChanges || sceneChanges.length === 0) return [];

  const durations = [];

  for (let i = 0; i < sceneChanges.length - 1; i++) {
    durations.push(sceneChanges[i + 1] - sceneChanges[i]);
  }

  // Add final segment from last scene change to end
  const lastSceneTime = sceneChanges[sceneChanges.length - 1];
  const finalSegmentDuration = totalDuration - lastSceneTime;

  if (finalSegmentDuration > 0.5) {
    durations.push(finalSegmentDuration);
  }

  return durations;
}

/**
 * Export audio to video using backend endpoint
 * @param {File} videoFile - Video file
 * @param {Blob} audioBlob - Audio blob (WAV or MP3)
 * @returns {Promise<Object>} - Task ID for polling status
 */
export async function exportAudioToVideo(videoFile, audioBlob) {
  const formData = new FormData();
  formData.append('video', videoFile);
  formData.append('audio', audioBlob, 'audio.wav');

  console.log('📤 Uploading video and audio for export...');

  const response = await fetch('https://doseedo.com/exportAudio/', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();

  if (!data.task_id) {
    throw new Error('Missing task_id in response');
  }

  console.log('✅ Export task created:', data.task_id);
  return data;
}

/**
 * Poll export task status
 * @param {string} taskId - Task ID to poll
 * @returns {Promise<Object>} - Task status response
 */
export async function pollExportTaskStatus(taskId) {
  const response = await fetch(`https://doseedo.com/export/status/${taskId}`);

  if (!response.ok) {
    throw new Error(`Failed to check export status: ${response.status}`);
  }

  const data = await response.json();
  return data;
}

/**
 * Poll until export completes and get download URL
 * @param {string} taskId - Task ID to poll
 * @param {Function} onProgress - Callback for progress updates (optional)
 * @param {number} pollInterval - Polling interval in ms (default: 2000)
 * @param {number} maxAttempts - Maximum polling attempts (default: 300 = 10 minutes)
 * @returns {Promise<string>} - Download URL for the exported video
 */
export async function pollExportUntilComplete(
  taskId,
  onProgress = null,
  pollInterval = 2000,
  maxAttempts = 300
) {
  let attempts = 0;

  while (attempts < maxAttempts) {
    try {
      const data = await pollExportTaskStatus(taskId);
      console.log(`📊 Export poll attempt ${attempts + 1}:`, data.status);

      // Check if task completed
      if (data.status === 'SUCCESS') {
        console.log('✅ Export completed!');
        return data;
      }

      // Check if task failed
      if (data.status === 'FAILURE') {
        throw new Error(data.error || 'Export failed');
      }

      // Call progress callback
      if (onProgress) {
        onProgress({
          attempts,
          status: data.status
        });
      }

      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, pollInterval));
      attempts++;

    } catch (error) {
      console.error('❌ Error polling export task status:', error);
      throw error;
    }
  }

  throw new Error('Export timeout (10 minutes)');
}
