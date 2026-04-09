/**
 * Session Export Service
 * Handles exporting complete sessions with all data and files to GCS
 *
 * GCS Storage Format:
 * users/{userId}/sessions/{sessionId}/
 *   - session.json          (session metadata and state)
 *   - tracks/{trackId}/
 *       - audio.{ext}       (audio file for track)
 *       - midi.mid          (MIDI data if applicable)
 *   - video/
 *       - original.{ext}    (original video file if uploaded)
 *   - thumbnail.jpg         (auto-generated thumbnail)
 */

import * as gcsUploadService from './gcsUploadService';
import { getLatentPlayer } from './latentPlayer';
import { getCurrentUser } from './authService';
import sessionAPI from './sessionAPI';

/**
 * Generate a unique session ID
 */
function generateSessionId() {
  return `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Convert audio URL (blob or data URL) to File object — legacy fallback
 * for environments without WebGPU.
 */
async function urlToFile(url, filename, mimeType) {
  if (!url) return null;

  try {
    const response = await fetch(url);
    const blob = await response.blob();
    return new File([blob], filename, { type: mimeType || blob.type });
  } catch (error) {
    console.warn(`Failed to convert URL to file: ${filename}`, error);
    return null;
  }
}

/**
 * Convert an audio blob URL to a .doae latent File. Uses the WebGPU
 * encoder via LatentPlayer to compress the audio ~30× before upload.
 */
async function urlToDoaeFile(url, filename, audioContext, player) {
  if (!url) return null;
  try {
    const response = await fetch(url);
    const blob = await response.blob();
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const doaeAb = await player.encodeBufferToDoae(audioBuffer);
    return new File([doaeAb], filename, { type: 'application/x-doae' });
  } catch (error) {
    console.warn(`Failed to encode track to .doae: ${filename}`, error);
    return null;
  }
}

/**
 * Extract all audio and MIDI files from tracks. Audio tracks are
 * encoded LOCALLY into .doae latent format via the WebGPU encoder
 * (~30× smaller than wav). Falls back to wav if WebGPU isn't available.
 */
async function extractTrackFiles(buses) {
  const files = [];

  // Lazy-init the latent player; this also fetches the encoder ONNX
  // bundle on first call (cached for the rest of the session).
  let player = null;
  let useLatents = false;
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
    player = await getLatentPlayer(audioCtx);
    useLatents = !!player.useWebGPU;
    if (useLatents) console.log('[session export] using .doae latent format');
    else console.warn('[session export] WebGPU unavailable — falling back to wav');
  } catch (e) {
    console.warn('[session export] latent encoder unavailable, will save raw wavs:', e);
    useLatents = false;
  }

  for (const bus of buses) {
    for (const track of bus.tracks || []) {
      // Extract audio file if exists
      if (track.audioUrl) {
        if (useLatents && track.type === 'audio') {
          const audioFile = await urlToDoaeFile(
            track.audioUrl,
            `track-${track.id}.doae`,
            player.audioContext,
            player,
          );
          if (audioFile) {
            files.push({
              file: audioFile,
              path: `tracks/${track.id}/audio.doae`,
              trackId: track.id,
              type: 'audio_latent',
            });
            continue;
          }
          // If encode failed, fall through to wav fallback below
        }
        // Wav fallback (legacy / non-WebGPU)
        const ext = track.type === 'audio' ? 'wav' : 'mp3';
        const audioFile = await urlToFile(
          track.audioUrl,
          `track-${track.id}.${ext}`,
          `audio/${ext === 'wav' ? 'wav' : 'mpeg'}`
        );
        if (audioFile) {
          files.push({
            file: audioFile,
            path: `tracks/${track.id}/audio.${ext}`,
            trackId: track.id,
            type: 'audio'
          });
        }
      }

      // Extract MIDI data if exists
      if (track.midiData && track.midiData.notes && track.midiData.notes.length > 0) {
        const midiBlob = createMidiFile(track.midiData);
        const midiFile = new File([midiBlob], `track-${track.id}.mid`, { type: 'audio/midi' });
        files.push({
          file: midiFile,
          path: `tracks/${track.id}/midi.mid`,
          trackId: track.id,
          type: 'midi'
        });
      }
    }
  }

  return files;
}

/**
 * Create a MIDI file from MIDI data
 * Simplified MIDI file creation (you may want to use a library like midi-writer-js)
 */
function createMidiFile(midiData) {
  // For now, just stringify the MIDI data as JSON
  // In production, you'd want to create a proper MIDI file
  const midiJson = JSON.stringify(midiData, null, 2);
  return new Blob([midiJson], { type: 'application/json' });
}

/**
 * Prepare session data for export (remove non-serializable data)
 */
function prepareSessionData(state, projectName) {
  // Create a clean copy of buses without audio URLs (files will be
  // uploaded separately). Audio paths point to .doae latents (preferred)
  // or .wav (legacy fallback) — the loader auto-detects from extension.
  const cleanBuses = state.buses.map(bus => ({
    ...bus,
    tracks: bus.tracks.map(track => ({
      ...track,
      // Remove blob URLs - we'll store file paths instead
      audioUrl: track.audioUrl
        ? (track.type === 'audio'
            ? `tracks/${track.id}/audio.doae`
            : `tracks/${track.id}/audio.mp3`)
        : null,
      // Keep MIDI data reference
      midiData: track.midiData ? {
        ...track.midiData,
        midiFile: track.midiData.notes?.length > 0 ? `tracks/${track.id}/midi.mid` : null
      } : null
    }))
  }));

  // Create session metadata
  return {
    version: '1.0',
    projectName,
    timestamp: Date.now(),
    lastModified: new Date().toISOString(),

    // Core session data
    state: {
      projectName: state.projectName,
      bpm: state.bpm,
      totalDuration: state.totalDuration,
      masterGain: state.masterGain,
      zoomLevel: state.zoomLevel,
      trackHeight: state.trackHeight,

      // Buses with cleaned track data
      buses: cleanBuses,

      // Generation parameters
      generationParams: state.generationParams,

      // Chords
      chords: state.chords,
      chordTrack: state.chordTrack,

      // Video metadata (file stored separately)
      video: {
        fileName: state.video?.fileName,
        duration: state.video?.duration,
        sceneChanges: state.video?.sceneChanges,
        sceneTempos: state.video?.sceneTempos,
        videoFile: state.video?.videoFile ? 'video/original' : null
      },

      // Master FX settings
      masterFX: state.masterFX,
      reverbDecay: state.reverbDecay,
      reverbPreDelay: state.reverbPreDelay,
      reverbRoomSize: state.reverbRoomSize,
      reverbDamping: state.reverbDamping,
      delayTime: state.delayTime,
      delayFeedback: state.delayFeedback,
      delayCutoff: state.delayCutoff,
      chorusRate: state.chorusRate,
      chorusDepth: state.chorusDepth,
      chorusFeedback: state.chorusFeedback,
      compressorThreshold: state.compressorThreshold,
      compressorRatio: state.compressorRatio,
      compressorAttack: state.compressorAttack,
      filterFrequency: state.filterFrequency,
      filterResonance: state.filterResonance,
      filterGain: state.filterGain,
      phaserRate: state.phaserRate,
      phaserDepth: state.phaserDepth,
      phaserFeedback: state.phaserFeedback
    }
  };
}

/**
 * Export complete session to GCS
 * @param {object} state - App state to export
 * @param {string} projectName - Name of the project
 * @param {object} options - Export options
 * @returns {Promise<object>} Session export result with GCS URLs
 */
export async function exportSessionToGCS(state, projectName, options = {}) {
  try {
    const user = getCurrentUser();
    if (!user || !user.id) {
      throw new Error('User not authenticated');
    }

    const sessionId = options.sessionId || generateSessionId();
    const basePath = `users/${user.id}/sessions/${sessionId}`;

    console.log(`📤 Exporting session: ${projectName} to ${basePath}`);

    // Step 1: Extract all track files (audio and MIDI)
    console.log('📦 Extracting track files...');
    const trackFiles = await extractTrackFiles(state.buses);

    // Step 2: Extract video file if exists
    let videoFile = null;
    if (state.video?.videoFile) {
      videoFile = {
        file: state.video.videoFile,
        path: 'video/original',
        type: 'video'
      };
    }

    // Step 3: Prepare session metadata JSON
    console.log('📝 Preparing session data...');
    const sessionData = prepareSessionData(state, projectName);
    const sessionJsonBlob = new Blob([JSON.stringify(sessionData, null, 2)], {
      type: 'application/json'
    });
    const sessionJsonFile = new File([sessionJsonBlob], 'session.json', {
      type: 'application/json'
    });

    // Step 4: Upload all files to GCS
    console.log(`📤 Uploading ${trackFiles.length + 1} files to GCS...`);

    const uploadPromises = [];

    // Upload session.json
    uploadPromises.push(
      gcsUploadService.uploadToGCS(
        sessionJsonFile,
        'session',
        {
          sessionId,
          projectName,
          filePath: `${basePath}/session.json`,
          fileType: 'metadata'
        }
      ).then(result => ({ ...result, path: 'session.json', type: 'metadata' }))
    );

    // Upload track files
    for (const trackFile of trackFiles) {
      uploadPromises.push(
        gcsUploadService.uploadToGCS(
          trackFile.file,
          'session',
          {
            sessionId,
            projectName,
            trackId: trackFile.trackId,
            filePath: `${basePath}/${trackFile.path}`,
            fileType: trackFile.type
          }
        ).then(result => ({ ...result, ...trackFile }))
      );
    }

    // Upload video file if exists
    if (videoFile) {
      uploadPromises.push(
        gcsUploadService.uploadToGCS(
          videoFile.file,
          'session',
          {
            sessionId,
            projectName,
            filePath: `${basePath}/${videoFile.path}`,
            fileType: 'video'
          }
        ).then(result => ({ ...result, ...videoFile }))
      );
    }

    const uploadResults = await Promise.all(uploadPromises);

    // Step 5: Build result object
    const result = {
      success: true,
      sessionId,
      basePath,
      files: uploadResults,
      metadata: {
        projectName,
        trackCount: state.buses.reduce((sum, bus) => sum + (bus.tracks?.length || 0), 0),
        duration: state.totalDuration,
        bpm: state.bpm,
        hasVideo: !!videoFile,
        exportedAt: new Date().toISOString()
      }
    };

    console.log('✅ Session exported successfully:', result);
    return result;

  } catch (error) {
    console.error('❌ Session export failed:', error);
    throw error;
  }
}

/**
 * Import session from GCS. Downloads session.json + each referenced
 * track file (`.doae` latent or `.wav`/`.mp3` legacy) and reconstructs
 * the studio state with playable blob URLs.
 *
 * For .doae tracks: fetches the latent, decodes via WebGPU, encodes
 * the resulting AudioBuffer back to a wav blob (only because the
 * existing studio mixer reads `track.audioUrl` as a blob URL — the
 * AudioBuffer never leaves the browser).
 */
export async function importSessionFromGCS(sessionId) {
  try {
    const user = getCurrentUser();
    if (!user || !user.id) {
      throw new Error('User not authenticated');
    }

    const basePath = `users/${user.id}/sessions/${sessionId}`;
    console.log(`📥 Importing session from ${basePath}`);

    // 1. Fetch session.json metadata via the auth backend's session API
    const sessionMeta = await sessionAPI.getSession(sessionId);
    if (!sessionMeta) throw new Error('Session not found');

    // session.json was uploaded to {basePath}/session.json — pull a
    // signed URL for it from gcsUploadService
    const sessionJsonUrl = sessionMeta.session_json_url
      || `${basePath}/session.json`;
    const signedJsonUrl = await gcsUploadService.getSignedUrl(sessionJsonUrl);
    const sessionJson = await fetch(signedJsonUrl).then(r => r.json());
    console.log(`  loaded session.json (${sessionJson.projectName})`);

    // 2. Lazy player for .doae decode
    let player = null;
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
      player = await getLatentPlayer(audioCtx);
    } catch (e) {
      console.warn('[session import] latent player unavailable; .doae tracks will be skipped:', e);
    }

    // 3. Walk every track and re-fetch its audio file
    const restoredBuses = [];
    for (const bus of sessionJson.buses || []) {
      const restoredTracks = [];
      for (const track of bus.tracks || []) {
        let audioUrl = null;
        if (track.audioUrl) {
          const trackPath = `${basePath}/${track.audioUrl}`;
          try {
            const signed = await gcsUploadService.getSignedUrl(trackPath);
            const ab = await fetch(signed).then(r => r.arrayBuffer());
            if (track.audioUrl.endsWith('.doae')) {
              if (!player || !player.useWebGPU) {
                console.warn(`  skip ${track.id}: WebGPU required to decode .doae`);
              } else {
                const audioBuffer = await player.decodeDoaeToBuffer(ab);
                // Convert AudioBuffer back to a wav blob URL so the
                // existing mixer/Wavesurfer code can play it.
                const wavBlob = audioBufferToWavBlob(audioBuffer);
                audioUrl = URL.createObjectURL(wavBlob);
              }
            } else {
              // Legacy wav/mp3 — use directly
              audioUrl = URL.createObjectURL(new Blob([ab], { type: 'audio/wav' }));
            }
          } catch (e) {
            console.warn(`  failed to load ${track.id} audio:`, e);
          }
        }
        restoredTracks.push({ ...track, audioUrl });
      }
      restoredBuses.push({ ...bus, tracks: restoredTracks });
    }

    return {
      ...sessionJson,
      buses: restoredBuses,
    };

  } catch (error) {
    console.error('❌ Session import failed:', error);
    throw error;
  }
}

/**
 * Encode an AudioBuffer to a 16-bit PCM wav Blob (browser-friendly,
 * no extra dependency). Used to feed decoded .doae latents back into
 * the existing wav-blob mixer code path.
 */
function audioBufferToWavBlob(audioBuffer) {
  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const numSamples = audioBuffer.length;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = numSamples * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  // WAV RIFF header
  let p = 0;
  const ws = (s) => { for (let i = 0; i < s.length; i++) view.setUint8(p++, s.charCodeAt(i)); };
  const w16 = (v) => { view.setUint16(p, v, true); p += 2; };
  const w32 = (v) => { view.setUint32(p, v, true); p += 4; };
  ws('RIFF'); w32(36 + dataSize); ws('WAVE');
  ws('fmt '); w32(16); w16(1); w16(numChannels);
  w32(sampleRate); w32(byteRate); w16(blockAlign); w16(8 * bytesPerSample);
  ws('data'); w32(dataSize);
  // PCM samples (interleaved)
  const channels = [];
  for (let c = 0; c < numChannels; c++) channels.push(audioBuffer.getChannelData(c));
  for (let i = 0; i < numSamples; i++) {
    for (let c = 0; c < numChannels; c++) {
      let s = Math.max(-1, Math.min(1, channels[c][i]));
      view.setInt16(p, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      p += 2;
    }
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Calculate session size (for display purposes)
 */
export function calculateSessionSize(state) {
  let totalSize = 0;

  // Estimate JSON size
  totalSize += JSON.stringify(state).length;

  // Count tracks with audio
  let audioTracks = 0;
  let midiTracks = 0;

  for (const bus of state.buses || []) {
    for (const track of bus.tracks || []) {
      if (track.audioUrl) audioTracks++;
      if (track.midiData?.notes?.length > 0) midiTracks++;
    }
  }

  // Rough estimates (actual size depends on duration)
  totalSize += audioTracks * 2 * 1024 * 1024; // ~2MB per audio track
  totalSize += midiTracks * 10 * 1024; // ~10KB per MIDI track

  if (state.video?.videoFile) {
    totalSize += state.video.videoFile.size || 0;
  }

  return {
    totalBytes: totalSize,
    totalMB: (totalSize / 1024 / 1024).toFixed(2),
    audioTracks,
    midiTracks,
    hasVideo: !!state.video?.videoFile
  };
}

export default {
  exportSessionToGCS,
  importSessionFromGCS,
  calculateSessionSize
};
