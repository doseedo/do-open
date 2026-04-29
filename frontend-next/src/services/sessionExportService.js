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

import * as r2UploadService from './r2UploadService';
import { getCurrentUser } from './authService';

/**
 * Generate a unique session ID
 */
function generateSessionId() {
  return `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Convert audio URL (blob or data URL) to File object
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

// Cache of already-uploaded files for this session, keyed by path. Each
// entry remembers the audioUrl at upload time plus the R2 result so we
// can reuse it across saves. Cleared when a new sessionId appears.
//
// Without this, moving a single track to a new position re-uploaded
// every stem's multi-MB audio on every autosave tick (saveService's
// fingerprint flips because startPosition changed, cloud save fires,
// this function re-fetches + re-uploads every audioUrl blindly).
const _uploadCache = { sessionId: null, byPath: new Map() };

function _resetCacheIfNewSession(sessionId) {
  if (_uploadCache.sessionId !== sessionId) {
    _uploadCache.sessionId = sessionId;
    _uploadCache.byPath = new Map();
  }
}

/**
 * Extract all audio and MIDI files from tracks. For audio tracks whose
 * audioUrl matches what we already uploaded this session, we emit a
 * cache-hit marker instead of fetching + rebuilding the File — the
 * export loop skips the upload entirely.
 */
async function extractTrackFiles(buses) {
  const files = [];

  for (const bus of buses) {
    for (const track of bus.tracks || []) {
      // Audio: cache on (path, audioUrl). Blob URLs are stable across
      // the session; /separate-stems/download/<id>/... is stable per
      // demucs run. MIDI and parameter edits don't touch audioUrl.
      if (track.audioUrl) {
        const ext = track.type === 'audio' ? 'wav' : 'mp3';
        const path = `tracks/${track.id}/audio.${ext}`;
        const cached = _uploadCache.byPath.get(path);
        if (cached && cached.audioUrl === track.audioUrl && cached.result) {
          files.push({ cached: cached.result, path, trackId: track.id, type: 'audio', audioUrl: track.audioUrl });
        } else {
          const audioFile = await urlToFile(
            track.audioUrl,
            `track-${track.id}.${ext}`,
            `audio/${ext === 'wav' ? 'wav' : 'mpeg'}`
          );
          if (audioFile) {
            files.push({ file: audioFile, path, trackId: track.id, type: 'audio', audioUrl: track.audioUrl });
          }
        }
      }

      // MIDI file — rebuilt from notes each time; the midiData content
      // hash would be the right cache key but tracking notes equality
      // cheaply is awkward, so just always re-export. It's small.
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
  // Create a clean copy of buses without audio URLs (files will be uploaded separately)
  const cleanBuses = state.buses.map(bus => ({
    ...bus,
    tracks: bus.tracks.map(track => ({
      ...track,
      // Remove blob URLs - we'll store file paths instead
      audioUrl: track.audioUrl ? `tracks/${track.id}/audio` : null,
      // Keep MIDI data reference
      midiData: track.midiData ? {
        ...track.midiData,
        // Keep essential MIDI data but reference the file
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


    // Step 1: Extract all track files (audio and MIDI)
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
    const sessionData = prepareSessionData(state, projectName);
    const sessionJsonBlob = new Blob([JSON.stringify(sessionData, null, 2)], {
      type: 'application/json'
    });
    const sessionJsonFile = new File([sessionJsonBlob], 'session.json', {
      type: 'application/json'
    });

    // Step 4: Upload to R2 (endpoint path kept as /api/upload/gcs for URL
    // back-compat — app/storage.py in the Fly auth service has been
    // R2-only for a while). Skip already-uploaded files whose audioUrl
    // didn't change since the last save.
    _resetCacheIfNewSession(sessionId);
    const toUpload = trackFiles.filter((tf) => !tf.cached);

    const uploadPromises = [];

    // Upload session.json (always — metadata changes every save).
    uploadPromises.push(
      r2UploadService.uploadToR2(
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

    // Emit reused entries immediately so the result array has them in order.
    for (const tf of trackFiles) {
      if (tf.cached) {
        uploadPromises.push(Promise.resolve({
          ...tf.cached, path: tf.path, trackId: tf.trackId, type: tf.type,
        }));
      }
    }

    // Upload only files that aren't cached.
    for (const trackFile of toUpload) {
      uploadPromises.push(
        r2UploadService.uploadToR2(
          trackFile.file,
          'session',
          {
            sessionId,
            projectName,
            trackId: trackFile.trackId,
            filePath: `${basePath}/${trackFile.path}`,
            fileType: trackFile.type
          }
        ).then(result => {
          // Cache for future saves (audio files only — midi is re-exported).
          if (trackFile.type === 'audio' && trackFile.audioUrl) {
            _uploadCache.byPath.set(trackFile.path, {
              audioUrl: trackFile.audioUrl,
              result: { ...result, trackId: trackFile.trackId, type: trackFile.type },
            });
          }
          return { ...result, ...trackFile };
        })
      );
    }

    // Upload video file if exists
    if (videoFile) {
      uploadPromises.push(
        r2UploadService.uploadToR2(
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

    return result;

  } catch (error) {
    console.error('❌ Session export failed:', error);
    throw error;
  }
}

/**
 * Import session from GCS
 * @param {string} sessionId - Session ID to import
 * @returns {Promise<object>} Restored session state
 */
export async function importSessionFromGCS(sessionId) {
  try {
    const user = getCurrentUser();
    if (!user || !user.id) {
      throw new Error('User not authenticated');
    }

    const basePath = `users/${user.id}/sessions/${sessionId}`;
    console.log(`📥 Importing session from ${basePath}`);

    // TODO: Implement session import
    // This would:
    // 1. Download session.json from GCS
    // 2. Parse the metadata
    // 3. Download all referenced track files
    // 4. Recreate blob URLs for audio files
    // 5. Return complete state object

    throw new Error('Session import not yet implemented');

  } catch (error) {
    console.error('❌ Session import failed:', error);
    throw error;
  }
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
