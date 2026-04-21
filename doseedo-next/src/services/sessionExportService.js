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

/**
 * Extract all audio and MIDI files from tracks
 */
async function extractTrackFiles(buses) {
  const files = [];

  for (const bus of buses) {
    for (const track of bus.tracks || []) {
      // Extract audio file if exists
      if (track.audioUrl) {
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
        // Convert MIDI data to MIDI file blob
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

    // Step 4: Upload all files to R2 (endpoint path kept as /api/upload/gcs
    // for URL back-compat — app/storage.py in the Fly auth service has
    // been R2-only for a while now).
    console.log(`📤 Uploading ${trackFiles.length + 1} files to R2…`);

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
