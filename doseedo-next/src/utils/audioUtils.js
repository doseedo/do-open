/**
 * Audio utility functions
 * Helper functions for audio processing and manipulation
 */

/**
 * Detect if file is MIDI
 * @param {File} file - File to check
 * @returns {boolean} - True if MIDI file
 */
export function isMidiFile(file) {
  return file.name.endsWith('.mid') || file.name.endsWith('.midi');
}

/**
 * Detect if file is audio
 * @param {File} file - File to check
 * @returns {boolean} - True if audio file
 */
export function isAudioFile(file) {
  return file.type.startsWith('audio/') ||
         file.name.endsWith('.wav') ||
         file.name.endsWith('.mp3') ||
         file.name.endsWith('.ogg') ||
         file.name.endsWith('.flac');
}

/**
 * Format duration from seconds to MM:SS
 * @param {number} seconds - Duration in seconds
 * @returns {string} - Formatted duration
 */
export function formatDuration(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Create a blob URL from audio data
 * @param {ArrayBuffer} audioData - Audio data
 * @returns {string} - Blob URL
 */
export function createAudioBlobUrl(audioData) {
  const blob = new Blob([audioData], { type: 'audio/wav' });
  return URL.createObjectURL(blob);
}

/**
 * Download audio file
 * @param {string} url - URL of audio file
 * @param {string} filename - Desired filename
 */
export function downloadAudio(url, filename) {
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
