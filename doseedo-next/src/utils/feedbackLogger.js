/**
 * Feedback Logger Utility
 * Logs user feedback (likes/dislikes) for generated tracks
 * Stores logs in localStorage and provides download functionality
 */

const FEEDBACK_LOG_KEY = 'doseedo_generation_feedback_log';

/**
 * Extract process ID from audio file path
 * Expected format: /static/outputs/<process_id>/voice_<n>.wav
 * or similar patterns
 * @param {string} audioUrl - Audio file URL/path
 * @returns {string|null} - Process ID or null if not found
 */
function extractProcessId(audioUrl) {
  if (!audioUrl) return null;

  // Try to extract from various patterns
  // Pattern 1: /static/outputs/<process_id>/voice_<n>.wav
  // eslint-disable-next-line no-useless-escape
  const pattern1 = /\/static\/outputs\/([^\/]+)\//;
  const match1 = audioUrl.match(pattern1);
  if (match1) return match1[1];

  // Pattern 2: /outputs/<process_id>/
  // eslint-disable-next-line no-useless-escape
  const pattern2 = /\/outputs\/([^\/]+)\//;
  const match2 = audioUrl.match(pattern2);
  if (match2) return match2[1];

  // Pattern 3: Any UUID-like string in the path
  const pattern3 = /([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})/i;
  const match3 = audioUrl.match(pattern3);
  if (match3) return match3[1];

  // Pattern 4: Timestamp-based process ID (e.g., 20250108_123456)
  const pattern4 = /(\d{8}_\d{6})/;
  const match4 = audioUrl.match(pattern4);
  if (match4) return match4[1];

  console.warn('Could not extract process ID from audioUrl:', audioUrl);
  return null;
}

/**
 * Log feedback for a generated track
 * @param {Object} track - Track object with metadata
 * @param {boolean} isLiked - True for like, false for dislike
 * @returns {Object} - The logged entry
 */
export function logFeedback(track, isLiked) {
  if (!track) {
    console.warn('Cannot log feedback: no track provided');
    return null;
  }

  const processId = extractProcessId(track.audioUrl);

  const entry = {
    timestamp: new Date().toISOString(),
    processId: processId,
    audioUrl: track.audioUrl,
    trackName: track.name,
    feedback: isLiked ? 'like' : 'dislike',
    params: track.metadata?.params || null,
    trackId: track.id
  };

  // Get existing log
  const existingLog = getFullLog();

  // Add new entry
  existingLog.push(entry);

  // Save to localStorage
  try {
    localStorage.setItem(FEEDBACK_LOG_KEY, JSON.stringify(existingLog));
    console.log(`📝 Logged ${isLiked ? 'like' : 'dislike'} for process ${processId}`);
  } catch (error) {
    console.error('Failed to save feedback log to localStorage:', error);
  }

  return entry;
}

/**
 * Get the full feedback log
 * @returns {Array} - Array of feedback entries
 */
export function getFullLog() {
  try {
    const logJson = localStorage.getItem(FEEDBACK_LOG_KEY);
    if (!logJson) return [];
    return JSON.parse(logJson);
  } catch (error) {
    console.error('Failed to load feedback log from localStorage:', error);
    return [];
  }
}

/**
 * Get log entries for a specific process ID
 * @param {string} processId - Process ID to filter by
 * @returns {Array} - Array of feedback entries for this process
 */
export function getLogForProcess(processId) {
  const fullLog = getFullLog();
  return fullLog.filter(entry => entry.processId === processId);
}

/**
 * Get statistics about the feedback log
 * @returns {Object} - Statistics object
 */
export function getLogStats() {
  const log = getFullLog();
  const likes = log.filter(e => e.feedback === 'like').length;
  const dislikes = log.filter(e => e.feedback === 'dislike').length;
  const uniqueProcesses = new Set(log.map(e => e.processId).filter(Boolean)).size;

  return {
    total: log.length,
    likes,
    dislikes,
    uniqueProcesses,
    likeRate: log.length > 0 ? (likes / log.length * 100).toFixed(1) : 0
  };
}

/**
 * Download the feedback log as a JSON file
 */
export function downloadLog() {
  const log = getFullLog();
  const blob = new Blob([JSON.stringify(log, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `doseedo_feedback_log_${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
  console.log('📥 Feedback log downloaded');
}

/**
 * Download the feedback log as a CSV file
 */
export function downloadLogCSV() {
  const log = getFullLog();

  if (log.length === 0) {
    console.warn('No feedback entries to export');
    return;
  }

  // CSV headers
  const headers = ['Timestamp', 'Process ID', 'Audio URL', 'Track Name', 'Feedback', 'Track ID'];
  const rows = log.map(entry => [
    entry.timestamp,
    entry.processId || '',
    entry.audioUrl || '',
    entry.trackName || '',
    entry.feedback,
    entry.trackId || ''
  ]);

  // Build CSV content
  const csvContent = [
    headers.join(','),
    ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
  ].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `doseedo_feedback_log_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
  console.log('📥 Feedback log downloaded as CSV');
}

/**
 * Clear the entire feedback log
 * Use with caution!
 */
export function clearLog() {
  try {
    localStorage.removeItem(FEEDBACK_LOG_KEY);
    console.log('🗑️  Feedback log cleared');
    return true;
  } catch (error) {
    console.error('Failed to clear feedback log:', error);
    return false;
  }
}

/**
 * Import a feedback log from a JSON file
 * @param {File} file - JSON file to import
 * @returns {Promise<boolean>} - Success status
 */
export async function importLog(file) {
  try {
    const text = await file.text();
    const importedLog = JSON.parse(text);

    if (!Array.isArray(importedLog)) {
      throw new Error('Invalid log format: expected an array');
    }

    // Merge with existing log
    const existingLog = getFullLog();
    const mergedLog = [...existingLog, ...importedLog];

    localStorage.setItem(FEEDBACK_LOG_KEY, JSON.stringify(mergedLog));
    console.log(`📥 Imported ${importedLog.length} feedback entries`);
    return true;
  } catch (error) {
    console.error('Failed to import feedback log:', error);
    return false;
  }
}
