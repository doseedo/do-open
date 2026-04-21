/**
 * Save Service
 * Handles both local (localStorage) and cloud (GCS) saving
 */

import * as sessionService from './sessionService';
import * as sessionExportService from './sessionExportService';
import * as sessionAPI from './sessionAPI';
import { getCurrentUser } from './authService';

// Save status types
export const SaveStatus = {
  SAVED: 'saved',
  SAVING: 'saving',
  UNSAVED: 'unsaved',
  ERROR: 'error'
};

// Listeners for save status changes
const saveStatusListeners = [];

let currentSaveStatus = SaveStatus.SAVED;
let lastSaveTime = null;
let lastCloudSaveTime = null;

/**
 * Subscribe to save status changes
 */
export function onSaveStatusChange(callback) {
  saveStatusListeners.push(callback);
  // Immediately call with current status
  callback(currentSaveStatus, lastSaveTime, lastCloudSaveTime);

  // Return unsubscribe function
  return () => {
    const index = saveStatusListeners.indexOf(callback);
    if (index > -1) {
      saveStatusListeners.splice(index, 1);
    }
  };
}

/**
 * Update save status and notify listeners
 */
function updateSaveStatus(status, localTime = null, cloudTime = null) {
  currentSaveStatus = status;
  if (localTime) lastSaveTime = localTime;
  if (cloudTime) lastCloudSaveTime = cloudTime;

  saveStatusListeners.forEach(callback => {
    callback(status, lastSaveTime, lastCloudSaveTime);
  });
}

/**
 * Save session locally (localStorage)
 * This is fast and used for auto-save
 */
export async function saveLocal(projectName, state) {
  try {
    updateSaveStatus(SaveStatus.SAVING);

    const success = sessionService.saveSession(projectName, state);

    if (success) {
      const now = new Date();
      updateSaveStatus(SaveStatus.SAVED, now);
      console.log(`💾 Saved locally: ${projectName}`);
      return { success: true, savedAt: now };
    } else {
      updateSaveStatus(SaveStatus.ERROR);
      return { success: false, error: 'Failed to save to localStorage' };
    }
  } catch (error) {
    console.error('Local save error:', error);
    updateSaveStatus(SaveStatus.ERROR);
    return { success: false, error: error.message };
  }
}

/**
 * Save session to cloud (GCS)
 * This is slower and used for manual save / backup
 */
export async function saveToCloud(projectName, state, options = {}) {
  try {
    // Short-circuit when the user isn't signed in. Autosave runs on a 5s
    // interval regardless of auth state; without this guard every tick
    // used to `throw new Error('User not authenticated')` and spam the
    // console. Local save keeps working via quickSave -> saveToLocal.
    const user = getCurrentUser();
    if (!user || !user.id) {
      return { success: false, skipped: 'not-authenticated' };
    }

    updateSaveStatus(SaveStatus.SAVING);
    console.log(`☁️ Saving to cloud: ${projectName}`);

    // Check if session already exists in cloud
    let existingSession = null;
    try {
      const sessions = await sessionAPI.getUserSessions({ name: projectName });
      existingSession = sessions.sessions?.find(s => s.name === projectName);
    } catch (error) {
      // Session doesn't exist yet, that's fine
      console.log('No existing cloud session found');
    }

    // Export session to R2 (endpoint path kept as /api/upload/gcs for
    // historical reasons — see uploads.py, storage is R2 under the hood).
    const exportResult = await sessionExportService.exportSessionToGCS(
      state,
      projectName,
      {
        sessionId: existingSession?.session_id || options.sessionId,
        overwrite: !!existingSession
      }
    );

    const sessionData = {
      name: projectName,
      description: options.description || '',
      type: options.type || 'project',
      is_public: options.isPublic || false,
      session_id: exportResult.sessionId,
      gcs_base_path: exportResult.basePath,
      metadata: {
        ...exportResult.metadata,
        fileCount: exportResult.files.length,
        files: exportResult.files.map(f => ({
          path: f.path,
          gcs_url: f.gcsUrl,
          file_name: f.fileName,
          file_size: f.fileSize,
          type: f.type
        }))
      }
    };

    if (existingSession) {
      await sessionAPI.updateSession(existingSession.id, sessionData);
      console.log(`✅ Updated cloud session: ${projectName}`);
    } else {
      await sessionAPI.createSession(sessionData);
      console.log(`✅ Created cloud session: ${projectName}`);
    }

    const now = new Date();
    updateSaveStatus(SaveStatus.SAVED, null, now);

    return {
      success: true,
      savedAt: now,
      sessionId: exportResult.sessionId,
      fileCount: exportResult.files.length
    };

  } catch (error) {
    console.error('Cloud save error:', error);
    updateSaveStatus(SaveStatus.ERROR);
    return { success: false, error: error.message };
  }
}

/**
 * Save session to both local and cloud
 * This is the "full save" operation
 */
export async function saveEverywhere(projectName, state, options = {}) {
  try {
    updateSaveStatus(SaveStatus.SAVING);

    // Save locally first (fast)
    const localResult = await saveLocal(projectName, state);
    if (!localResult.success) {
      return localResult;
    }

    // Then save to cloud (slower)
    const cloudResult = await saveToCloud(projectName, state, options);

    return {
      success: cloudResult.success,
      local: localResult,
      cloud: cloudResult
    };

  } catch (error) {
    console.error('Save everywhere error:', error);
    updateSaveStatus(SaveStatus.ERROR);
    return { success: false, error: error.message };
  }
}

/**
 * Quick save - saves locally and queues cloud save
 * Good for Cmd+S / Ctrl+S shortcut
 */
export async function quickSave(projectName, state) {
  // Save locally immediately
  const localResult = await saveLocal(projectName, state);

  // Queue cloud save in background (don't wait)
  saveToCloud(projectName, state, { background: true }).catch(error => {
    console.warn('Background cloud save failed:', error);
  });

  return localResult;
}

/**
 * Get current save status
 */
export function getSaveStatus() {
  return {
    status: currentSaveStatus,
    lastSaveTime,
    lastCloudSaveTime
  };
}

/**
 * Mark session as unsaved (e.g., when user makes changes)
 */
export function markUnsaved() {
  updateSaveStatus(SaveStatus.UNSAVED);
}

export default {
  saveLocal,
  saveToCloud,
  saveEverywhere,
  quickSave,
  onSaveStatusChange,
  getSaveStatus,
  markUnsaved,
  SaveStatus
};
