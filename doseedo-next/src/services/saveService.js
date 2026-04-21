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

    // Export session to GCS
    const exportResult = await sessionExportService.exportSessionToGCS(
      state,
      projectName,
      {
        sessionId: existingSession?.session_id || options.sessionId,
        overwrite: !!existingSession
      }
    );

    // Update or create session record in backend
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
      // Update existing session
      await sessionAPI.updateSession(existingSession.id, sessionData);
      console.log(`✅ Updated cloud session: ${projectName}`);
    } else {
      // Create new session
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
 * Quick save — local + debounced backend autosave.
 *
 * Heavy GCS/R2 file re-uploads (`saveToCloud`) no longer run per tick; they
 * only fire from the explicit "Post" / "Save As…" flows. Autosave pushes
 * the project-state JSON to R2 via PUT /api/sessions/{id}/state, which is
 * cheap and keeps the user's work synced across devices.
 */
export async function quickSave(projectName, state) {
  const localResult = await saveLocal(projectName, state);
  try {
    queueBackendAutosave(projectName, state);
  } catch (e) {
    console.warn('[saveService] queueBackendAutosave threw:', e?.message);
  }
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

// ── Backend autosave (cross-device) ────────────────────────────────────
//
// Debounced PUT /api/sessions/{id}/state. The state JSON gets written to R2
// so opening the project on another device restores exactly what the user
// left behind — audio URLs embedded in the state still resolve via the same
// R2 keys, so no per-file re-upload is needed.
//
//   • 2.5s idle debounce, 750ms leading edge on the first edit burst.
//   • Requires a remote session id — `ensureSessionIdFor` must run on the
//     user's first explicit save (Save / Save As) so autosave has something
//     to target.
//   • No-ops when signed out or when there's no remote id yet.

const AUTOSAVE_IDLE_MS = 2500;
const AUTOSAVE_LEADING_MS = 750;

let _autosaveTimer = null;
let _autosaveLeadingTimer = null;
let _autosaveLastFiredAt = 0;
let _autosaveInFlight = false;
let _lastQueuedState = null;
let _lastQueuedName = null;

async function performBackendAutosave(projectName, state) {
  const user = getCurrentUser();
  if (!user || !user.id) return { skipped: 'not-authenticated' };

  const remoteId = sessionService.getSessionIdForName(projectName);
  if (!remoteId) return { skipped: 'no-remote-session' };

  try {
    _autosaveInFlight = true;
    updateSaveStatus(SaveStatus.SAVING);
    await sessionAPI.putSessionState(remoteId, state);
    _autosaveLastFiredAt = Date.now();
    const now = new Date();
    updateSaveStatus(SaveStatus.SAVED, lastSaveTime, now);
    return { success: true, savedAt: now };
  } catch (err) {
    console.warn('[saveService] backend autosave failed', err?.message);
    updateSaveStatus(SaveStatus.ERROR);
    return { success: false, error: err?.message };
  } finally {
    _autosaveInFlight = false;
  }
}

export function queueBackendAutosave(projectName, state) {
  if (!projectName || !state) return;
  _lastQueuedName = projectName;
  _lastQueuedState = state;
  const now = Date.now();

  if (
    !_autosaveLeadingTimer &&
    !_autosaveInFlight &&
    now - _autosaveLastFiredAt > AUTOSAVE_IDLE_MS
  ) {
    _autosaveLeadingTimer = setTimeout(() => {
      _autosaveLeadingTimer = null;
      if (_lastQueuedName && _lastQueuedState) {
        performBackendAutosave(_lastQueuedName, _lastQueuedState).catch(() => {});
      }
    }, AUTOSAVE_LEADING_MS);
  }

  if (_autosaveTimer) clearTimeout(_autosaveTimer);
  _autosaveTimer = setTimeout(() => {
    _autosaveTimer = null;
    if (_lastQueuedName && _lastQueuedState) {
      performBackendAutosave(_lastQueuedName, _lastQueuedState).catch(() => {});
    }
  }, AUTOSAVE_IDLE_MS);
}

export async function flushBackendAutosave() {
  if (_autosaveTimer) { clearTimeout(_autosaveTimer); _autosaveTimer = null; }
  if (_autosaveLeadingTimer) {
    clearTimeout(_autosaveLeadingTimer);
    _autosaveLeadingTimer = null;
  }
  if (_lastQueuedName && _lastQueuedState) {
    return performBackendAutosave(_lastQueuedName, _lastQueuedState);
  }
  return { skipped: 'nothing-queued' };
}

/**
 * Ensure a remote session id exists for `projectName`. Creates one on the
 * backend via POST /api/sessions + seeds the initial state via
 * PUT /api/sessions/{id}/state. Call from the Save / Save As UI flow.
 */
export async function ensureSessionIdFor(projectName, initialState, options = {}) {
  if (!projectName) return null;
  const existing = sessionService.getSessionIdForName(projectName);
  if (existing) return existing;

  const user = getCurrentUser();
  if (!user || !user.id) return null;

  try {
    const created = await sessionAPI.createSession({
      name: projectName,
      description: options.description || '',
      type: options.type || 'project',
      is_public: !!options.isPublic,
    });
    const id = created?.id || created?.session_id;
    if (!id) return null;
    sessionService.rememberSessionId(projectName, id);
    if (initialState) {
      try {
        await sessionAPI.putSessionState(id, initialState);
      } catch (e) {
        console.warn('[saveService] initial state upload failed', e?.message);
      }
    }
    return id;
  } catch (err) {
    console.warn('[saveService] ensureSessionIdFor failed', err?.message);
    return null;
  }
}

export default {
  saveLocal,
  saveToCloud,
  saveEverywhere,
  quickSave,
  onSaveStatusChange,
  getSaveStatus,
  markUnsaved,
  queueBackendAutosave,
  flushBackendAutosave,
  ensureSessionIdFor,
  SaveStatus
};
