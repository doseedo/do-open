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

// Single-flight mutex for cloud save. The autosave loop fires every 8s
// but a session with 8 stems uploads 8 multi-MB files serially through
// the Fly R2 gateway — easily longer than the autosave interval. Without
// this guard, uploads stack up unboundedly and the UI drowns in "Saving
// to cloud" log spam that never completes.
let cloudSaveInFlight = false;

// Cache of the Neon session row id keyed by projectName so subsequent
// autosaves go through sessionAPI.updateSession instead of re-creating.
// The Fly rate limit is 20 session CREATEs per user per hour — an 8s
// autosave would burn that in 2.5 minutes without this.
const _projectSessionRowId = new Map();
const _projectSessionId = new Map();

// Dirty-check fingerprint of the last state that was successfully
// pushed to R2. The 8s autosave was uploading even when nothing
// changed — the user was watching "Uploading 2 files to R2…" scroll
// every 8s while sitting idle. We hash a stripped copy of the state
// and skip the cloud export when the hash matches the last one we
// sent for the same project. Buffer/latent/onset data is already
// stripped by sessionService's allowlist, so the hash stays stable
// across in-flight analysis writes that don't change user-visible
// content.
const _lastCloudSaveHash = new Map();

function _fingerprintForCloud(state) {
  try {
    // Only hash what would actually get uploaded. Using the same
    // allowlisted shape that sessionService.saveSession uses keeps the
    // hash stable across analysis metadata churn.
    const stripped = sessionService._stripForCloud
      ? sessionService._stripForCloud(state)
      : state;
    const s = JSON.stringify(stripped);
    // FNV-1a 32-bit — fast, good enough for equality checks.
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
    }
    return `${s.length}:${h.toString(16)}`;
  } catch (_) {
    return null;
  }
}

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

    // Skip if another cloud save is still uploading. Autosave reruns
    // every 8s; for multi-stem sessions one upload pass can take longer.
    if (cloudSaveInFlight) {
      return { success: false, skipped: 'cloud-save-in-flight' };
    }

    // Dirty check: bail out silently when the serialized-for-cloud
    // fingerprint matches the last successful push for this project.
    // Explicit user saves (Cmd+S through quickSave → saveToCloud) still
    // flow through because options.force=true.
    const fp = _fingerprintForCloud(state);
    if (fp && !options.force && _lastCloudSaveHash.get(projectName) === fp) {
      return { success: true, skipped: 'unchanged' };
    }

    cloudSaveInFlight = true;

    updateSaveStatus(SaveStatus.SAVING);
    console.log(`☁️ Saving to cloud: ${projectName}`);

    // Look up the existing session row. Cached first — the Fly
    // list_sessions endpoint doesn't filter by name and the response
    // shape is {items, total}, not {sessions, ...}, so the original
    // round-trip was silently always returning null and creating a
    // brand-new session on every autosave tick (burning the 20/hour
    // rate limit in ~3 minutes). If the cache is empty we fall through
    // to a one-shot paginated match by name.
    let existingRowId = _projectSessionRowId.get(projectName) || null;
    let cachedSessionId = _projectSessionId.get(projectName) || null;

    if (!existingRowId) {
      try {
        const list = await sessionAPI.getUserSessions({ limit: 200 });
        const items = list?.items || list?.sessions || [];
        const match = items.find((s) => s.name === projectName);
        if (match) {
          existingRowId = match.id;
          cachedSessionId = match.session_id || null;
          _projectSessionRowId.set(projectName, existingRowId);
          if (cachedSessionId) _projectSessionId.set(projectName, cachedSessionId);
        }
      } catch (_) {
        // First-ever save for this project — ok to fall through to create.
      }
    }

    // Export session to R2 via /api/upload/r2 (see uploads.py).
    const exportResult = await sessionExportService.exportSessionToGCS(
      state,
      projectName,
      {
        sessionId: cachedSessionId || options.sessionId,
        overwrite: !!existingRowId,
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

    if (existingRowId) {
      await sessionAPI.updateSession(existingRowId, sessionData);
      console.log(`✅ Updated cloud session: ${projectName}`);
    } else {
      const created = await sessionAPI.createSession(sessionData);
      // Remember the row id + R2 session_id so future autosaves
      // UPDATE this same row instead of creating another.
      if (created?.id) _projectSessionRowId.set(projectName, created.id);
      if (exportResult.sessionId) _projectSessionId.set(projectName, exportResult.sessionId);
      console.log(`✅ Created cloud session: ${projectName}`);
    }

    // Remember the fingerprint so identical-state autosaves can skip.
    if (fp) _lastCloudSaveHash.set(projectName, fp);

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
  } finally {
    cloudSaveInFlight = false;
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
