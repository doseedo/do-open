/**
 * Session Service - Handles project/session persistence using localStorage
 *
 * Session Format:
 * {
 *   projectName: string,
 *   timestamp: number,
 *   state: {
 *     buses: [],
 *     video: {},
 *     generationParams: {},
 *     ... all app state
 *   }
 * }
 */

const SESSION_PREFIX = 'session-';
const PROJECTS_KEY = 'projects';
const ACTIVE_PROJECT_KEY = 'activeProject';
const LAST_SESSION_KEY = 'lastSession';

/**
 * Get list of all project names
 */
export function getProjects() {
  try {
    return JSON.parse(localStorage.getItem(PROJECTS_KEY)) || [];
  } catch (error) {
    console.error('Error loading projects:', error);
    return [];
  }
}

// Metadata fields that are typed-array / binary blobs — cheap to
// recompute but catastrophic to serialize. `JSON.stringify(Float32Array)`
// produces an object with one numeric string key per sample, which blows
// a 12 KB envelope into ~150 KB of JSON. Over many stem tracks that
// exhausts the 5-10 MB localStorage quota and blocks the main thread
// on save — which is why the studio tab was hanging on close.
const HEAVY_METADATA_FIELDS = [
  'envelopeData',     // Float32Array from rmsDemucs / sem4Decoder
  'latent',           // per-stem distill_demucs latent [64*T] Float32Array
  'latents',          // per-stem oobleck latents
  'stemLatents',
  'waveformBuffer',   // pre-decoded stereo PCM
  'midiBuffer',       // ArrayBuffer of a MIDI file
  'audioBuffer',      // AudioBuffer / ArrayBuffer
];

function stripHeavyTrackMetadata(state) {
  // Shallow-clone the buses spine; deep-clone only the parts that carry
  // the heavy fields. Cheaper than structuredClone on the full tree.
  if (!state || !Array.isArray(state.buses)) return state;
  return {
    ...state,
    buses: state.buses.map((bus) => ({
      ...bus,
      tracks: (bus.tracks || []).map((track) => {
        if (!track?.metadata) return track;
        let metadata = track.metadata;
        let cloned = false;
        for (const f of HEAVY_METADATA_FIELDS) {
          if (metadata[f] != null) {
            if (!cloned) { metadata = { ...metadata }; cloned = true; }
            delete metadata[f];
          }
        }
        return cloned ? { ...track, metadata } : track;
      }),
    })),
  };
}

/**
 * Save a session to localStorage
 */
export function saveSession(projectName, state) {
  try {
    const sessionKey = `${SESSION_PREFIX}${projectName}`;
    const sessionData = {
      projectName,
      timestamp: Date.now(),
      state: stripHeavyTrackMetadata(state),
    };

    // Save session data
    localStorage.setItem(sessionKey, JSON.stringify(sessionData));

    // Add to projects list if not already there
    const projects = getProjects();
    if (!projects.includes(projectName)) {
      projects.unshift(projectName); // Add to beginning
      localStorage.setItem(PROJECTS_KEY, JSON.stringify(projects));
    }

    console.log(`✅ Session saved: ${projectName}`);
    return true;
  } catch (error) {
    console.error('Error saving session:', error);
    return false;
  }
}

/**
 * Load a session from localStorage
 */
export function loadSession(projectName) {
  try {
    const sessionKey = `${SESSION_PREFIX}${projectName}`;
    const sessionData = localStorage.getItem(sessionKey);

    if (!sessionData) {
      console.warn(`No session found for: ${projectName}`);
      return null;
    }

    const parsed = JSON.parse(sessionData);
    // Migration: strip the legacy hardcoded chord progression so the
    // chord row always loads empty. Users re-detect via the sidebar.
    if (parsed?.state?.chordTrack) {
      parsed.state.chordTrack = { ...parsed.state.chordTrack, chords: {} };
    }
    if (parsed?.state?.chords) {
      parsed.state.chords = {};
    }
    console.log(`✅ Session loaded: ${projectName}`);
    return parsed;
  } catch (error) {
    console.error('Error loading session:', error);
    return null;
  }
}

/**
 * Delete a project and its session
 */
export function deleteProject(projectName) {
  try {
    // Remove session data
    const sessionKey = `${SESSION_PREFIX}${projectName}`;
    localStorage.removeItem(sessionKey);

    // Remove from projects list
    const projects = getProjects();
    const updatedProjects = projects.filter(p => p !== projectName);
    localStorage.setItem(PROJECTS_KEY, JSON.stringify(updatedProjects));

    console.log(`✅ Project deleted: ${projectName}`);
    return true;
  } catch (error) {
    console.error('Error deleting project:', error);
    return false;
  }
}

/**
 * Rename a project
 */
export function renameProject(oldName, newName) {
  try {
    // Load old session
    const sessionData = loadSession(oldName);
    if (!sessionData) {
      return false;
    }

    // Save with new name
    sessionData.projectName = newName;
    saveSession(newName, sessionData.state);

    // Delete old session
    deleteProject(oldName);

    // Update active project if it was the renamed one
    if (getActiveProject() === oldName) {
      setActiveProject(newName);
    }

    console.log(`✅ Project renamed: ${oldName} → ${newName}`);
    return true;
  } catch (error) {
    console.error('Error renaming project:', error);
    return false;
  }
}

/**
 * Get/Set active project
 */
export function getActiveProject() {
  return localStorage.getItem(ACTIVE_PROJECT_KEY);
}

export function setActiveProject(projectName) {
  localStorage.setItem(ACTIVE_PROJECT_KEY, projectName);
  localStorage.setItem(LAST_SESSION_KEY, projectName);
}

/**
 * Clear active project
 */
export function clearActiveProject() {
  localStorage.removeItem(ACTIVE_PROJECT_KEY);
}

/**
 * Get last session (for auto-load)
 */
export function getLastSession() {
  return localStorage.getItem(LAST_SESSION_KEY);
}

/**
 * Auto-save helper - creates a debounced save function
 */
export function createAutoSave(projectName, delay = 3000) {
  let timeout = null;

  return (state) => {
    if (timeout) {
      clearTimeout(timeout);
    }

    timeout = setTimeout(() => {
      saveSession(projectName, state);
      console.log(`💾 Auto-saved: ${projectName}`);
    }, delay);
  };
}
