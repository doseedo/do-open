/**
 * Plugin Projects API Service
 * Handles save/load/CRUD for plugin creator projects, community browsing, and code generation.
 */

const API_BASE = '/_chat/api/plugin-projects';
const CODEGEN_BASE = '/_chat/api/codegen';

// ── Guest token management ──────────────────────────────────────────────────

function getGuestToken() {
  // The guest token is managed as an httpOnly cookie by the backend.
  // We can't read it from JS, but it's sent automatically with requests.
  return null;
}

async function fetchWithCredentials(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Project CRUD ────────────────────────────────────────────────────────────

/**
 * Save or update a plugin project.
 * @param {Object} project - Project data
 * @param {string} [project.id] - If provided, updates existing project
 * @param {string} project.name
 * @param {string} [project.description]
 * @param {Object} project.plugin_config - Canvas config (width, height, bgColor, etc.)
 * @param {Array} project.components - PluginLang components array
 * @param {Object} [project.dsp_config] - DSPLang config
 * @param {string} [project.thumbnail_data] - Base64 canvas snapshot
 * @param {Array} [project.tags]
 * @returns {Promise<{id: string, slug: string, created?: boolean, updated?: boolean}>}
 */
export async function saveProject(project) {
  return fetchWithCredentials(API_BASE, {
    method: 'POST',
    body: JSON.stringify(project),
  });
}

/**
 * Load a plugin project by ID or slug.
 * @param {string} idOrSlug
 * @returns {Promise<Object>} Full project data
 */
export async function loadProject(idOrSlug) {
  return fetchWithCredentials(`${API_BASE}/${idOrSlug}`);
}

/**
 * Delete a plugin project.
 * @param {string} projectId
 * @returns {Promise<{deleted: boolean}>}
 */
export async function deleteProject(projectId) {
  return fetchWithCredentials(`${API_BASE}/${projectId}`, {
    method: 'DELETE',
  });
}

// ── Listing ─────────────────────────────────────────────────────────────────

/**
 * List the current user's (or guest's) plugin projects.
 * @returns {Promise<Array>}
 */
export async function listMyProjects() {
  return fetchWithCredentials(`${API_BASE}/my`);
}

/**
 * Browse public community plugin projects.
 * @param {Object} [params]
 * @param {string} [params.search]
 * @param {string} [params.tag]
 * @param {string} [params.sort] - "newest", "popular", "name"
 * @param {number} [params.limit]
 * @param {number} [params.offset]
 * @returns {Promise<{total: number, projects: Array}>}
 */
export async function listCommunityProjects(params = {}) {
  const query = new URLSearchParams();
  if (params.search) query.set('search', params.search);
  if (params.tag) query.set('tag', params.tag);
  if (params.sort) query.set('sort', params.sort);
  if (params.limit) query.set('limit', String(params.limit));
  if (params.offset) query.set('offset', String(params.offset));
  const qs = query.toString();
  return fetchWithCredentials(`${API_BASE}/community${qs ? `?${qs}` : ''}`);
}

// ── Publishing ──────────────────────────────────────────────────────────────

/**
 * Publish or unpublish a plugin project to the community.
 * @param {string} projectId
 * @param {Object} data
 * @param {boolean} data.is_public
 * @param {string} [data.description]
 * @param {Array} [data.tags]
 * @returns {Promise<{is_public: boolean, slug: string}>}
 */
export async function publishProject(projectId, data) {
  return fetchWithCredentials(`${API_BASE}/${projectId}/publish`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

/**
 * Record a download of a community project (increments count).
 * @param {string} projectId
 * @returns {Promise<{download_count: number}>}
 */
export async function recordDownload(projectId) {
  return fetchWithCredentials(`${API_BASE}/${projectId}/download`, {
    method: 'POST',
  });
}

// ── Guest migration ─────────────────────────────────────────────────────────

/**
 * Migrate guest projects to the authenticated user. Call after login.
 * @returns {Promise<{migrated: number}>}
 */
export async function migrateGuestProjects() {
  return fetchWithCredentials(`${API_BASE}/migrate-guest`, {
    method: 'POST',
  });
}

// ── Code generation ─────────────────────────────────────────────────────────

/**
 * Generate JUCE C++ code from DSP config + optional UI layout.
 * @param {Object} dspConfig - DSPLang JSON
 * @param {Object} [uiLayout] - { pluginConfig, components } from PluginLang
 * @returns {Promise<{files: Object}>} - { files: { "PluginProcessor.h": "...", ... } }
 */
export async function generateCode(dspConfig, uiLayout = null) {
  const body = { ...dspConfig };
  if (uiLayout) {
    body.ui_layout = uiLayout;
  }
  return fetchWithCredentials(`${CODEGEN_BASE}/generate`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}
