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
 * @param {Object} [images] - { "bg_image.png": "data:image/png;base64,..." } image assets
 * @returns {Promise<{files: Object, images?: Object}>}
 */
export async function generateCode(dspConfig, uiLayout = null, images = null) {
  const body = { ...dspConfig };
  if (uiLayout) {
    body.ui_layout = uiLayout;
  }
  if (images) {
    body.images = images;
  }
  return fetchWithCredentials(`${CODEGEN_BASE}/generate`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

// ── Plugin Build (Mac Mini) ────────────────────────────────────────────────

/**
 * Build a plugin from pre-generated JUCE files. Returns a zip blob (VST3 + AU).
 * @param {string} pluginName - Plugin name
 * @param {Object} files - { "CMakeLists.txt": "...", "PluginProcessor.h": "...", ... }
 * @param {Object} [images] - { "bg_image.png": "<base64>" } binary image assets
 * @returns {Promise<Blob>} - Zip file blob
 */
export async function buildPlugin(pluginName, files, images = null) {
  const body = { plugin_name: pluginName, files };
  if (images) body.images = images;
  const res = await fetch(`${CODEGEN_BASE}/build`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Build failed: HTTP ${res.status}`);
  }
  return res.blob();
}

/**
 * Build a plugin with streaming SSE progress logs.
 * Falls back to non-streaming build if SSE endpoint is unavailable.
 * @param {string} pluginName
 * @param {Object} files - JUCE source files
 * @param {Object} [images] - Binary image assets
 * @param {Object} callbacks - { onLog: (line) => void, onStage: (stage) => void }
 * @returns {Promise<Blob>} - Zip file blob
 */
export async function buildPluginStream(pluginName, files, images = null, { onLog, onStage } = {}) {
  try {
    const body = { plugin_name: pluginName, files };
    if (images) body.images = images;
    const res = await fetch(`${CODEGEN_BASE}/build-stream`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Build failed: HTTP ${res.status}`);
    }
    const contentType = res.headers.get('content-type') || '';
    // If the server supports SSE streaming (text/event-stream), read lines
    if (contentType.includes('text/event-stream')) {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let downloadUrl = null;
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const evt = JSON.parse(line.slice(6));
              if (evt.type === 'log' && onLog) onLog(evt.message);
              if (evt.type === 'stage' && onStage) onStage(evt.stage);
              if (evt.type === 'done') downloadUrl = evt.download_url;
              if (evt.type === 'error') throw new Error(evt.message);
            } catch (e) {
              if (e.message && !e.message.includes('JSON')) throw e;
            }
          }
        }
      }
      if (downloadUrl) {
        const dlRes = await fetch(downloadUrl, { credentials: 'include' });
        if (!dlRes.ok) throw new Error('Download failed');
        return dlRes.blob();
      }
      throw new Error('Build completed but no download URL received');
    }
    // Non-streaming fallback — response is the zip directly
    if (onStage) onStage('Building (no streaming)...');
    return res.blob();
  } catch (err) {
    // If stream endpoint doesn't exist, fall back to regular build
    if (err.message?.includes('404') || err.message?.includes('Not Found')) {
      if (onStage) onStage('Building...');
      return buildPlugin(pluginName, files);
    }
    throw err;
  }
}

/**
 * Build a plugin with AI auto-fix for compile errors.
 * On failure, feeds compiler errors to GPT-4o to fix the C++ code, then retries.
 * @param {string} pluginName
 * @param {Object} files - filename -> content
 * @param {Object} [images] - filename -> base64 content
 * @param {Object} callbacks - { onLog, onStage, onAutoFix }
 * @returns {Promise<Blob>} The built plugin zip
 */
export async function buildPluginAutoFix(pluginName, files, images = null, { onLog, onStage, onAutoFix } = {}) {
  const body = { plugin_name: pluginName, files, max_fix_rounds: 2 };
  if (images) body.images = images;

  const res = await fetch(`${CODEGEN_BASE}/build-autofix`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Build failed: HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let downloadUrl = null;
  let fixedFiles = null;
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const evt = JSON.parse(line.slice(6));
          if (evt.type === 'log' && onLog) onLog(evt.message);
          if (evt.type === 'stage' && onStage) onStage(evt.stage);
          if (evt.type === 'done') {
            downloadUrl = evt.download_url;
            fixedFiles = evt.fixed_files || null;
            if (fixedFiles && onAutoFix) onAutoFix(fixedFiles);
          }
          if (evt.type === 'error') throw new Error(evt.message);
        } catch (e) {
          if (e.message && !e.message.includes('JSON')) throw e;
        }
      }
    }
  }

  if (downloadUrl) {
    const dlRes = await fetch(downloadUrl, { credentials: 'include' });
    if (!dlRes.ok) throw new Error('Download failed');
    return dlRes.blob();
  }
  throw new Error('Build completed but no download URL received');
}
