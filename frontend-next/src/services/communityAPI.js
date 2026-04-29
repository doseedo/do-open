/**
 * Community API — creations, likes, favorites, profiles, downloads, fork
 * All endpoints route to the auth Cloud Run service (acauth.py)
 */

// ── Debounce helper ──

function debounce(fn, delay) {
  let timer = null;
  let pendingPromise = null;
  let pendingResolve = null;
  let pendingReject = null;

  return function (...args) {
    if (timer) {
      clearTimeout(timer);
    }
    if (!pendingPromise) {
      pendingPromise = new Promise((resolve, reject) => {
        pendingResolve = resolve;
        pendingReject = reject;
      });
    }
    const currentPromise = pendingPromise;
    timer = setTimeout(() => {
      const resolve = pendingResolve;
      const reject = pendingReject;
      pendingPromise = null;
      pendingResolve = null;
      pendingReject = null;
      timer = null;
      fn.apply(this, args).then(resolve).catch(reject);
    }, delay);
    return currentPromise;
  };
}

const BASE = '/api/creations';
const PROFILE = '/api/profiles';

async function fetchJSON(url, opts = {}) {
  const res = await fetch(url, { credentials: 'include', ...opts });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

// ── Creations CRUD ──

export function listCreations(params = {}) {
  const qs = new URLSearchParams(params).toString();
  return fetchJSON(`${BASE}${qs ? '?' + qs : ''}`);
}

export function getCreation(id) {
  return fetchJSON(`${BASE}/${id}`);
}

export function publishCreation(data) {
  return fetchJSON(BASE, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}

export function unpublishCreation(id) {
  return fetchJSON(`${BASE}/${id}`, { method: 'DELETE' });
}

// ── Likes ──

function _toggleLike(creationId) {
  return fetchJSON(`${BASE}/${creationId}/like`, { method: 'POST' });
}
export const toggleLike = debounce(_toggleLike, 300);

// ── Favorites ──

function _toggleFavorite(creationId) {
  return fetchJSON(`${BASE}/${creationId}/favorite`, { method: 'POST' });
}
export const toggleFavorite = debounce(_toggleFavorite, 300);

// ── Downloads ──

export function recordDownload(creationId) {
  return fetchJSON(`${BASE}/${creationId}/download`, { method: 'POST' });
}

// ── Fork ──

export function forkCreation(creationId) {
  return fetchJSON(`${BASE}/${creationId}/fork`, { method: 'POST' });
}

// ── User Library ──

export function getMyLikes(params = {}) {
  const qs = new URLSearchParams(params).toString();
  return fetchJSON(`/api/me/likes${qs ? '?' + qs : ''}`);
}

export function getMyFavorites(params = {}) {
  const qs = new URLSearchParams(params).toString();
  return fetchJSON(`/api/me/favorites${qs ? '?' + qs : ''}`);
}

export function getMyDownloads(params = {}) {
  const qs = new URLSearchParams(params).toString();
  return fetchJSON(`/api/me/downloads${qs ? '?' + qs : ''}`);
}

// ── Profiles ──

export function getProfile(username) {
  return fetchJSON(`${PROFILE}/${encodeURIComponent(username)}`);
}

export function updateProfile(data) {
  return fetchJSON(`${PROFILE}/me`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}

export function uploadAvatar(file) {
  const fd = new FormData();
  fd.append('file', file);
  return fetchJSON(`${PROFILE}/me/avatar`, { method: 'POST', body: fd });
}
