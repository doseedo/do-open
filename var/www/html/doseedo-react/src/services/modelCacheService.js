/**
 * Model Cache Service — IndexedDB-backed cache for large ONNX model bytes.
 *
 * Why: the oobleck VAE encoder weights are 161 MB. Relying on the HTTP cache
 * alone works on the first few reloads but browsers evict entries that big
 * under memory pressure, and a cold 161 MB fetch over a shaky connection is
 * a 30-60s delay before the user can use the feature.
 *
 * Parallel to audioCacheService.js but tuned for models:
 *   - Cache never expires (models are immutable per URL; R2 serves with
 *     `Cache-Control: immutable` + a stable filename per version)
 *   - Much larger individual entries (~200 MB) allowed
 *   - Keyed by full URL so model swaps (e.g. oobleck fp16 → fp32) naturally
 *     miss the old cache entry
 *
 * Usage:
 *   import { fetchModelWithCache } from './modelCacheService';
 *   const bytes = await fetchModelWithCache('/static/models/foo.onnx');
 *   // bytes is an ArrayBuffer ready for ort.InferenceSession.create
 */

const DB_NAME = 'doseedo-model-cache';
const DB_VERSION = 1;
const STORE_NAME = 'onnx-models';

let _db = null;
let _openPromise = null;

function openDB() {
  if (_db) return Promise.resolve(_db);
  if (_openPromise) return _openPromise;

  _openPromise = new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('indexedDB unavailable'));
      return;
    }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(req.error);
    req.onsuccess = () => { _db = req.result; resolve(_db); };
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'url' });
      }
    };
  });
  return _openPromise;
}

/**
 * Look up a cached model. Returns the stored ArrayBuffer or null.
 */
export async function getCachedModel(url) {
  try {
    const db = await openDB();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).get(url);
      req.onerror = () => reject(req.error);
      req.onsuccess = () => {
        const row = req.result;
        resolve(row ? row.bytes : null);
      };
    });
  } catch (e) {
    // Never let a cache miss block model loading — fall through to network.
    return null;
  }
}

/**
 * Store model bytes under a URL key. Fire-and-forget — if the browser refuses
 * (quota, private-mode, etc.) we still returned the bytes to the caller, so
 * the feature works, it just loses the persistent cache benefit.
 */
export async function putCachedModel(url, bytes) {
  try {
    const db = await openDB();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put({
        url,
        bytes,
        size: bytes.byteLength,
        cached_at: Date.now(),
      });
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  } catch (e) {
    // Benign; log once so devs see when quota evicts.
    // eslint-disable-next-line no-console
    console.warn('[modelCache] put failed for', url, '—', e?.message || e);
  }
}

/**
 * High-level: fetch a URL, but check the IndexedDB cache first and stash the
 * response on a miss. Calls `onProgress({bytesLoaded, bytesTotal})` during
 * the network fetch so the UI can show a progress bar the first time.
 *
 * Returns the raw bytes as an ArrayBuffer.
 */
/** Reject obviously-wrong cached payloads. A Next.js HTML fallback (served
 * before /static/models/* rewrites deployed) starts with `<` — if we cached
 * that once, every subsequent load hands ORT the HTML and `InferenceSession.
 * create` throws "protobuf parsing failed". Check the first byte before
 * trusting the cache, and on ONNX specifically also sniff the magic bytes. */
function _looksLikeModelBytes(buf, url) {
  if (!buf || buf.byteLength < 16) return false;
  const u8 = new Uint8Array(buf, 0, Math.min(64, buf.byteLength));
  // If the URL is an .onnx graph file (not a weights .data sibling), the
  // first real byte should not be ASCII '<' (HTML fallback).
  if (url.endsWith('.onnx') && u8[0] === 0x3C) return false;
  return true;
}

async function _deleteCachedModel(url) {
  try {
    const db = await openDB();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).delete(url);
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
    });
  } catch (_) { /* ignore — benign */ }
}

export async function fetchModelWithCache(url, onProgress = null) {
  const cached = await getCachedModel(url);
  if (cached && _looksLikeModelBytes(cached, url)) {
    // Mimic a "100% loaded from cache" progress event so UI code that watches
    // the callback doesn't hang.
    if (onProgress) onProgress({ bytesLoaded: cached.byteLength, bytesTotal: cached.byteLength, fromCache: true });
    return cached;
  }
  if (cached) {
    // Poisoned cache entry (HTML served before rewrite deployed). Evict.
    console.warn(`[modelCache] evicting bad cached payload for ${url} (first byte 0x${new Uint8Array(cached, 0, 1)[0].toString(16)})`);
    await _deleteCachedModel(url);
  }

  const resp = await fetch(url, { cache: 'no-store' });
  if (!resp.ok) throw new Error(`model fetch ${url} HTTP ${resp.status}`);

  // If we can't stream (no response.body reader available), fall back to .arrayBuffer().
  if (!resp.body || !resp.body.getReader || !onProgress) {
    const buf = await resp.arrayBuffer();
    if (!_looksLikeModelBytes(buf, url)) {
      throw new Error(`fetched ${url} doesn't look like a model (first byte 0x${new Uint8Array(buf, 0, 1)[0].toString(16)})`);
    }
    putCachedModel(url, buf);
    return buf;
  }

  const total = parseInt(resp.headers.get('content-length') || '0', 10);
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress({ bytesLoaded: loaded, bytesTotal: total, fromCache: false });
  }
  const merged = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { merged.set(c, off); off += c.byteLength; }
  const buf = merged.buffer;
  if (!_looksLikeModelBytes(buf, url)) {
    throw new Error(`fetched ${url} doesn't look like a model (first byte 0x${new Uint8Array(buf, 0, 1)[0].toString(16)})`);
  }
  putCachedModel(url, buf);
  return buf;
}

/**
 * Wipe everything. Useful if you ship a new model at the same URL and want
 * to force a re-fetch without changing the URL.
 */
export async function clearModelCache() {
  try {
    const db = await openDB();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).clear();
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
    });
  } catch (e) {
    // ignore
  }
}
