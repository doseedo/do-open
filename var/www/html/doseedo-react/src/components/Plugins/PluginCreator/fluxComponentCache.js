/**
 * Flux Component Cache — IndexedDB-backed cache for AI-generated component images.
 *
 * Handles:
 * - Caching Flux-generated component images (knobs, slider parts, buttons)
 * - Smart reuse: after 100+ unique images, increasingly reuses cached variants
 * - Prompt normalization for consistent cache keys
 * - Async orchestration: cache check → API call → cache store → SVG build
 */

import { generateComponentImage } from '../../../services/chatAPI';
import { buildFluxSliderSVG, buildFluxButtonSVG } from './svgComponentLibrary';

const DB_NAME = 'flux-component-cache';
const STORE_NAME = 'images';
const DB_VERSION = 1;

// ── IndexedDB helpers ─────────────────────────────────────────────────────

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'key' });
        store.createIndex('componentType', 'componentType', { unique: false });
        store.createIndex('createdAt', 'createdAt', { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function dbGet(key) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const req = tx.objectStore(STORE_NAME).get(key);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}

async function dbPut(record) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.objectStore(STORE_NAME).put(record);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function dbGetAll() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const req = tx.objectStore(STORE_NAME).getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
}

async function dbGetByType(componentType) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const idx = tx.objectStore(STORE_NAME).index('componentType');
    const req = idx.getAll(componentType);
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
}

// ── Prompt normalization ──────────────────────────────────────────────────

const NOISE_WORDS = new Set([
  'a', 'an', 'the', 'with', 'and', 'or', 'for', 'of', 'in', 'on', 'at', 'to',
  'that', 'this', 'is', 'are', 'was', 'be', 'it', 'its', 'very', 'really',
  'audio', 'synthesizer', 'synth', 'plugin', 'control', 'style', 'looking',
]);

export function normalizePromptKey(prompt, componentType) {
  if (!prompt) return '';
  const words = prompt.toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')
    .split(/\s+/)
    .filter(w => w.length > 1 && !NOISE_WORDS.has(w))
    .sort();
  return `${componentType}:${words.join('_')}`;
}

// ── Cache statistics ──────────────────────────────────────────────────────

export async function getCacheStats() {
  try {
    const all = await dbGetAll();
    const stats = { totalCount: all.length, knobCount: 0, sliderTrackCount: 0, sliderThumbCount: 0, buttonCount: 0 };
    for (const entry of all) {
      if (entry.componentType === 'knob') stats.knobCount++;
      else if (entry.componentType === 'slider-track') stats.sliderTrackCount++;
      else if (entry.componentType === 'slider-thumb') stats.sliderThumbCount++;
      else if (entry.componentType === 'button') stats.buttonCount++;
    }
    return stats;
  } catch {
    return { totalCount: 0, knobCount: 0, sliderTrackCount: 0, sliderThumbCount: 0, buttonCount: 0 };
  }
}

// ── Reuse probability curve ───────────────────────────────────────────────

export function shouldReuse(totalCount) {
  if (totalCount < 50) return false;
  let probability = 0;
  if (totalCount < 100) probability = 0.10;
  else if (totalCount < 200) probability = 0.30;
  else if (totalCount < 500) probability = 0.50;
  else probability = 0.70;
  return Math.random() < probability;
}

// ── Find similar cached entry ─────────────────────────────────────────────

export async function findSimilar(componentType) {
  try {
    const entries = await dbGetByType(componentType);
    if (entries.length === 0) return null;
    // Pick a random entry of the same type
    const idx = Math.floor(Math.random() * entries.length);
    const entry = entries[idx];
    // Increment use count
    entry.useCount = (entry.useCount || 0) + 1;
    await dbPut(entry);
    return { dataUrl: entry.dataUrl, specularUrl: entry.specularUrl || null };
  } catch {
    return null;
  }
}

// ── Cache get/put ─────────────────────────────────────────────────────────

export async function getFromCache(promptKey) {
  try {
    const entry = await dbGet(promptKey);
    if (entry) {
      entry.useCount = (entry.useCount || 0) + 1;
      await dbPut(entry);
      return { dataUrl: entry.dataUrl, specularUrl: entry.specularUrl || null };
    }
    return null;
  } catch {
    return null;
  }
}

export async function putInCache(promptKey, dataUrl, componentType, originalPrompt, specularUrl = null) {
  try {
    const record = {
      key: promptKey,
      dataUrl,
      componentType,
      promptKey,
      originalPrompt,
      createdAt: Date.now(),
      useCount: 0,
    };
    if (specularUrl) record.specularUrl = specularUrl;
    await dbPut(record);
  } catch (err) {
    console.warn('[FluxCache] Failed to cache:', err);
  }
}

// ── Main orchestrator ─────────────────────────────────────────────────────

/**
 * Generate or retrieve a Flux component image.
 * @param {string} componentType - "knob", "slider-track", "slider-thumb", "button"
 * @param {string} styleDescription - Visual description
 * @param {Object} [options]
 * @param {number} [options.size=128] - Image size
 * @returns {Promise<string|null>} - base64 data URL or null on failure
 */
export async function generateFluxComponentImage(componentType, styleDescription, options = {}) {
  const { size = 128 } = options;
  const promptKey = normalizePromptKey(styleDescription, componentType);

  // 1. Check exact cache match
  const cached = await getFromCache(promptKey);
  if (cached) {
    console.log('[FluxCache] Cache hit:', promptKey);
    return cached;  // { dataUrl, specularUrl }
  }

  // 2. Check reuse probability
  const stats = await getCacheStats();
  if (shouldReuse(stats.totalCount)) {
    const reused = await findSimilar(componentType);
    if (reused) {
      console.log('[FluxCache] Reusing similar:', componentType, `(${stats.totalCount} cached)`);
      return reused;  // { dataUrl, specularUrl }
    }
  }

  // 3. Generate fresh via API (with retry)
  const MAX_RETRIES = 2;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      console.log(`[FluxCache] Generating ${componentType}${attempt > 0 ? ` (retry ${attempt})` : ''}:`, styleDescription);
      const result = await generateComponentImage({
        component_type: componentType,
        style_description: styleDescription,
        size,
      });
      if (result.data_url) {
        const flatRingUrl = result.flat_ring_url || null;
        const pressedDataUrl = result.pressed_data_url || null;
        const dataUrl = result.data_url;
        await putInCache(promptKey, dataUrl, componentType, styleDescription, flatRingUrl);
        return { dataUrl, flatRingUrl, pressedDataUrl };
      }
    } catch (err) {
      console.warn(`[FluxCache] Attempt ${attempt + 1}/${MAX_RETRIES + 1} failed:`, err.message);
      if (attempt < MAX_RETRIES) {
        await new Promise(r => setTimeout(r, 1000 * (attempt + 1))); // backoff
      } else {
        console.error('[FluxCache] All retries exhausted for:', componentType, styleDescription);
      }
    }
  }

  return null;
}

// ── SVG builders (orchestrate image generation + SVG construction) ─────

/**
 * Resolve a Flux component and return the final SVG string.
 * @param {Object} component - Component with type, fluxPrompt, width, height
 * @returns {Promise<string|null>} - Complete SVG string or null
 */
export async function resolveFluxComponent(component) {
  const { type, fluxPrompt, width, height } = component;
  if (!fluxPrompt) return null;

  try {
    if (type === 'knob') {
      const result = await generateFluxComponentImage('knob', fluxPrompt);
      if (result?.dataUrl) return { sprite: result.dataUrl, flatRing: result.flatRingUrl };
    } else if (type === 'slider') {
      const [trackResult, thumbResult] = await Promise.allSettled([
        generateFluxComponentImage('slider-track', fluxPrompt),
        generateFluxComponentImage('slider-thumb', fluxPrompt + ' fader cap'),
      ]);
      const trackUrl = trackResult.status === 'fulfilled' ? trackResult.value?.dataUrl : null;
      const thumbUrl = thumbResult.status === 'fulfilled' ? thumbResult.value?.dataUrl : null;
      if (trackUrl && thumbUrl) return buildFluxSliderSVG(width, height, trackUrl, thumbUrl);
      if (trackUrl || thumbUrl) return buildFluxSliderSVG(width, height, trackUrl || thumbUrl, thumbUrl || trackUrl);
    } else if (type === 'button') {
      const result = await generateFluxComponentImage('button', fluxPrompt);
      if (result?.dataUrl) return buildFluxButtonSVG(width, height, result.dataUrl);
    }
  } catch (err) {
    console.error(`[FluxCache] resolveFluxComponent failed for ${type}:`, err.message);
  }

  return null;
}
