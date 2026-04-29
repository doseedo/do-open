/**
 * Hugging Face API Service — client wrapper around /api/hf.
 *
 * The token used to live on the client (NEXT_PUBLIC_HF_API_TOKEN) which
 * leaked it into every browser bundle. It now lives server-side as
 * HF_API_TOKEN and the proxy at app/api/hf/route.ts forwards SDK calls.
 * Keep this module's public surface stable so existing callers don't
 * have to change.
 */

const HF_PROXY = '/api/hf';

const DEFAULT_MODELS = {
  textToMusic: process.env.NEXT_PUBLIC_HF_TEXT_TO_MUSIC_MODEL || 'facebook/musicgen-small',
  melodyToMusic: process.env.NEXT_PUBLIC_HF_AUDIO_MODEL || 'facebook/musicgen-melody',
  textToSpeech: process.env.NEXT_PUBLIC_HF_VOICE_MODEL || 'suno/bark',
};

let _configured = null;

/**
 * Check whether the server has HF_API_TOKEN configured. Cached after
 * first call. Synchronous fallback returns true so existing UI gates
 * (which were boolean-typed) don't all need to become async; we still
 * await this on first probe in places that already do.
 * @returns {Promise<boolean>}
 */
export async function isHFConfigured() {
  if (_configured !== null) return _configured;
  try {
    const r = await fetch(HF_PROXY, { method: 'GET' });
    if (!r.ok) {
      _configured = false;
      return false;
    }
    const data = await r.json();
    _configured = !!data.configured;
    return _configured;
  } catch {
    _configured = false;
    return false;
  }
}

async function postProxy(payload) {
  const r = await fetch(HF_PROXY, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return r;
}

async function postProxyJson(payload) {
  const r = await postProxy(payload);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
    throw new Error(err.error || `HF proxy error: ${r.status}`);
  }
  return r.json();
}

async function postProxyBlob(payload) {
  const r = await postProxy(payload);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
    throw new Error(err.error || `HF proxy error: ${r.status}`);
  }
  return r.blob();
}

/**
 * Query a Hugging Face model.
 * @param {string} modelId
 * @param {Object} data
 * @param {Object} options
 * @returns {Promise<any>}
 */
export async function queryModel(modelId, data, options = {}) {
  if (options.task === 'text-generation') {
    return postProxyJson({
      kind: 'textGeneration',
      model: modelId,
      inputs: data.inputs,
      parameters: data.parameters,
    });
  }
  return postProxyJson({
    kind: 'request',
    model: modelId,
    inputs: data.inputs,
    parameters: data.parameters,
  });
}

/**
 * Generate music from text prompt.
 * @param {string} prompt
 * @param {Object} options
 * @returns {Promise<Blob>}
 */
export async function generateMusicFromText(prompt, options = {}) {
  const modelId = options.model || DEFAULT_MODELS.textToMusic;
  return postProxyBlob({
    kind: 'textToAudio',
    model: modelId,
    inputs: prompt,
    parameters: options.parameters || {},
  });
}

/**
 * Generate music from melody (audio conditioning).
 * @param {File|Blob} audioFile
 * @param {string} prompt
 * @param {Object} options
 * @returns {Promise<Blob>}
 */
export async function generateMusicFromMelody(audioFile, prompt = '', options = {}) {
  // The HF SDK's `request` accepts a Blob in `inputs`, but our JSON proxy
  // can't carry a Blob inline. Send the file as a base64 data URL string;
  // the proxy currently passes inputs straight to the SDK, which accepts
  // base64-encoded audio for these endpoints. If a future caller needs
  // raw streaming uploads, swap this to multipart at the proxy layer.
  const modelId = options.model || DEFAULT_MODELS.melodyToMusic;
  const buf = await audioFile.arrayBuffer();
  const b64 = typeof Buffer !== 'undefined'
    ? Buffer.from(buf).toString('base64')
    : btoa(String.fromCharCode(...new Uint8Array(buf)));
  return postProxyBlob({
    kind: 'request',
    model: modelId,
    inputs: b64,
    parameters: { prompt, ...(options.parameters || {}) },
  });
}

/**
 * Generate speech from text.
 * @param {string} text
 * @param {Object} options
 * @returns {Promise<Blob>}
 */
export async function generateSpeech(text, options = {}) {
  const modelId = options.model || DEFAULT_MODELS.textToSpeech;
  return postProxyBlob({
    kind: 'request',
    model: modelId,
    inputs: text,
    parameters: options.parameters || {},
  });
}

/**
 * Get model info.
 * @param {string} modelId
 * @returns {Promise<Object>}
 */
export async function getModelInfo(modelId) {
  return postProxyJson({ kind: 'modelInfo', model: modelId });
}

/**
 * Check if a model is loaded and ready.
 * @param {string} modelId
 * @returns {Promise<boolean>}
 */
export async function isModelReady(modelId) {
  try {
    const out = await postProxyJson({ kind: 'isModelReady', model: modelId });
    return !!out.ready;
  } catch {
    return false;
  }
}

/**
 * Wait for model to load (cold-start helper).
 * @param {string} modelId
 * @param {number} maxWaitTime - seconds
 * @returns {Promise<boolean>}
 */
export async function waitForModel(modelId, maxWaitTime = 120) {
  const startTime = Date.now();
  const maxWaitMs = maxWaitTime * 1000;
  while (Date.now() - startTime < maxWaitMs) {
    if (await isModelReady(modelId)) return true;
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  return false;
}

export { DEFAULT_MODELS };

export default {
  isHFConfigured,
  queryModel,
  generateMusicFromText,
  generateMusicFromMelody,
  generateSpeech,
  getModelInfo,
  isModelReady,
  waitForModel,
  DEFAULT_MODELS,
};
