/**
 * Video Processing API Service
 *
 * Talks to the Next.js /api/video-score route, which proxies (with Clerk
 * auth + VLLM_GATE_TOKEN) to the Modal video-scoring app at
 * arlo--doseedo-video-scoring-…score.modal.run.
 *
 * The Modal endpoint **streams Server-Sent Events** while it works:
 *
 *   event: shots       { count }
 *   event: scene       { i, of, start, end }
 *   event: scene_done  { i, of, mood, tension }
 *   event: midi        {}
 *   event: done        { scene_data, scene_changes, duration, midi_base64 }
 *   event: error       { message }
 *
 * `uploadVideo` consumes the stream, fires an `onProgress` callback on
 * every event, and resolves with the final `done` payload (or rejects on
 * `error` / network failure).
 *
 * Idempotency: the result of a successful run is cached in `sessionStorage`
 * keyed by a fast file fingerprint (size + first-MB SHA-256). Re-dropping
 * the same clip in the same browser session reuses the cached result
 * instead of paying for another Modal run.
 */

const VIDEO_SCORE_ENDPOINT = '/api/video-score';

const CACHE_PREFIX = 'video-score:';
const CACHE_TTL_MS = 30 * 60 * 1000; // 30 min — bound stale data within a session

/**
 * Cheap content-addressed fingerprint: SHA-256 of the first 1 MB plus the
 * file size. Two distinct clips of the same length and similar opening
 * frames (e.g. identical cold-open intro) collide only if the first 1 MB
 * is bitwise-identical AND the byte length matches — extremely unlikely
 * for unrelated MP4s.
 */
async function fingerprint(file) {
  const head = await file.slice(0, 1024 * 1024).arrayBuffer();
  const digest = await crypto.subtle.digest('SHA-256', head);
  const hex = Array.from(new Uint8Array(digest))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
  return `${file.size}:${hex}`;
}

function readCache(key) {
  try {
    const raw = sessionStorage.getItem(CACHE_PREFIX + key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || (Date.now() - (parsed.t || 0)) > CACHE_TTL_MS) return null;
    return parsed.v;
  } catch (_) {
    return null;
  }
}

function writeCache(key, value) {
  try {
    sessionStorage.setItem(
      CACHE_PREFIX + key,
      JSON.stringify({ t: Date.now(), v: value }),
    );
  } catch (_) {
    // sessionStorage quota or disabled — drop silently.
  }
}

/**
 * SSE parser over a fetch stream. Yields {event, data} objects. Skips
 * lines that aren't a complete event yet.
 */
async function* readEvents(response) {
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE separator is a blank line (\n\n).
    let idx;
    while ((idx = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let event = 'message';
      const dataLines = [];
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
      }
      let data = null;
      if (dataLines.length) {
        try { data = JSON.parse(dataLines.join('\n')); }
        catch (_) { data = dataLines.join('\n'); }
      }
      yield { event, data };
    }
  }
}

/**
 * Upload video for scene detection + scoring.
 * @param {File} videoFile
 * @param {Object} [opts]
 * @param {number} [opts.bpm=120]
 * @param {string} [opts.baseProgression]   - 'Cm:0,Fm:4,G7:8,Cm:12'
 * @param {number} [opts.framesPerScene=3]
 * @param {(stage: string, payload: any) => void} [opts.onProgress]
 * @param {boolean} [opts.useCache=true]
 * @returns {Promise<{
 *   scene_data: Array<Object>,
 *   scene_changes: number[],
 *   duration: number,
 *   midi_base64: string,
 * }>}
 */
export async function uploadVideo(videoFile, opts = {}) {
  const {
    bpm = 120,
    baseProgression,
    framesPerScene = 3,
    onProgress,
    useCache = true,
  } = opts;

  // Idempotency cache lookup.
  let fp = null;
  if (useCache) {
    try {
      fp = await fingerprint(videoFile);
      const hit = readCache(`${fp}:${bpm}:${baseProgression || ''}:${framesPerScene}`);
      if (hit) {
        if (onProgress) {
          onProgress('cache_hit', {});
          onProgress('done', hit);
        }
        return hit;
      }
    } catch (_) { /* fingerprint optional */ }
  }

  const fd = new FormData();
  fd.append('file', videoFile);
  fd.append('bpm', String(bpm));
  if (baseProgression) fd.append('base_progression', baseProgression);
  fd.append('frames_per_scene', String(framesPerScene));

  const response = await fetch(VIDEO_SCORE_ENDPOINT, {
    method: 'POST',
    headers: { Accept: 'text/event-stream' },
    body: fd,
  });

  if (!response.ok) {
    let detail = '';
    try { detail = (await response.json())?.error || ''; } catch (_) {}
    throw new Error(`video-score HTTP ${response.status}${detail ? `: ${detail}` : ''}`);
  }

  let final = null;
  let serverError = null;
  for await (const { event, data } of readEvents(response)) {
    if (onProgress) onProgress(event, data);
    if (event === 'done') final = data;
    if (event === 'error') serverError = (data && data.message) || 'unknown error';
  }

  if (serverError) throw new Error(`video-score: ${serverError}`);
  if (!final || !Array.isArray(final.scene_changes) || typeof final.midi_base64 !== 'string') {
    throw new Error('video-score: malformed response');
  }

  if (useCache && fp) {
    writeCache(`${fp}:${bpm}:${baseProgression || ''}:${framesPerScene}`, final);
  }
  return final;
}

/**
 * Decode a base64 MIDI payload into a browser Blob suitable for createObjectURL.
 */
export function decodeMidiBase64(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: 'audio/midi' });
}

/**
 * Extract audio from a video file using Web Audio API + MediaRecorder.
 * Replaces the legacy GCV path that returned an audio_url from the server.
 */
export async function extractAudioFromVideo(videoFile) {
  return new Promise((resolve, reject) => {
    const v = document.createElement('video');
    v.preload = 'metadata';

    v.onloadedmetadata = () => { v.currentTime = 0; };
    v.onerror = () => reject(new Error('Failed to load video'));

    v.onloadeddata = async () => {
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const src = ctx.createMediaElementSource(v);
        const dest = ctx.createMediaStreamDestination();
        src.connect(dest);

        const rec = new MediaRecorder(dest.stream);
        const chunks = [];
        rec.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
        rec.onstop = () => resolve(new Blob(chunks, { type: 'audio/webm' }));

        rec.start();
        v.play();
        v.onended = () => { rec.stop(); ctx.close(); };
      } catch (err) {
        reject(err);
      }
    };

    v.src = URL.createObjectURL(videoFile);
  });
}

/**
 * Collapse short scene changes (merge scenes closer than threshold).
 */
export function collapseSceneChanges(sceneChanges, threshold = 3) {
  if (!sceneChanges || sceneChanges.length === 0) return [];

  const out = [];
  let start = sceneChanges[0];
  let end = start;

  for (let i = 1; i < sceneChanges.length; i++) {
    const t = sceneChanges[i];
    if (t - end < threshold) {
      end = t;
    } else {
      out.push(start);
      if (end !== start) out.push(end);
      start = end = t;
    }
  }

  if (start === end) out.push(start);
  else { out.push(start); out.push(end); }
  return out;
}

/** Compute optimal tempo for each scene to align with musical bars. */
export function computeBestTempos(sceneChanges) {
  const MIN_TEMPO = 70;
  const MAX_TEMPO = 160;
  const MAX_TEMPO_JUMP = 20;

  const tempos = [];
  for (let i = 0; i < sceneChanges.length - 1; i++) {
    const duration = sceneChanges[i + 1] - sceneChanges[i];
    let best = null;
    let bestScore = Infinity;
    for (let bpm = MIN_TEMPO; bpm <= MAX_TEMPO; bpm++) {
      const secondsPerBeat = 60 / bpm;
      const beats = duration / secondsPerBeat;
      const residual = Math.abs(Math.round(beats) - beats);
      const fullBars = beats / 4;
      const barResidual = Math.abs(Math.round(fullBars) - fullBars);
      let score = residual + (barResidual * 0.5);
      if (tempos.length > 0) {
        const jump = Math.abs(bpm - tempos[tempos.length - 1]);
        if (jump > MAX_TEMPO_JUMP) score += (jump - MAX_TEMPO_JUMP) * 0.3;
      }
      if (score < bestScore) { best = bpm; bestScore = score; }
    }
    tempos.push(best);
  }
  return tempos;
}

/** Convert scene change timestamps to per-scene durations. */
export function sceneToDurations(sceneChanges, totalDuration) {
  if (!sceneChanges || sceneChanges.length === 0) return [];
  const durations = [];
  for (let i = 0; i < sceneChanges.length - 1; i++) {
    durations.push(sceneChanges[i + 1] - sceneChanges[i]);
  }
  const lastSceneTime = sceneChanges[sceneChanges.length - 1];
  const finalSegmentDuration = totalDuration - lastSceneTime;
  if (finalSegmentDuration > 0.5) durations.push(finalSegmentDuration);
  return durations;
}

/** @deprecated server-side mux is no longer supported on this backend */
export async function exportAudioToVideo() {
  throw new Error('exportAudioToVideo is no longer supported on this backend');
}

/** @deprecated server-side mux is no longer supported on this backend */
export async function pollExportTaskStatus() {
  return { status: 'SUCCESS' };
}

/** @deprecated server-side mux is no longer supported on this backend */
export async function pollExportUntilComplete() {
  throw new Error('pollExportUntilComplete is no longer supported on this backend');
}
