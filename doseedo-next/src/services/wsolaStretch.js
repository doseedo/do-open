/*
 * wsolaStretch — pitch-preserving time-stretch (WSOLA) used by the
 * playback engine to render schedule segments whose seg.rate !== 1.
 *
 * WSOLA = Waveform Similarity Overlap-Add. Standard pitch-preserving
 * time-stretch:
 *   - Slice the input into Hann-windowed analysis frames at hop H_a.
 *   - Lay them down at synthesis hop H_s (= H_a * stretch_ratio).
 *   - Before placing each frame, search ±L samples around the analysis
 *     position for the offset whose first H_s samples best correlate
 *     with the LAST H_s samples of the previous synthesis frame. This
 *     "phase-locks" successive frames so the overlap-add stays smooth
 *     instead of producing the metallic phasing of dumb OLA.
 *
 * Pre-renders to a new AudioBuffer (synchronously — ~5ms CPU per second
 * of audio at 48 kHz) and caches the result keyed by
 *   `${audioUrl}|${srcStart.toFixed(4)}|${srcEnd.toFixed(4)}|${ratio.toFixed(4)}`
 * so re-scheduling the same segment (e.g. on play/pause cycles or live
 * meter changes) is O(1).
 *
 * For stereo, the cross-correlation uses a MID (sum-to-mono) signal so
 * both channels apply the same offset — preserves stereo image instead
 * of letting L and R drift to different lag minima.
 */

const cache = new Map();              // key → AudioBuffer
const MAX_CACHE_ENTRIES = 100;

const FRAME_SEC = 0.040;              // 40ms analysis frame (~1920 @ 48k)
const SEARCH_SEC = 0.012;             // ±12ms similarity search window
const MIN_INPUT_SAMPLES = 2048;       // skip WSOLA if input is shorter

function hannWindow(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  }
  return w;
}

/**
 * Core WSOLA loop. Returns an array of output Float32Arrays, one per
 * channel, all the same length (= round(inputLen * ratio)).
 *
 * Channels share the same per-frame offset (computed from the mid mix)
 * so stereo image is preserved.
 */
function wsolaProcess(channels, ratio) {
  const numChannels = channels.length;
  const inputLen = channels[0].length;
  // Sample-rate-relative frame/search/hop. Caller passed in arrays so we
  // recover sample rate from the AudioBuffer; for the algorithm itself,
  // FRAME etc. are already in samples (caller supplies sample-rate-aware
  // values via the caller wrapping function).
  return null; // overridden — see wsolaProcessAtSr below
}

function wsolaProcessAtSr(channels, sampleRate, ratio) {
  const numChannels = channels.length;
  const inputLen = channels[0].length;
  const outLen = Math.max(1, Math.round(inputLen * ratio));

  // Identity-or-near, or too short to stretch: linear-resample as a fallback.
  // (Our inputs for meter edits are typically several beats long, so this
  // is rare; it's here to avoid pathological behavior on micro-segments.)
  if (Math.abs(ratio - 1) < 0.001 || inputLen < MIN_INPUT_SAMPLES) {
    const outputs = [];
    for (let c = 0; c < numChannels; c++) {
      const out = new Float32Array(outLen);
      const ch = channels[c];
      for (let i = 0; i < outLen; i++) {
        const srcIdx = Math.min(inputLen - 1, Math.floor(i / ratio));
        out[i] = ch[srcIdx];
      }
      outputs.push(out);
    }
    return outputs;
  }

  const FRAME = Math.max(64, Math.round(sampleRate * FRAME_SEC));
  const HOP_S = Math.floor(FRAME / 2);                  // 50% overlap → Hann COLA
  const HOP_A = Math.max(1, Math.round(HOP_S / ratio)); // analysis hop scaled by inverse ratio
  const SEARCH = Math.max(0, Math.round(sampleRate * SEARCH_SEC));
  const window = hannWindow(FRAME);

  // Mid mix for cross-correlation matching (same offset applied to all
  // channels so stereo image stays coherent).
  const mid = new Float32Array(inputLen);
  if (numChannels === 1) {
    mid.set(channels[0]);
  } else {
    const inv = 1 / numChannels;
    for (let i = 0; i < inputLen; i++) {
      let s = 0;
      for (let c = 0; c < numChannels; c++) s += channels[c][i];
      mid[i] = s * inv;
    }
  }

  const outputs = [];
  for (let c = 0; c < numChannels; c++) outputs.push(new Float32Array(outLen));
  const lastSynthMid = new Float32Array(FRAME);     // previous synthesis frame's mid (windowed)

  let inputPos = 0;
  let outputPos = 0;
  let isFirst = true;

  while (outputPos + FRAME <= outLen) {
    // Find best offset in ±SEARCH that maximizes cross-correlation with
    // the last HOP_S samples of the previous synthesis frame.
    let bestOffset = 0;
    if (!isFirst && SEARCH > 0) {
      let bestScore = -Infinity;
      const tailStart = FRAME - HOP_S;
      const lagMin = Math.max(-SEARCH, -inputPos);
      const lagMax = Math.min(SEARCH, inputLen - inputPos - HOP_S - 1);
      for (let lag = lagMin; lag <= lagMax; lag++) {
        const candStart = inputPos + lag;
        let score = 0;
        for (let i = 0; i < HOP_S; i++) {
          score += mid[candStart + i] * lastSynthMid[tailStart + i];
        }
        if (score > bestScore) { bestScore = score; bestOffset = lag; }
      }
    }
    const grabPos = inputPos + bestOffset;

    // Bail out if we've consumed all the input we can window.
    if (grabPos < 0 || grabPos + FRAME > inputLen) break;

    // Windowed overlap-add per channel; track windowed mid for the next
    // similarity search.
    for (let i = 0; i < FRAME; i++) {
      const w = window[i];
      let midSum = 0;
      for (let c = 0; c < numChannels; c++) {
        const sample = channels[c][grabPos + i] * w;
        outputs[c][outputPos + i] += sample;
        midSum += channels[c][grabPos + i];
      }
      lastSynthMid[i] = (numChannels === 1 ? midSum : midSum / numChannels) * w;
    }

    inputPos += HOP_A;
    outputPos += HOP_S;
    isFirst = false;
  }

  return outputs;
}

/**
 * Stretch a slice of an AudioBuffer to a new duration via WSOLA.
 *
 *   srcStart, srcEnd  — slice bounds in seconds (within audioBuffer)
 *   ratio             — output_length / input_length:
 *                         ratio > 1  → stretched (slower, longer)
 *                         ratio < 1  → compressed (faster, shorter)
 *                         ratio = 1  → identity passthrough
 *
 * For seg.rate (= src_len / dst_len), pass `ratio = 1 / seg.rate`.
 *
 * Returns a freshly created AudioBuffer at the same sample rate and
 * channel count as the input. Synchronous (no Promise) — pre-render
 * cost is small enough that we don't need a worker.
 */
export function stretchBuffer(audioContext, audioBuffer, srcStart, srcEnd, ratio) {
  const sr = audioBuffer.sampleRate;
  const startSample = Math.max(0, Math.floor(srcStart * sr));
  const endSample = Math.min(audioBuffer.length, Math.floor(srcEnd * sr));
  if (endSample <= startSample) {
    return audioContext.createBuffer(audioBuffer.numberOfChannels, 1, sr);
  }

  const numChannels = audioBuffer.numberOfChannels;
  const slices = [];
  for (let c = 0; c < numChannels; c++) {
    slices.push(audioBuffer.getChannelData(c).slice(startSample, endSample));
  }

  const stretched = wsolaProcessAtSr(slices, sr, ratio);
  const outLen = Math.max(1, ...stretched.map((s) => s.length));
  const out = audioContext.createBuffer(numChannels, outLen, sr);
  for (let c = 0; c < numChannels; c++) {
    out.copyToChannel(stretched[c], c);
  }
  return out;
}

/**
 * Cached wrapper. Same args as stretchBuffer plus an `audioUrl` used as
 * a cache discriminator. Repeated calls with the same (url, srcStart,
 * srcEnd, ratio) return the same AudioBuffer instance.
 *
 * Cache evicts oldest on overflow (FIFO) — simple and bounded.
 */
export function getStretchedBuffer(audioContext, audioBuffer, audioUrl, srcStart, srcEnd, ratio) {
  const key = `${audioUrl}|${srcStart.toFixed(4)}|${srcEnd.toFixed(4)}|${ratio.toFixed(4)}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const buf = stretchBuffer(audioContext, audioBuffer, srcStart, srcEnd, ratio);
  if (cache.size >= MAX_CACHE_ENTRIES) {
    const firstKey = cache.keys().next().value;
    cache.delete(firstKey);
  }
  cache.set(key, buf);
  return buf;
}

/** For tests / debugging: clear the stretch cache. */
export function clearStretchCache() {
  cache.clear();
}
