/**
 * audio.ts — pure audio helpers used by the pipeline, worker, and UI.
 *
 * Sample-rate conversion is a polyphase Kaiser-window resampler. No runtime
 * dependencies on Web Audio: the functions here accept `Float32Array` or our
 * internal `AudioBuffer` type, so they're usable equally in the main thread and
 * a Web Worker (where `window.AudioContext` is absent).
 */

import { AudioBuffer, PUBLIC_SR } from "../pipeline/types";

/**
 * Downmix to mono. Planar stereo input becomes the average of L and R.
 * Mono input is returned unchanged (a copy is made so callers can safely
 * transfer the returned buffer).
 */
export function audioBufferToMono(buf: AudioBuffer): Float32Array {
  if (buf.channels === 1) {
    return new Float32Array(buf.samples);
  }
  const { samples, frames } = buf;
  const out = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    out[i] = 0.5 * (samples[i] + samples[frames + i]);
  }
  return out;
}

/**
 * Planar stereo → interleaved LRLR. Some encoders/consumers (e.g. WAV writer)
 * want interleaved samples even though the pipeline uses planar internally.
 */
export function interleave(buf: AudioBuffer): Float32Array {
  if (buf.channels === 1) return new Float32Array(buf.samples);
  const { samples, frames } = buf;
  const out = new Float32Array(frames * 2);
  for (let i = 0; i < frames; i++) {
    out[2 * i] = samples[i];
    out[2 * i + 1] = samples[frames + i];
  }
  return out;
}

/**
 * Fast GCD for rational resampling ratios. Polyphase resampler uses this to
 * reduce `fromSr:toSr` to lowest terms so the filter bank stays compact.
 */
function gcd(a: number, b: number): number {
  a = Math.abs(a) | 0;
  b = Math.abs(b) | 0;
  while (b) {
    const t = b;
    b = a % b;
    a = t;
  }
  return a || 1;
}

/**
 * Modified Bessel function of the first kind, order 0. Needed for the Kaiser
 * window used by the resampler. Stock JS has no I0, so we compute it via the
 * standard infinite-series expansion truncated when the next term drops below
 * a micro-fraction of the running total.
 */
function besselI0(x: number): number {
  let sum = 1;
  let term = 1;
  const halfX = x / 2;
  for (let k = 1; k < 64; k++) {
    term *= (halfX / k) * (halfX / k);
    sum += term;
    if (term < sum * 1e-12) break;
  }
  return sum;
}

/**
 * Build a windowed-sinc prototype FIR at `cutoff` (relative to 1.0 = Nyquist
 * of the oversampled rate) with `halfTaps` taps on each side of center, gained
 * by `upFactor` so the polyphase decomposition preserves signal amplitude.
 *
 * The Kaiser beta of 8.6 gives ~80 dB sidelobe rejection, comfortable for the
 * 44.1↔48 kHz and 48→22.05 kHz conversions we do.
 */
function buildKaiserSinc(halfTaps: number, cutoff: number, upFactor: number): Float32Array {
  const taps = halfTaps * 2 + 1;
  const out = new Float32Array(taps);
  const beta = 8.6;
  const denom = besselI0(beta);
  for (let i = 0; i < taps; i++) {
    const n = i - halfTaps;
    const sinc =
      n === 0 ? 1 : Math.sin(Math.PI * cutoff * n) / (Math.PI * cutoff * n);
    // Kaiser window in [-halfTaps, halfTaps]
    const r = n / halfTaps;
    const arg = beta * Math.sqrt(Math.max(0, 1 - r * r));
    const w = besselI0(arg) / denom;
    out[i] = cutoff * sinc * w * upFactor;
  }
  return out;
}

/**
 * Polyphase Kaiser resampler. Single-channel. Callers are expected to run
 * per-channel for stereo. Same-rate input short-circuits to a copy.
 *
 * Implementation notes:
 *   - We reduce L = fromSr/gcd and M = toSr/gcd; conceptually upsample by L,
 *     lowpass to min(1/L, 1/M), decimate by M.
 *   - We never materialise the upsampled signal; polyphase means we index the
 *     prototype filter by phase = (outputIdx * M) % L.
 *   - Filter length scales with the larger of L, M to keep sidelobes down at
 *     large ratios.
 */
export function audioResample(
  input: Float32Array,
  fromSr: number,
  toSr: number,
): Float32Array {
  if (fromSr === toSr) return new Float32Array(input);
  if (input.length === 0) return new Float32Array(0);

  const g = gcd(Math.round(fromSr), Math.round(toSr));
  const L = Math.round(toSr) / g; // upsample factor
  const M = Math.round(fromSr) / g; // decimation factor

  // Cutoff is the smaller of the input/output Nyquists in the oversampled
  // domain (where rate is fromSr * L). Leave a tiny guard of 0.98.
  const cutoff = Math.min(1 / L, 1 / M) * 0.98;

  // Tap count: ~32 zero-crossings on each side of the sinc, scaled by max(L,M)
  // so aggressive down-sampling still gets enough support.
  const halfTaps = 32 * Math.max(L, M);
  const proto = buildKaiserSinc(halfTaps, cutoff, L);

  // Split the prototype into L polyphase sub-filters. Sub-filter `phase` holds
  // taps (phase, phase+L, phase+2L, ...). Sub-filter length ceil((2H+1)/L).
  const subLen = Math.ceil(proto.length / L);
  const subFilters: Float32Array[] = new Array(L);
  for (let phase = 0; phase < L; phase++) {
    const sf = new Float32Array(subLen);
    for (let k = 0; k < subLen; k++) {
      const idx = phase + k * L;
      sf[k] = idx < proto.length ? proto[idx] : 0;
    }
    subFilters[phase] = sf;
  }

  const outLen = Math.floor((input.length * toSr) / fromSr);
  const out = new Float32Array(outLen);

  // Map output index n_out back to an (input_index_base, phase) pair:
  //   t = n_out * M         (in oversampled samples)
  //   input_base = floor(t / L)
  //   phase = t - input_base * L
  // Then accumulate sum_k sf[phase][k] * x[input_base - k]
  for (let n = 0; n < outLen; n++) {
    const t = n * M;
    const baseF = Math.floor(t / L);
    const phase = t - baseF * L;
    const sf = subFilters[phase];
    // Centre the filter: shift so impulse response peak aligns with the
    // current sample. halfTaps / L gives the corresponding polyphase offset.
    const center = Math.floor(halfTaps / L);
    let acc = 0;
    for (let k = 0; k < subLen; k++) {
      const inIdx = baseF + center - k;
      if (inIdx < 0 || inIdx >= input.length) continue;
      acc += sf[k] * input[inIdx];
    }
    out[n] = acc;
  }

  return out;
}

/**
 * Convert a web `AudioBuffer` (as returned by `AudioContext.decodeAudioData`)
 * into our planar `AudioBuffer` at the canonical pipeline rate (48 kHz).
 * Handles any channel count by downmixing to stereo (planar L, R).
 *
 * Normalises to peak 0.99 only when the input would clip (peak > 1.0) so we
 * don't silently crush headroom on well-mastered files.
 */
export async function decodeAudioFile(file: File): Promise<AudioBuffer> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const Ctx = (window as any).OfflineAudioContext || (window as any).webkitOfflineAudioContext;
  const decodeCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
  let decoded: globalThis.AudioBuffer;
  try {
    const arrayBuf = await file.arrayBuffer();
    decoded = await decodeCtx.decodeAudioData(arrayBuf.slice(0));
  } finally {
    void decodeCtx.close();
  }

  // Downmix to at most 2 channels, planar.
  const channels: 1 | 2 = decoded.numberOfChannels >= 2 ? 2 : 1;
  const inSr = decoded.sampleRate;
  const outSr = PUBLIC_SR;

  // Grab source channels first (up to 2).
  const srcL = decoded.getChannelData(0);
  const srcR = channels === 2 ? decoded.getChannelData(1) : srcL;

  // Resample per channel.
  const outL = audioResample(srcL, inSr, outSr);
  const outR = channels === 2 ? audioResample(srcR, inSr, outSr) : outL;

  // Peak-normalise only if clipping.
  let peak = 0;
  for (let i = 0; i < outL.length; i++) {
    const a = Math.abs(outL[i]);
    if (a > peak) peak = a;
  }
  if (channels === 2) {
    for (let i = 0; i < outR.length; i++) {
      const a = Math.abs(outR[i]);
      if (a > peak) peak = a;
    }
  }

  let gain = 1;
  if (peak > 1.0) gain = 0.99 / peak;
  if (gain !== 1) {
    for (let i = 0; i < outL.length; i++) outL[i] *= gain;
    if (channels === 2) for (let i = 0; i < outR.length; i++) outR[i] *= gain;
  }

  // Assemble planar stereo.
  const frames = outL.length;
  const samples = new Float32Array(frames * channels);
  samples.set(outL, 0);
  if (channels === 2) samples.set(outR, frames);

  // eslint reference to keep the unused-offline-context import from tripping:
  // OfflineAudioContext was probed only to check for presence on legacy Safari.
  void Ctx;

  return { samples, channels, sampleRate: outSr, frames };
}

/**
 * Apply a linear gain to audio samples in-place. Used by the render path when
 * recombining edited and unedited material.
 */
export function scaleInPlace(samples: Float32Array, gain: number): void {
  if (gain === 1) return;
  for (let i = 0; i < samples.length; i++) samples[i] *= gain;
}

/**
 * Add `src` into `dst` in-place. `dst` must be ≥ `src.length` samples.
 * Optional `gain` scales `src` during accumulation.
 */
export function addInPlace(dst: Float32Array, src: Float32Array, gain = 1): void {
  const n = Math.min(dst.length, src.length);
  if (gain === 1) {
    for (let i = 0; i < n; i++) dst[i] += src[i];
  } else {
    for (let i = 0; i < n; i++) dst[i] += gain * src[i];
  }
}

/**
 * Subtract `src` from `dst` in-place (dst -= src). Used during render to
 * cancel an unedited note from the mix before re-adding the edited version.
 */
export function subInPlace(dst: Float32Array, src: Float32Array): void {
  const n = Math.min(dst.length, src.length);
  for (let i = 0; i < n; i++) dst[i] -= src[i];
}

/**
 * Convert gainDb (decibels) to linear amplitude. `-Infinity` → 0.
 */
export function dbToGain(db: number): number {
  if (!Number.isFinite(db)) return 0;
  return Math.pow(10, db / 20);
}

/**
 * Build an empty planar stereo AudioBuffer of the given length at PUBLIC_SR.
 */
export function silence(frames: number, channels: 1 | 2 = 2): AudioBuffer {
  return {
    samples: new Float32Array(frames * channels),
    channels,
    sampleRate: PUBLIC_SR,
    frames,
  };
}
