/**
 * MaskUNet — per-note soft-mask predictor.
 *
 * Two modes:
 *
 *   1. "classical-fallback" (current default, no checkpoint yet).
 *      Given a note's pitch track, compute a time-varying soft mask over STFT
 *      bins from a sum of Hann-4 windows centred on the note's harmonic bin
 *      indices. Masks across overlapping notes are normalised so they sum to
 *      at most 1 per (frame, bin).
 *
 *   2. "learned" (V2 — plug-in point for the ONNX export of `NoteExtractor`).
 *      Real model input shape `[B, 1, 257, T]` (log-mag, mean-of-stereo), plus
 *      FiLM conditioning `[B, cond_dim, T]`. The ONNX export will bake in the
 *      config; this wrapper just pipes tensors through.
 *
 * The `hann4` function is a 4-lobe Hann window: zero outside [-2, 2], peak
 * of 1 at 0, smooth. The half-width is expressed in bins and widens the
 * window around each harmonic — wider = more tolerant of mistuning, narrower
 * = sharper note isolation. Default half_width = 1.5 bins is a reasonable
 * compromise at 48 kHz / 2048-FFT.
 *
 * This file runs inside a Web Worker.
 */

import { InferenceSession, Tensor } from "onnxruntime-web";

import { fetchWithOpfsCache } from "../utils/opfs";
import { configureOrtEnv, resolveExecutionProviders, type ExecutionProvider } from "./ortEnv";

/** One entry per note we want a mask for. */
export interface NoteQuery {
  pitchTrack: Float32Array;  // per-frame MIDI pitch (length === frames across [startFrame, endFrame])
  instrumentClass: number;   // int id; 0 = unknown
  velocity: number;          // 0..1
  startFrame: number;
  endFrame: number;
}

export interface MaskUNetOpts {
  executionProviders?: ExecutionProvider[];
  /** Number of harmonics summed in the classical fallback. */
  nHarmonics?: number;
  /** Hann4 half-width in STFT bins. Larger = more tolerance to mistuning. */
  halfWidthBins?: number;
}

export type MaskUNetKind = "learned" | "classical-fallback";

export class MaskUNet {
  readonly kind: MaskUNetKind;
  private readonly session: InferenceSession | null;
  private readonly opts: Required<Omit<MaskUNetOpts, "executionProviders">>;
  private disposed = false;

  private constructor(
    kind: MaskUNetKind,
    session: InferenceSession | null,
    opts: Required<Omit<MaskUNetOpts, "executionProviders">>,
  ) {
    this.kind = kind;
    this.session = session;
    this.opts = opts;
  }

  /**
   * Load the learned model from `modelUrl`, or construct a classical-fallback
   * instance when `modelUrl` is null. A null/"" URL means "no ONNX available,
   * use the deterministic hann4 mask".
   */
  static async load(modelUrl: string | null, opts?: MaskUNetOpts): Promise<MaskUNet> {
    const resolved = {
      nHarmonics: opts?.nHarmonics ?? 8,
      halfWidthBins: opts?.halfWidthBins ?? 1.5,
    };

    if (!modelUrl) {
      return new MaskUNet("classical-fallback", null, resolved);
    }

    configureOrtEnv();
    let bytes: ArrayBuffer;
    try {
      bytes = await fetchWithOpfsCache(modelUrl, filenameFromUrl(modelUrl, "mask_unet.onnx"));
    } catch {
      // No learned checkpoint published yet — fall back silently.
      return new MaskUNet("classical-fallback", null, resolved);
    }
    // Guard against dev servers returning an HTML 404 page (which the cache
    // would happily store as bytes; ORT would then crash with a cryptic
    // "protobuf parsing failed"). ONNX files start with the protobuf magic
    // byte 0x08; HTML starts with "<!" or "<h".
    const head = new Uint8Array(bytes.slice(0, 8));
    const looksLikeOnnx = head.length > 0 && head[0] !== 0x3c; // "<"
    if (!looksLikeOnnx) {
      return new MaskUNet("classical-fallback", null, resolved);
    }
    try {
      const providers = await resolveExecutionProviders(opts?.executionProviders);
      const session = await InferenceSession.create(bytes, {
        executionProviders: providers,
        graphOptimizationLevel: "all",
      });
      return new MaskUNet("learned", session, resolved);
    } catch {
      return new MaskUNet("classical-fallback", null, resolved);
    }
  }

  /**
   * Compute one `[frames, bins]` soft mask per note query.
   *
   * `logMagStft` is the shared log-magnitude spectrogram for the current
   * audio segment (row-major `[frames, bins]`). It's only used by the learned
   * path today; the classical fallback ignores it because the mask is purely
   * pitch-driven.
   */
  async predict(
    logMagStft: Float32Array,
    noteQueries: NoteQuery[],
    sampleRate: number,
    nFft: number,
  ): Promise<Float32Array[]> {
    if (this.disposed) throw new Error("MaskUNet: already disposed");

    const bins = Math.floor(nFft / 2) + 1;
    const frames = bins > 0 ? Math.floor(logMagStft.length / bins) : 0;

    if (this.kind === "classical-fallback") {
      return classicalMasks(
        noteQueries,
        frames,
        bins,
        sampleRate,
        nFft,
        this.opts.nHarmonics,
        this.opts.halfWidthBins,
      );
    }

    // Learned path — one pass per note. V1 loop is correct but slow; a future
    // optimisation batches all notes into the N-axis of the ONNX graph.
    if (!this.session) {
      throw new Error("MaskUNet: learned mode requested but session is null");
    }
    const outMasks: Float32Array[] = [];
    for (const q of noteQueries) {
      outMasks.push(await this.runLearnedOne(logMagStft, frames, bins, q));
    }
    return outMasks;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.session) {
      void this.session.release().catch(() => void 0);
    }
  }

  private async runLearnedOne(
    logMag: Float32Array,
    frames: number,
    bins: number,
    query: NoteQuery,
  ): Promise<Float32Array> {
    if (!this.session) throw new Error("MaskUNet: session unavailable");
    // Build the two model inputs. Shapes are documented at file top. We pass
    // the full mix spectrogram as context; conditioning is a [1, cond, T]
    // tensor derived from the note's pitch track + instrument class + velocity.
    const specTensor = new Tensor("float32", logMag, [1, 1, bins, frames]);
    const cond = buildCondTensor(query, frames);
    const input: Record<string, Tensor> = {};
    const inputNames = this.session.inputNames;
    input[inputNames[0]] = specTensor;
    if (inputNames.length > 1) input[inputNames[1]] = cond;
    const outputs = await this.session.run(input);
    const first = Object.values(outputs)[0];
    const data = first.data as Float32Array;
    // Expected output shape `[1, 1, bins, frames]` or `[1, frames, bins]` —
    // both flatten to `frames*bins`. Return raw; caller knows the layout is
    // row-major [frames, bins].
    return data.slice(0, frames * bins);
  }
}

// ---------------------------------------------------------------------------
// Classical fallback
// ---------------------------------------------------------------------------

/**
 * 4-lobe Hann window. Zero outside [-2, 2], 1 at 0.
 *
 *     hann4(x) = cos²(π x / 4)     for |x| ≤ 2
 *              = 0                 otherwise
 *
 * Equivalent to a raised cosine of total width 4 and peak 1.
 */
export function hann4(x: number): number {
  const a = Math.abs(x);
  if (a >= 2) return 0;
  const c = Math.cos((Math.PI * x) / 4);
  return c * c;
}

/**
 * Compute classical per-note soft masks via harmonic Hann4 summation.
 *
 * For each note, for each frame in the note's live window and each STFT bin,
 *   m_i[f, b] = Σ_{k=1..K} a_k · hann4((b - b_k(f)) / halfWidth)
 * where b_k(f) is the bin index of the k-th harmonic at frame f (computed from
 * the note's per-frame pitch), and a_k = 1/k is a mild harmonic roll-off.
 * Masks are then normalised so Σ_i m_i[f, b] ≤ 1.
 */
function classicalMasks(
  notes: NoteQuery[],
  frames: number,
  bins: number,
  sampleRate: number,
  nFft: number,
  nHarmonics: number,
  halfWidthBins: number,
): Float32Array[] {
  // Allocate one mask buffer per note.
  const masks: Float32Array[] = notes.map(() => new Float32Array(frames * bins));

  // Harmonic amplitudes: flat-ish roll-off, 1/k normalised so Σ a_k = 1.
  const amps = new Float32Array(nHarmonics);
  let ampSum = 0;
  for (let k = 0; k < nHarmonics; k++) {
    amps[k] = 1 / (k + 1);
    ampSum += amps[k];
  }
  if (ampSum > 0) {
    for (let k = 0; k < nHarmonics; k++) amps[k] /= ampSum;
  }

  const binHz = sampleRate / nFft;

  for (let i = 0; i < notes.length; i++) {
    const q = notes[i];
    const mask = masks[i];
    const vel = Math.max(0, Math.min(1, q.velocity));
    const trackLen = q.pitchTrack.length;
    if (trackLen === 0) continue;

    const startFrame = Math.max(0, q.startFrame);
    const endFrame = Math.min(frames - 1, q.endFrame);

    const noteSpan = Math.max(1, q.endFrame - q.startFrame);
    for (let f = startFrame; f <= endFrame; f++) {
      // Map global frame (q.startFrame..q.endFrame) onto the pitchTrack
      // sample index space (0..trackLen-1).
      const u = (f - q.startFrame) / noteSpan;          // 0..1
      const localIdx = u * Math.max(0, trackLen - 1);
      const iLo = Math.max(0, Math.min(trackLen - 1, Math.floor(localIdx)));
      const iHi = Math.min(trackLen - 1, iLo + 1);
      const frac = localIdx - iLo;
      const midi = q.pitchTrack[iLo] + (q.pitchTrack[iHi] - q.pitchTrack[iLo]) * frac;

      const f0Hz = 440 * Math.pow(2, (midi - 69) / 12);
      if (!Number.isFinite(f0Hz) || f0Hz <= 0) continue;

      const rowBase = f * bins;
      for (let k = 0; k < nHarmonics; k++) {
        const fk = f0Hz * (k + 1);
        const binK = fk / binHz;
        if (binK >= bins) break;
        // Only touch bins within ±2·halfWidth of the harmonic centre.
        const lo = Math.max(0, Math.floor(binK - 2 * halfWidthBins));
        const hi = Math.min(bins - 1, Math.ceil(binK + 2 * halfWidthBins));
        const amp = amps[k] * vel;
        for (let b = lo; b <= hi; b++) {
          const x = (b - binK) / halfWidthBins;
          const w = hann4(x);
          if (w > 0) mask[rowBase + b] += amp * w;
        }
      }
    }
  }

  // Normalise across notes so masks sum to ≤ 1 per (frame, bin).
  const totals = new Float32Array(frames * bins);
  for (const m of masks) {
    for (let i = 0; i < totals.length; i++) totals[i] += m[i];
  }
  for (let i = 0; i < totals.length; i++) {
    const t = totals[i];
    if (t > 1) {
      const inv = 1 / t;
      for (const m of masks) m[i] *= inv;
    }
  }
  return masks;
}

// ---------------------------------------------------------------------------
// Learned-path helpers
// ---------------------------------------------------------------------------

function buildCondTensor(query: NoteQuery, frames: number): Tensor {
  // Conditioning tensor shape `[1, cond_dim, T]`. Layout:
  //   [0] pitch_midi / 127 (normalised)
  //   [1] velocity (0..1)
  //   [2] note-active gate (1 inside [startFrame, endFrame], 0 outside)
  //   [3] instrument_class / 15 (normalised)
  const condDim = 4;
  const data = new Float32Array(condDim * frames);
  const trackLen = query.pitchTrack.length;
  const velGate = Math.max(0, Math.min(1, query.velocity));
  const instNorm = Math.max(0, Math.min(1, query.instrumentClass / 15));
  const startF = query.startFrame;
  const endF = query.endFrame;

  const noteSpan = Math.max(1, endF - startF);
  for (let t = 0; t < frames; t++) {
    let midi = 0;
    if (t >= startF && t <= endF && trackLen > 0) {
      const u = (t - startF) / noteSpan;
      const localIdx = u * Math.max(0, trackLen - 1);
      const iLo = Math.max(0, Math.min(trackLen - 1, Math.floor(localIdx)));
      const iHi = Math.min(trackLen - 1, iLo + 1);
      const frac = localIdx - iLo;
      midi = query.pitchTrack[iLo] + (query.pitchTrack[iHi] - query.pitchTrack[iLo]) * frac;
    }
    data[0 * frames + t] = midi / 127;
    data[1 * frames + t] = velGate;
    data[2 * frames + t] = t >= startF && t <= endF ? 1 : 0;
    data[3 * frames + t] = instNorm;
  }
  return new Tensor("float32", data, [1, condDim, frames]);
}

function filenameFromUrl(url: string, fallback: string): string {
  try {
    const u = new URL(url);
    const base = u.pathname.split("/").pop();
    return base && base.length > 0 ? base : fallback;
  } catch {
    const base = url.split("/").pop();
    return base && base.length > 0 ? base : fallback;
  }
}
