/**
 * NoteExtractor — per-note extraction math (mask → complex STFT).
 *
 * Split out of `Pipeline` so unit tests can exercise the math without standing
 * up a Web Worker, GPU device, and ONNX runtime.
 *
 * The class is intentionally small: the `applyMaskToComplex` loop that
 * multiplies a soft mask into a complex STFT, and the `noteToQuery` adapter
 * that shapes a `Note` into whatever `MaskUNet.predict` expects. The GPU-side
 * mask→time-domain wiring lives in `Pipeline.ts` where it has access to the
 * kernels.
 *
 * Axis conventions (from `types.ts`):
 *   - complex STFT: Float32Array of shape [frames, bins, 2] (re, im), length
 *     frames * bins * 2.
 *   - magnitude STFT: Float32Array of shape [frames, bins], length frames * bins.
 *   - mask:       Float32Array of shape [frames, bins], same layout as mag.
 */

import type { NoteQuery } from "../models/MaskUNet";
import { STFT_HOP, PUBLIC_SR, type Note } from "./types";

export interface NoteToQueryOpts {
  sampleRate: number;
  hop: number;
  nFrames: number;
  /** Frame rate of `note.pitchTrack` in Hz; Basic Pitch defaults to 100. */
  pitchTrackHz?: number;
}

export class NoteExtractor {
  /**
   * Convert a `Note` → `NoteQuery` suitable for `MaskUNet.predict()`.
   *
   * Agent B's NoteQuery shape requires per-frame pitch values aligned to the
   * STFT grid; we resample `note.pitchTrack` (if present) from its native 100
   * Hz grid onto the STFT hop. When `pitchTrack` is absent we synthesise a
   * constant track at `pitchMidi + pitchCents/100`.
   */
  static noteToQuery(note: Note, opts: NoteToQueryOpts): NoteQuery {
    const { sampleRate, hop, nFrames, pitchTrackHz = 100 } = opts;
    const startFrame = Math.max(0, Math.min(nFrames, Math.round((note.startSec * sampleRate) / hop)));
    const endFrame = Math.max(startFrame, Math.min(nFrames, Math.round((note.endSec * sampleRate) / hop)));
    const spanFrames = Math.max(1, endFrame - startFrame);

    const baselineMidi = note.pitchMidi + (note.pitchCents || 0) / 100;
    let pitchTrack: Float32Array;

    if (note.pitchTrack && note.pitchTrack.length > 0) {
      pitchTrack = resamplePitchTrackToStftGrid(
        note.pitchTrack,
        pitchTrackHz,
        spanFrames,
        (sampleRate / hop),
      );
    } else {
      pitchTrack = new Float32Array(spanFrames);
      pitchTrack.fill(baselineMidi);
    }

    return {
      pitchTrack,
      instrumentClass: 0, // unknown; downstream may map `note.instrumentClass` string → id
      velocity: Math.max(0, Math.min(1, note.velocity)),
      startFrame,
      endFrame,
    };
  }

  /**
   * Apply a soft mask (frames × bins) to a complex STFT. Returns a new
   * Float32Array with the masked complex STFT, leaving the input untouched.
   *
   * The mask may be shorter than the full spectrum in the time axis (covering
   * only the note's active range); callers pass the `frameOffset` and `frames`
   * length of the mask so the caller owns the clipping policy.
   */
  static applyMaskToComplex(
    stftComplex: Float32Array,
    mask: Float32Array,
    totalFrames: number,
    bins: number,
    maskFrameOffset = 0,
    maskFrames = totalFrames,
  ): Float32Array {
    const out = new Float32Array(stftComplex.length);
    for (let t = 0; t < totalFrames; t++) {
      const rowC = t * bins * 2;
      const rel = t - maskFrameOffset;
      const inMask = rel >= 0 && rel < maskFrames;
      if (!inMask) {
        // Outside the mask window → bins are zeroed.
        continue;
      }
      const rowM = rel * bins;
      for (let b = 0; b < bins; b++) {
        const m = mask[rowM + b];
        out[rowC + b * 2] = stftComplex[rowC + b * 2] * m;
        out[rowC + b * 2 + 1] = stftComplex[rowC + b * 2 + 1] * m;
      }
    }
    return out;
  }
}

/**
 * Resample a pitch track from its native sampling rate onto the STFT grid
 * using linear interpolation. Length is fixed to `outLen`.
 */
function resamplePitchTrackToStftGrid(
  src: Float32Array,
  srcHz: number,
  outLen: number,
  dstHz: number,
): Float32Array {
  const out = new Float32Array(outLen);
  if (src.length === 0 || outLen === 0) return out;
  const ratio = srcHz / dstHz;
  for (let i = 0; i < outLen; i++) {
    const sPos = i * ratio;
    const i0 = Math.floor(sPos);
    const i1 = Math.min(src.length - 1, i0 + 1);
    const frac = sPos - i0;
    out[i] = src[Math.min(src.length - 1, i0)] * (1 - frac) + src[i1] * frac;
  }
  return out;
}

/**
 * Convenience re-exports for tests that used the V1 (pre-alignment) signature.
 * Prefer the explicit-opts `noteToQuery(note, opts)` in new code.
 */
export function noteToQueryDefault(note: Note, nFrames: number): NoteQuery {
  return NoteExtractor.noteToQuery(note, {
    sampleRate: PUBLIC_SR,
    hop: STFT_HOP,
    nFrames,
  });
}
