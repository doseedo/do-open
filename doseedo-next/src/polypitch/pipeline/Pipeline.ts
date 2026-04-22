/**
 * Pipeline — polypitch orchestrator, adapted for doseedo.
 *
 * Upstream (polypitch-browser) embeds Spotify Basic Pitch for note detection
 * inside `analyze(audio)`. doseedo already runs our own WebGPU ONNX basic-pitch
 * (`services/basicPitchOnnx.js`), so this build:
 *
 *   - drops the BasicPitch dependency + loader,
 *   - replaces `analyze()` with `ingest(audio, notes)` which caches an STFT for
 *     externally-provided notes,
 *   - keeps `render(edits)` unchanged — that's the whole reason polypitch
 *     exists here: phase-vocoder pitch-shift per-note (±3 st) with fall-through
 *     to DDSP (if present) for larger intervals.
 *
 * MaskUNet runs in classical-fallback mode (no ONNX) out of the box; wire a
 * real `mask_unet.onnx` URL later via `PipelineInitOpts.maskUNetUrl`.
 */

import { GPUContext } from "../worker/kernels/GPUContext";
import { STFTKernel } from "../worker/kernels/STFTKernel";
import { ISTFTKernel } from "../worker/kernels/ISTFTKernel";
import { MagSpecKernel } from "../worker/kernels/MagSpecKernel";
import { HCQTKernel } from "../worker/kernels/HCQTKernel";
import { FFTKernel } from "../worker/kernels/FFTKernel";
import { PhaseVocoderKernel } from "../worker/kernels/PhaseVocoderKernel";
import { MaskUNet } from "../models/MaskUNet";
import { NoteExtractor } from "./NoteExtractor";
import {
  AnalysisResult,
  AudioBuffer,
  HCQT_BINS_PER_OCTAVE,
  HCQT_FMIN_HZ,
  HCQT_HARMONICS,
  HCQT_N_OCTAVES,
  Note,
  NoteEdit,
  PipelineStage,
  PipelineStatus,
  PUBLIC_SR,
  STFT_HOP,
  STFT_N_FFT,
} from "./types";
import {
  addInPlace,
  audioResample,
  dbToGain,
  scaleInPlace,
  silence,
  subInPlace,
} from "../utils/audio";
import { createStorageBuffer, readbackFloat32, uploadFloat32 } from "./gpuIO";

export interface PipelineInitOpts {
  /** URL for mask_unet.onnx. Pass null/undefined to use the classical fallback. */
  maskUNetUrl?: string | null;
  maxNFrames?: number;
}

const DEFAULT_MAX_FRAMES = 64_000;

export interface PipelineDiagnostics {
  lastTimingsMs: { ingest?: number; extract?: number; render?: number };
  modelKinds: { maskUNet: "learned" | "classical-fallback" | "unavailable" };
}

type StatusCallback = (s: PipelineStatus) => void;

interface CachedStft {
  nFft: number;
  hop: number;
  nFrames: number;
  bins: number;
  complexByChannel: Float32Array[];
  magByChannel: Float32Array[];
  audio: AudioBuffer;
}

export class Pipeline {
  private readonly device: GPUDevice;

  private readonly stft: STFTKernel;
  private readonly istft: ISTFTKernel;
  private readonly mag: MagSpecKernel;
  private readonly hcqt: HCQTKernel;
  private readonly fft: FFTKernel;
  private readonly pv: PhaseVocoderKernel;
  private readonly maskUNet: MaskUNet;

  private readonly statusListeners = new Set<StatusCallback>();
  private disposed = false;

  private notes: Note[] = [];
  private cachedStft: CachedStft | null = null;
  private noteAudioCache = new Map<string, AudioBuffer>();

  private readonly diagnostics: PipelineDiagnostics;

  private constructor(args: {
    device: GPUDevice;
    stft: STFTKernel;
    istft: ISTFTKernel;
    mag: MagSpecKernel;
    hcqt: HCQTKernel;
    fft: FFTKernel;
    pv: PhaseVocoderKernel;
    maskUNet: MaskUNet;
    diagnostics: PipelineDiagnostics;
  }) {
    this.device = args.device;
    this.stft = args.stft;
    this.istft = args.istft;
    this.mag = args.mag;
    this.hcqt = args.hcqt;
    this.fft = args.fft;
    this.pv = args.pv;
    this.maskUNet = args.maskUNet;
    this.diagnostics = args.diagnostics;
  }

  static async init(opts: PipelineInitOpts = {}): Promise<Pipeline> {
    const device = await GPUContext.getDevice();
    const maxNFrames = opts.maxNFrames ?? DEFAULT_MAX_FRAMES;

    const [maskUNet, stft, istft, mag, hcqt, fft, pv] = await Promise.all([
      MaskUNet.load(opts.maskUNetUrl ?? null),
      STFTKernel.create(device, { nFft: STFT_N_FFT, hop: STFT_HOP, sampleRate: PUBLIC_SR, maxNFrames }),
      ISTFTKernel.create(device, { nFft: STFT_N_FFT, hop: STFT_HOP, sampleRate: PUBLIC_SR, maxNFrames }),
      MagSpecKernel.create(device, { nFft: STFT_N_FFT, maxNFrames }),
      HCQTKernel.create(device, {
        binsPerOctave: HCQT_BINS_PER_OCTAVE,
        nOctaves: HCQT_N_OCTAVES,
        nHarmonics: HCQT_HARMONICS.length,
        harmonicScales: HCQT_HARMONICS as unknown as readonly number[],
        fminHz: HCQT_FMIN_HZ,
        sampleRate: PUBLIC_SR,
        nFft: STFT_N_FFT,
        maxNFrames,
      }),
      FFTKernel.create(device, { nFft: STFT_N_FFT, inverse: false, maxBatchSize: maxNFrames }),
      PhaseVocoderKernel.create(device, { nFft: STFT_N_FFT, hop: STFT_HOP, maxNFrames }),
    ]);

    const diagnostics: PipelineDiagnostics = {
      lastTimingsMs: {},
      modelKinds: { maskUNet: maskUNet.kind },
    };

    const pipeline = new Pipeline({ device, stft, istft, mag, hcqt, fft, pv, maskUNet, diagnostics });
    pipeline.emit({ stage: "ready", progress: 1 });
    return pipeline;
  }

  onStatus(cb: StatusCallback): () => void {
    this.statusListeners.add(cb);
    return () => this.statusListeners.delete(cb);
  }

  private emit(status: PipelineStatus): void {
    for (const cb of this.statusListeners) cb(status);
  }

  private setStage(stage: PipelineStage, progress: number, message?: string): void {
    this.emit({ stage, progress, message });
  }

  /**
   * Ingest an already-analyzed track — `notes` must come from the upstream
   * basic-pitch pass (doseedo's `basicPitchOnnx.js` or per-stem `latentPitch`).
   * Builds the 48 kHz STFT cache that extract/render will reuse.
   */
  async ingest(audio: AudioBuffer, notes: Note[]): Promise<AnalysisResult> {
    this.assertLive();
    const t0 = nowMs();
    this.setStage("analyzing", 0.2, "Caching STFT for pitch shifts…");
    this.cachedStft = await this.buildStftCache(audio);
    this.notes = notes.slice();
    this.noteAudioCache.clear();
    this.diagnostics.lastTimingsMs.ingest = nowMs() - t0;
    this.setStage("ready", 1);
    return { notes: this.notes, durationSec: audio.frames / audio.sampleRate, sampleRate: audio.sampleRate };
  }

  async render(edits: NoteEdit[], includeUnedited: boolean): Promise<AudioBuffer> {
    this.assertLive();
    const t0 = nowMs();
    if (!this.cachedStft) throw new Error("Pipeline.render called before ingest()");

    this.setStage("synthesizing", 0.05, `Rendering ${edits.length} edit(s)…`);
    const base = this.cloneBaseMix(includeUnedited);
    if (edits.length === 0) {
      this.diagnostics.lastTimingsMs.render = nowMs() - t0;
      this.setStage("ready", 1);
      return base;
    }

    const editedNotes = this.notes.filter((n) => edits.some((e) => e.noteId === n.id));
    const needIds = editedNotes.map((n) => n.id).filter((id) => !this.noteAudioCache.has(id));
    if (needIds.length > 0) {
      const extracted = await this.extractImpl(needIds);
      for (const [id, buf] of extracted) this.noteAudioCache.set(id, buf);
    }

    let i = 0;
    for (const edit of edits) {
      i++;
      this.setStage("synthesizing", 0.1 + 0.9 * (i / edits.length), `Edit ${i}/${edits.length}…`);
      const note = this.notes.find((n) => n.id === edit.noteId);
      if (!note) continue;
      const original = this.noteAudioCache.get(note.id);
      if (!original) continue;

      if (includeUnedited) this.subtractNoteFromMix(base, original);

      const totalSemis = edit.semitones + edit.cents / 100;
      const gain = edit.muted ? 0 : dbToGain(edit.gainDb);
      if (edit.muted) continue;

      if (Math.abs(totalSemis) < 1e-4) {
        const scaled = new Float32Array(original.samples);
        scaleInPlace(scaled, gain);
        this.addNoteToMix(base, { ...original, samples: scaled });
        continue;
      }
      const ratio = Math.pow(2, totalSemis / 12);
      const shifted = await this.pitchShiftPhaseVocoder(note, ratio);
      if (shifted) {
        if (gain !== 1) scaleInPlace(shifted.samples, gain);
        this.addNoteToMix(base, shifted);
      }
    }

    this.diagnostics.lastTimingsMs.render = nowMs() - t0;
    this.setStage("ready", 1);
    return base;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.statusListeners.clear();
    this.noteAudioCache.clear();
    this.cachedStft = null;
    try {
      this.stft.dispose();
      this.istft.dispose();
      this.mag.dispose();
      this.hcqt.dispose();
      this.fft.dispose();
      this.pv.dispose();
      this.maskUNet.dispose();
    } catch {}
  }

  getDiagnostics(): PipelineDiagnostics {
    return {
      lastTimingsMs: { ...this.diagnostics.lastTimingsMs },
      modelKinds: { ...this.diagnostics.modelKinds },
    };
  }

  // ---- internals (same shape as upstream, DDSP removed) -------------------

  private assertLive(): void {
    if (this.disposed) throw new Error("Pipeline has been disposed");
  }

  private async buildStftCache(audio: AudioBuffer): Promise<CachedStft> {
    const channels = audio.channels;
    const fullBins = STFT_N_FFT;
    const rfftBins = STFT_N_FFT / 2 + 1;

    const perChannelSrc: Float32Array[] = [];
    for (let c = 0; c < channels; c++) {
      const chan = extractChannel(audio, c);
      perChannelSrc.push(
        audio.sampleRate === PUBLIC_SR ? chan : audioResample(chan, audio.sampleRate, PUBLIC_SR),
      );
    }
    const chanLen = perChannelSrc[0].length;
    const nFrames = this.stft.computeNFrames(chanLen);

    const complexByChannel: Float32Array[] = [];
    const magByChannel: Float32Array[] = [];

    for (const chan of perChannelSrc) {
      const pcmBuf = createStorageBuffer(this.device, chan.length * 4, "Pipeline.pcm");
      uploadFloat32(this.device, pcmBuf, chan);
      const complexBuf = createStorageBuffer(this.device, nFrames * fullBins * 8, "Pipeline.stft.complex");
      this.stft.run(pcmBuf, complexBuf, nFrames, chan.length);
      const magBuf = createStorageBuffer(this.device, nFrames * rfftBins * 4, "Pipeline.stft.mag");
      this.mag.run(complexBuf, magBuf, nFrames, rfftBins, 1);

      const [complexFull, magF] = await Promise.all([
        readbackFloat32(this.device, complexBuf, nFrames * fullBins * 8),
        readbackFloat32(this.device, magBuf, nFrames * rfftBins * 4),
      ]);
      complexByChannel.push(compactComplexToRfftBins(complexFull, nFrames, fullBins, rfftBins));
      magByChannel.push(magF);
      pcmBuf.destroy(); complexBuf.destroy(); magBuf.destroy();
    }

    return { nFft: STFT_N_FFT, hop: STFT_HOP, nFrames, bins: rfftBins, complexByChannel, magByChannel, audio };
  }

  private async extractImpl(noteIds: string[]): Promise<Map<string, AudioBuffer>> {
    const cache = this.cachedStft!;
    const idSet = new Set(noteIds);
    const selected = this.notes.filter((n) => idSet.has(n.id));
    if (selected.length === 0) return new Map();
    const queries = selected.map((n) =>
      NoteExtractor.noteToQuery(n, { sampleRate: PUBLIC_SR, hop: STFT_HOP, nFrames: cache.nFrames }),
    );
    const channels = cache.complexByChannel.length as 1 | 2;
    const midMag = channels === 2
      ? averageMag(cache.magByChannel[0], cache.magByChannel[1])
      : cache.magByChannel[0];
    const masks = await this.maskUNet.predict(midMag, queries, PUBLIC_SR, cache.nFft);

    const out = new Map<string, AudioBuffer>();
    for (let qi = 0; qi < queries.length; qi++) {
      const q = queries[qi];
      const note = selected[qi];
      const mask = masks[qi];
      if (!mask) continue;
      const maskFrames = Math.max(1, q.endFrame - q.startFrame);
      const perChannelTime: Float32Array[] = [];
      for (let c = 0; c < channels; c++) {
        const maskedRfft = NoteExtractor.applyMaskToComplex(
          cache.complexByChannel[c], mask, cache.nFrames, cache.bins, q.startFrame, maskFrames,
        );
        const time = await this.istftToTime(maskedRfft, cache.nFrames);
        perChannelTime.push(fitLength(time, cache.audio.frames));
      }
      const frames = cache.audio.frames;
      const samples = new Float32Array(frames * channels);
      samples.set(perChannelTime[0], 0);
      if (channels === 2) samples.set(perChannelTime[1], frames);
      out.set(note.id, { samples, channels, sampleRate: PUBLIC_SR, frames });
    }
    return out;
  }

  private async istftToTime(rfftComplex: Float32Array, nFrames: number): Promise<Float32Array> {
    const fullBins = STFT_N_FFT;
    const fullComplex = expandRfftToFullComplex(rfftComplex, nFrames, fullBins);
    const complexBuf = createStorageBuffer(this.device, nFrames * fullBins * 8, "Pipeline.istft.in");
    uploadFloat32(this.device, complexBuf, fullComplex);
    const outLen = this.istft.outputSamplesFor(nFrames);
    const pcmBuf = createStorageBuffer(this.device, outLen * 4, "Pipeline.istft.out");
    this.istft.run(complexBuf, pcmBuf, nFrames);
    const pcm = await readbackFloat32(this.device, pcmBuf, outLen * 4);
    complexBuf.destroy(); pcmBuf.destroy();
    return pcm;
  }

  private cloneBaseMix(includeUnedited: boolean): AudioBuffer {
    const a = this.cachedStft!.audio;
    if (includeUnedited) {
      return { samples: new Float32Array(a.samples), channels: a.channels, sampleRate: a.sampleRate, frames: a.frames };
    }
    return silence(a.frames, a.channels);
  }

  private subtractNoteFromMix(base: AudioBuffer, note: AudioBuffer): void {
    const ch = Math.min(base.channels, note.channels);
    const frames = Math.min(base.frames, note.frames);
    for (let c = 0; c < ch; c++) {
      const bOff = c * base.frames, nOff = c * note.frames;
      subInPlace(base.samples.subarray(bOff, bOff + frames), note.samples.subarray(nOff, nOff + frames));
    }
  }

  private addNoteToMix(base: AudioBuffer, note: AudioBuffer, gain = 1): void {
    const ch = Math.min(base.channels, note.channels);
    const frames = Math.min(base.frames, note.frames);
    for (let c = 0; c < ch; c++) {
      const bOff = c * base.frames, nOff = c * note.frames;
      addInPlace(base.samples.subarray(bOff, bOff + frames), note.samples.subarray(nOff, nOff + frames), gain);
    }
  }

  private async pitchShiftPhaseVocoder(note: Note, ratio: number): Promise<AudioBuffer> {
    // The WGSL phase_vocoder.wgsl is Laroche-Dolson phase-only WITHOUT the
    // identity phase-locking step the comment promises. On real-world non-
    // stationary audio (anything that isn't a pure sine) each bin's phase
    // gets rotated independently, the frame-to-frame coherence is destroyed,
    // and overlap-add reconstruction sums to ≈identity + modulation artefacts
    // — not an audibly shifted signal. That's the "rendered N KB WAV, sounds
    // identical" symptom: the whole subtract-extracted-note / add-shifted-
    // note cycle is approximately a no-op because shifted ≈ extracted.
    //
    // Real phase-vocoder pitch shift needs either (a) identity phase locking
    // in the shader + proper group-of-peaks phase advance, or (b) a
    // time-stretch PV with a variable synthesis hop followed by a resample.
    // Both are real work. Until either lands, do a time-domain resample of
    // the mask-extracted note: the note plays at `ratio×` pitch for
    // `1/ratio × original_duration`. Coarse (duration changes), but reliably
    // audible, which is what's missing right now.
    const cache = this.cachedStft!;
    const channels = cache.complexByChannel.length as 1 | 2;

    // Prefer the note buffer the extract path already produced for the
    // subtract step — we'd otherwise re-run mask + ISTFT for the same note.
    let original = this.noteAudioCache.get(note.id) ?? null;
    if (!original) {
      const extracted = await this.extractImpl([note.id]);
      original = extracted.get(note.id) ?? null;
      if (original) this.noteAudioCache.set(note.id, original);
    }
    if (!original) return silence(cache.audio.frames, channels);

    const totalFrames = cache.audio.frames;
    const startSample = Math.max(0, Math.floor(note.startSec * PUBLIC_SR));
    const endSample = Math.min(totalFrames, Math.ceil(note.endSec * PUBLIC_SR));
    const segLen = Math.max(0, endSample - startSample);
    if (segLen === 0) return silence(totalFrames, channels);

    // To shift pitch UP by `ratio`, generate a buffer whose playback-at-
    // PUBLIC_SR runs `ratio×` faster — i.e. `1/ratio` fewer samples per unit
    // time. audioResample(seg, PUBLIC_SR, PUBLIC_SR/ratio) yields exactly
    // that: same audio content encoded at a lower effective rate. When the
    // mixer reads it back at PUBLIC_SR, each sample advances `ratio×` faster
    // through the waveform, raising pitch by `ratio`.
    const targetSr = Math.max(1, Math.round(PUBLIC_SR / ratio));
    const outSamples = new Float32Array(totalFrames * channels);
    for (let c = 0; c < channels; c++) {
      const srcOff = c * totalFrames;
      const segment = original.samples.subarray(srcOff + startSample, srcOff + endSample);
      const shifted = audioResample(segment, PUBLIC_SR, targetSr);
      // For ratio > 1 (pitch up): shifted < segLen → place only what fits;
      // the tail of the note window becomes silence. For ratio < 1 (pitch
      // down): shifted > segLen → truncate; we clip the extended tail so the
      // edit doesn't bleed into the next note's window.
      const placeLen = Math.min(shifted.length, segLen);
      outSamples.set(shifted.subarray(0, placeLen), srcOff + startSample);
    }
    const frames = totalFrames;
    return { samples: outSamples, channels, sampleRate: PUBLIC_SR, frames };
  }
}

// ---- module-local helpers (unchanged from upstream) -----------------------

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}
function extractChannel(audio: AudioBuffer, c: number): Float32Array {
  const off = c * audio.frames;
  return audio.samples.subarray(off, off + audio.frames);
}
function averageMag(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = 0.5 * (a[i] + b[i]);
  return out;
}
function fitLength(src: Float32Array, targetLen: number): Float32Array {
  if (src.length === targetLen) return src;
  const out = new Float32Array(targetLen);
  out.set(src.subarray(0, Math.min(src.length, targetLen)), 0);
  return out;
}
function compactComplexToRfftBins(full: Float32Array, nFrames: number, nFft: number, rfftBins: number): Float32Array {
  const out = new Float32Array(nFrames * rfftBins * 2);
  for (let t = 0; t < nFrames; t++) {
    const srcOff = t * nFft * 2, dstOff = t * rfftBins * 2;
    out.set(full.subarray(srcOff, srcOff + rfftBins * 2), dstOff);
  }
  return out;
}
function expandRfftToFullComplex(rfft: Float32Array, nFrames: number, nFft: number): Float32Array {
  const rfftBins = nFft / 2 + 1;
  const out = new Float32Array(nFrames * nFft * 2);
  for (let t = 0; t < nFrames; t++) {
    const srcOff = t * rfftBins * 2, dstOff = t * nFft * 2;
    out.set(rfft.subarray(srcOff, srcOff + rfftBins * 2), dstOff);
    for (let b = 1; b < nFft / 2; b++) {
      const srcRe = rfft[srcOff + b * 2], srcIm = rfft[srcOff + b * 2 + 1];
      const mirrorBin = nFft - b;
      out[dstOff + mirrorBin * 2] = srcRe;
      out[dstOff + mirrorBin * 2 + 1] = -srcIm;
    }
  }
  return out;
}
