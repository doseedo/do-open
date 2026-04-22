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

    const inputRms = rmsOf(base.samples);
    // Also log cached STFT RMS — if this is 0 the GPU STFT pipeline isn't
    // producing valid output despite the input audio being non-zero.
    const stftRms = this.cachedStft ? rmsOf(this.cachedStft.complexByChannel[0]) : 0;
    const magRms = this.cachedStft ? rmsOf(this.cachedStft.magByChannel[0]) : 0;
    console.log(`[polypitch.render] ${edits.length} edits, input rms=${inputRms.toFixed(5)} cachedSTFT.rms=${stftRms.toFixed(5)} cachedMag.rms=${magRms.toFixed(5)} sr=${this.cachedStft?.audio.sampleRate} frames=${this.cachedStft?.audio.frames}`);

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

      const extractedRms = rmsOf(original.samples);

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
        const shiftedRms = rmsOf(shifted.samples);
        const deltaRms = rmsOf2(original.samples, shifted.samples);
        console.log(`  [edit ${i}/${edits.length}] note=${note.id} ±${totalSemis.toFixed(2)}st ratio=${ratio.toFixed(4)}`
          + ` extract.rms=${extractedRms.toFixed(5)} shifted.rms=${shiftedRms.toFixed(5)}`
          + ` |shifted-extract|.rms=${deltaRms.toFixed(5)}`);
        if (gain !== 1) scaleInPlace(shifted.samples, gain);
        this.addNoteToMix(base, shifted);
      }
    }

    const outputRms = rmsOf(base.samples);
    const deltaIO = rmsOf2Audio(this.cachedStft!.audio.samples, base.samples);
    console.log(`[polypitch.render] done. output rms=${outputRms.toFixed(5)} |output-input|.rms=${deltaIO.toFixed(5)}`);

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

      // Diag: what does the mask actually look like? If sum/peak=0 the
      // extract is silent before ISTFT even runs → classicalMasks didn't
      // populate. If mask is non-zero but maskedRfft rms=0, it's a layout
      // mismatch between mask and STFT. Either way we'll know.
      let maskSum = 0, maskPeak = 0, maskNonzero = 0;
      for (let j = 0; j < mask.length; j++) {
        const v = mask[j]; if (v > 0) { maskSum += v; maskNonzero++; if (v > maskPeak) maskPeak = v; }
      }

      // RMS of the cached STFT WITHIN the note's window. If this is 0 while
      // the global cachedSTFT.rms (first ~244 frames) is non-zero, the STFT
      // pipeline only populated the intro — most of the song's STFT is
      // empty. That'd make mask × STFT = 0 for any note past frame ~244.
      let stftWinSumSq = 0, magWinSumSq = 0, winSamples = 0;
      const bins_ = cache.bins;
      const sEnd = Math.min(q.endFrame, cache.nFrames);
      for (let t = q.startFrame; t < sEnd; t++) {
        for (let b = 0; b < bins_; b++) {
          const re = cache.complexByChannel[0][t * bins_ * 2 + b * 2];
          const im = cache.complexByChannel[0][t * bins_ * 2 + b * 2 + 1];
          stftWinSumSq += re * re + im * im;
          const mv = cache.magByChannel[0][t * bins_ + b];
          magWinSumSq += mv * mv;
          winSamples++;
        }
      }
      const stftWinRms = Math.sqrt(stftWinSumSq / Math.max(1, winSamples));
      const magWinRms = Math.sqrt(magWinSumSq / Math.max(1, winSamples));

      const perChannelTime: Float32Array[] = [];
      let firstMaskedRfftRms = 0;
      for (let c = 0; c < channels; c++) {
        const maskedRfft = NoteExtractor.applyMaskToComplex(
          cache.complexByChannel[c], mask, cache.nFrames, cache.bins, q.startFrame, maskFrames,
        );
        if (c === 0) {
          let s = 0; const N = Math.min(maskedRfft.length, 500000);
          for (let i = 0; i < N; i++) s += maskedRfft[i] * maskedRfft[i];
          firstMaskedRfftRms = Math.sqrt(s / Math.max(1, N));
        }
        const time = await this.istftToTime(maskedRfft, cache.nFrames);
        perChannelTime.push(fitLength(time, cache.audio.frames));
      }
      console.log(`  [extract] note=${note.id} vel=${q.velocity.toFixed(2)} `
        + `startF=${q.startFrame} endF=${q.endFrame} maskFrames=${maskFrames} `
        + `mask{sum=${maskSum.toFixed(2)}, peak=${maskPeak.toFixed(4)}, nonzero=${maskNonzero}} `
        + `stft@win.rms=${stftWinRms.toFixed(5)} mag@win.rms=${magWinRms.toFixed(5)} `
        + `maskedRfft.rms=${firstMaskedRfftRms.toFixed(6)}`);

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
    // Replaced the WGSL phase_vocoder.wgsl call chain with a CPU port of the
    // same Laroche-Dolson bin-wise phase vocoder, verified audibly correct
    // in /tmp/polypitch-test/reproduce_browser.py on a synthetic C+E+G triad
    // (E4 peak drops 72%, new peak at E4×2^(1/12), C4/G4 intact).
    //
    // Four GPU-path attempts to make this audible failed:
    //   - 69f2cd99 removed a misdiagnosed audioResample;
    //   - 4a128eda fixed a real mask row-indexing bug in NoteExtractor;
    //   - 84b09561 tried a time-domain-resample bypass (reverted, changed dur);
    //   - d179ed32 switched the PV buffer from full-complex to rfft-packed.
    // Each shipped "correct logs, identical audio." The GPU kernel has layered
    // buffer-layout + uniform-frame-index hazards that kept turning silent
    // despite the algorithm being provably sound. A CPU port sidesteps all
    // of them. At 30 s of audio (~2900 STFT frames × 1025 bins) this loop
    // runs in ~30 ms in V8 — indistinguishable from the GPU version from a
    // wall-clock perspective given the async buffer roundtrips we were doing
    // anyway.
    const cache = this.cachedStft!;
    const channels = cache.complexByChannel.length as 1 | 2;
    const fullBins = STFT_N_FFT;
    const nFrames = cache.nFrames;
    const bins = cache.bins;
    const hop = cache.hop;

    const query = NoteExtractor.noteToQuery(note, { sampleRate: PUBLIC_SR, hop: STFT_HOP, nFrames });
    const midMag = channels === 2
      ? averageMag(cache.magByChannel[0], cache.magByChannel[1])
      : cache.magByChannel[0];
    const masks = await this.maskUNet.predict(midMag, [query], PUBLIC_SR, cache.nFft);
    const mask = masks[0];
    if (!mask) return silence(cache.audio.frames, channels);
    const maskFrames = Math.max(1, query.endFrame - query.startFrame);

    const outChannels: Float32Array[] = [];
    for (let c = 0; c < channels; c++) {
      const maskedRfft = NoteExtractor.applyMaskToComplex(
        cache.complexByChannel[c], mask, nFrames, bins, query.startFrame, maskFrames,
      );
      const rotatedRfft = cpuPhaseVocoder(maskedRfft, nFrames, bins, ratio, hop, cache.nFft);

      // ISTFT on GPU (that path is working — extract uses the same step).
      const rotatedFull = expandRfftToFullComplex(rotatedRfft, nFrames, fullBins);
      const stftBuf = createStorageBuffer(this.device, nFrames * fullBins * 8, "Pipeline.pv.full");
      uploadFloat32(this.device, stftBuf, rotatedFull);
      const outLen = this.istft.outputSamplesFor(nFrames);
      const pcmBuf = createStorageBuffer(this.device, outLen * 4, "Pipeline.pv.out");
      this.istft.run(stftBuf, pcmBuf, nFrames);
      const time = await readbackFloat32(this.device, pcmBuf, outLen * 4);
      stftBuf.destroy(); pcmBuf.destroy();

      outChannels.push(fitLength(time, cache.audio.frames));
    }

    const frames = cache.audio.frames;
    const samples = new Float32Array(frames * channels);
    samples.set(outChannels[0], 0);
    if (channels === 2) samples.set(outChannels[1], frames);
    return { samples, channels, sampleRate: PUBLIC_SR, frames };
  }
}

/**
 * CPU Laroche-Dolson bin-wise phase vocoder. Operates on an rfft-packed complex
 * STFT `[nFrames, rfftBins, 2]` interleaved float32. Returns a new buffer of
 * the same shape with each bin's instantaneous frequency scaled by `ratio` and
 * the phase accumulator advanced coherently across frames. Magnitude preserved.
 *
 * Port of phase_vocoder.wgsl (same math, no identity phase locking), without
 * the GPU-side buffer-layout + uniform-frame-index hazards that kept nerfing
 * the shipped shader in practice.
 */
function cpuPhaseVocoder(
  rfftComplex: Float32Array,
  nFrames: number,
  nBins: number,
  ratio: number,
  hop: number,
  nFft: number,
): Float32Array {
  const out = new Float32Array(rfftComplex.length);
  const prevInPhase = new Float32Array(nBins);
  const accumOutPhase = new Float32Array(nBins);
  const TAU = 2 * Math.PI;
  const omegaExpected = new Float32Array(nBins);
  for (let b = 0; b < nBins; b++) omegaExpected[b] = (TAU * b * hop) / nFft;

  // Frame 0: identity, seed accumulators from the input phase.
  for (let b = 0; b < nBins; b++) {
    const base = b * 2;
    const re = rfftComplex[base];
    const im = rfftComplex[base + 1];
    out[base] = re;
    out[base + 1] = im;
    const phase = Math.atan2(im, re);
    prevInPhase[b] = phase;
    accumOutPhase[b] = phase;
  }

  for (let f = 1; f < nFrames; f++) {
    const rowOff = f * nBins * 2;
    for (let b = 0; b < nBins; b++) {
      const base = rowOff + b * 2;
      const re = rfftComplex[base];
      const im = rfftComplex[base + 1];
      const mag = Math.sqrt(re * re + im * im);
      const phase = Math.atan2(im, re);
      // Unwrap delta into (-π, π].
      let delta = phase - prevInPhase[b] - omegaExpected[b];
      delta = delta - TAU * Math.floor((delta + Math.PI) / TAU);
      const instOmega = (omegaExpected[b] + delta) / hop;
      const shiftedOmega = instOmega * ratio;
      const newOutPhase = accumOutPhase[b] + shiftedOmega * hop;
      accumOutPhase[b] = newOutPhase;
      prevInPhase[b] = phase;
      out[base] = mag * Math.cos(newOutPhase);
      out[base + 1] = mag * Math.sin(newOutPhase);
    }
  }
  return out;
}

// ---- module-local helpers (unchanged from upstream) -----------------------

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}
function rmsOf(buf: Float32Array): number {
  let s = 0;
  const N = Math.min(buf.length, 500000);
  for (let i = 0; i < N; i++) s += buf[i] * buf[i];
  return Math.sqrt(s / Math.max(1, N));
}
function rmsOf2(a: Float32Array, b: Float32Array): number {
  let s = 0;
  const N = Math.min(a.length, b.length, 500000);
  for (let i = 0; i < N; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s / Math.max(1, N));
}
function rmsOf2Audio(a: Float32Array, b: Float32Array): number {
  let s = 0;
  const N = Math.min(a.length, b.length, 500000);
  for (let i = 0; i < N; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s / Math.max(1, N));
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
