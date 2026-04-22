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
    console.log(`[polypitch.render] edits=${edits.length} includeUnedited=${includeUnedited} cachedNotes=${this.notes.length}`);
    if (edits.length === 0) {
      this.diagnostics.lastTimingsMs.render = nowMs() - t0;
      this.setStage("ready", 1);
      return base;
    }

    // Extract per-note time signals via spectral-portion allocation. This
    // ports polypitch-research's `resynthesize_notes`: each STFT bin's
    // magnitude gets distributed across overlapping notes by their
    // normalised harmonic demand, so Σ note.time_signal ≈ input_audio.
    // Unlike the earlier hann4-mask extract (~5 % energy capture) or the
    // broadband copy (couldn't separate overlapping notes), this yields
    // clean per-note isolation.
    const editedNotes = this.notes.filter((n) => edits.some((e) => e.noteId === n.id));
    const missing = edits.filter((e) => !this.notes.some((n) => n.id === e.noteId));
    if (missing.length > 0) {
      console.warn(`[polypitch.render] ${missing.length} edit(s) reference unknown noteIds:`, missing.map((e) => e.noteId));
    }
    const needIds = editedNotes.map((n) => n.id).filter((id) => !this.noteAudioCache.has(id));
    console.log(`[polypitch.render] edited=${editedNotes.length} extractNeeded=${needIds.length} cached=${editedNotes.length - needIds.length}`);
    if (needIds.length > 0) {
      const tExtract = nowMs();
      const extracted = await this.extractImpl(needIds);
      for (const [id, buf] of extracted) this.noteAudioCache.set(id, buf);
      console.log(`[polypitch.render] extract done in ${(nowMs() - tExtract).toFixed(0)}ms → ${extracted.size} note(s)`);
    }

    let i = 0;
    for (const edit of edits) {
      i++;
      this.setStage("synthesizing", 0.1 + 0.9 * (i / edits.length), `Edit ${i}/${edits.length}…`);
      const note = this.notes.find((n) => n.id === edit.noteId);
      if (!note) {
        console.warn(`[polypitch.render] edit ${i}: note ${edit.noteId} not in this.notes`);
        continue;
      }
      const original = this.noteAudioCache.get(note.id);
      if (!original) {
        console.warn(`[polypitch.render] edit ${i}: no extracted buffer for ${note.id}`);
        continue;
      }

      const origNoteRms = windowRms(original, note.startSec, note.endSec, PUBLIC_SR);
      if (includeUnedited) this.subtractNoteFromMix(base, original);

      const totalSemis = edit.semitones + edit.cents / 100;
      const gain = edit.muted ? 0 : dbToGain(edit.gainDb);
      if (edit.muted) {
        console.log(`[polypitch.render] edit ${i}/${edits.length} noteId=${edit.noteId} midi=${note.pitchMidi} MUTED (origRms=${origNoteRms.toExponential(2)})`);
        continue;
      }

      if (Math.abs(totalSemis) < 1e-4) {
        const scaled = new Float32Array(original.samples);
        scaleInPlace(scaled, gain);
        this.addNoteToMix(base, { ...original, samples: scaled });
        console.log(`[polypitch.render] edit ${i}/${edits.length} midi=${note.pitchMidi} semis=0 (gain-only, origRms=${origNoteRms.toExponential(2)})`);
        continue;
      }
      const ratio = Math.pow(2, totalSemis / 12);
      // Pitch-shift the isolated note's time signal by resampling its
      // audio segment. Clean shift with no phasiness artefacts because
      // the signal is already single-note-isolated. Duration shortens by
      // 1/ratio for pitch-up; the gap at the note's tail is imperceptible
      // at chord-edit granularity.
      const shifted = pitchShiftByResample(note, original, ratio, PUBLIC_SR);
      // Shifted-note RMS is measured over the post-resample active window:
      // length shrinks to (noteDur/ratio) for pitch-up, grows for pitch-down.
      const shiftedEnd = note.startSec + (note.endSec - note.startSec) / ratio;
      const shiftedNoteRms = windowRms(shifted, note.startSec, shiftedEnd, PUBLIC_SR);
      if (gain !== 1) scaleInPlace(shifted.samples, gain);
      this.addNoteToMix(base, shifted);
      console.log(`[polypitch.render] edit ${i}/${edits.length} midi=${note.pitchMidi} semis=${totalSemis.toFixed(2)} ratio=${ratio.toFixed(4)} gain=${gain.toFixed(3)} origRms=${origNoteRms.toExponential(2)} shiftedRms=${shiftedNoteRms.toExponential(2)} t=[${note.startSec.toFixed(2)},${note.endSec.toFixed(2)}]s`);
    }

    this.diagnostics.lastTimingsMs.render = nowMs() - t0;
    this.setStage("ready", 1);
    console.log(`[polypitch.render] complete in ${(nowMs() - t0).toFixed(0)}ms, baseFrames=${base.frames}`);
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
    if (selected.length === 0) {
      console.warn(`[polypitch.extract] no cached notes match ${noteIds.length} requested id(s)`);
      return new Map();
    }
    const channels = cache.complexByChannel.length as 1 | 2;
    const nFrames = cache.nFrames;
    const nBins = cache.bins;
    const nFft = cache.nFft;
    const hop = cache.hop;
    const sr = PUBLIC_SR;
    console.log(`[polypitch.extract] notes=${this.notes.length} extracting=${selected.length} nFrames=${nFrames} nBins=${nBins} channels=${channels}`);

    // Build the portion-factor input shape the Python algorithm wants:
    // per-note {startFrame, endFrame, f0Hz, energy}. Basic-pitch gives us
    // a constant midi pitch per note (no vibrato track), so f0Hz is
    // constant for the note's duration.
    const portionNotes: PortionNote[] = this.notes.map((n) => {
      const startFrame = Math.max(0, Math.round((n.startSec * sr) / hop));
      const endFrame = Math.min(
        nFrames - 1,
        Math.max(startFrame, Math.round((n.endSec * sr) / hop)),
      );
      const midi = n.pitchMidi + (n.pitchCents || 0) / 100;
      const f0Hz = 440 * Math.pow(2, (midi - 69) / 12);
      const energy = Math.max(0.01, Math.min(1, n.velocity));
      return { startFrame, endFrame, f0Hz, energy };
    });

    // Figure out which portion-notes we're extracting (indices into
    // portionNotes that correspond to the requested noteIds).
    const selectedIdxs: number[] = [];
    const selectedByIdx = new Map<number, Note>();
    this.notes.forEach((n, i) => {
      if (idSet.has(n.id)) {
        selectedIdxs.push(i);
        selectedByIdx.set(i, n);
      }
    });

    // Per-edited-note scratch STFT buffers. Sized to each note's local frame
    // window rather than full-track length — a 0.5 s note at 48 kHz/hop=512 is
    // ~48 frames vs ~2900 for a 30 s track. Drops peak memory from ~200 MB to
    // ~MB-scale for typical chord edits.
    const editedCount = selectedIdxs.length;
    type LocalBuf = { startFrame: number; localFrames: number; byChannel: Float32Array[] };
    const perEditLocal: LocalBuf[] = selectedIdxs.map((nIdx) => {
      const pn = portionNotes[nIdx];
      const localFrames = Math.max(1, pn.endFrame - pn.startFrame + 1);
      return {
        startFrame: pn.startFrame,
        localFrames,
        byChannel: Array.from({ length: channels }, () => new Float32Array(localFrames * nBins * 2)),
      };
    });
    const demand = new Float32Array(portionNotes.length * nBins);  // per-frame, reused
    const colSum = new Float32Array(nBins);

    // Sweep frames: compute demand for all notes active at t, normalise
    // by col-sum per bin, and for each *edited* note write its share of
    // the complex STFT into that edit's scratch buffer.
    const HARMONICS = 20;
    const halfRatio = Math.pow(2, 2 / 12 / 2) - 1;  // "2-semitone full width" in bin terms
    const binPerHz = nFft / sr;

    // Map: portionNote idx → edit slot (or -1 if not edited)
    const idxToEdit = new Int32Array(portionNotes.length).fill(-1);
    selectedIdxs.forEach((nIdx, editSlot) => { idxToEdit[nIdx] = editSlot; });

    for (let t = 0; t < nFrames; t++) {
      // Figure out which notes are active at frame t. Skip the whole
      // frame if no edit-candidate is among them.
      let anyEdited = false;
      demand.fill(0);
      for (let nIdx = 0; nIdx < portionNotes.length; nIdx++) {
        const pn = portionNotes[nIdx];
        if (t < pn.startFrame || t > pn.endFrame) continue;
        if (idxToEdit[nIdx] >= 0) anyEdited = true;
        // Accumulate harmonic demand into demand[nIdx, :]
        const noteOff = nIdx * nBins;
        for (let k = 1; k <= HARMONICS; k++) {
          const fk = k * pn.f0Hz;
          const binK = fk * binPerHz;
          if (binK >= nBins) break;
          let halfWidthBins = binK * halfRatio;
          if (halfWidthBins < 1) halfWidthBins = 1;
          const amp = (1 / k) * pn.energy;
          const lo = Math.max(0, Math.floor(binK - halfWidthBins));
          const hi = Math.min(nBins - 1, Math.ceil(binK + halfWidthBins));
          for (let b = lo; b <= hi; b++) {
            const u = (b - binK) / halfWidthBins;
            if (u <= -1 || u >= 1) continue;
            // hann⁴: (0.5 + 0.5 cos(πu))⁴
            const hann = 0.5 + 0.5 * Math.cos(Math.PI * u);
            const w = hann * hann * hann * hann;
            demand[noteOff + b] += amp * w;
          }
        }
      }
      if (!anyEdited) continue;

      // col_sum across all notes for normalisation
      colSum.fill(0);
      for (let nIdx = 0; nIdx < portionNotes.length; nIdx++) {
        const noteOff = nIdx * nBins;
        for (let b = 0; b < nBins; b++) colSum[b] += demand[noteOff + b];
      }

      // For each edited note active at this frame, write its share of the
      // STFT into its (local) buffer.
      for (let editSlot = 0; editSlot < editedCount; editSlot++) {
        const nIdx = selectedIdxs[editSlot];
        const pn = portionNotes[nIdx];
        if (t < pn.startFrame || t > pn.endFrame) continue;
        const noteOff = nIdx * nBins;
        const local = perEditLocal[editSlot];
        const localT = t - local.startFrame;
        const frameStartSrc = t * nBins * 2;
        const frameStartDst = localT * nBins * 2;
        for (let c = 0; c < channels; c++) {
          const stftC = cache.complexByChannel[c];
          const buf = local.byChannel[c];
          for (let b = 0; b < nBins; b++) {
            const cs = colSum[b];
            if (cs < 1e-12) continue;
            const factor = demand[noteOff + b] / cs;
            if (factor === 0) continue;
            const iSrc = frameStartSrc + b * 2;
            const iDst = frameStartDst + b * 2;
            buf[iDst] = factor * stftC[iSrc];
            buf[iDst + 1] = factor * stftC[iSrc + 1];
          }
        }
      }
    }

    // ISTFT each edited note's local spectrum → per-channel time signal,
    // placed back into a full-length output buffer at startFrame * hop.
    const out = new Map<string, AudioBuffer>();
    const fullFrames = cache.audio.frames;
    for (let editSlot = 0; editSlot < editedCount; editSlot++) {
      const nIdx = selectedIdxs[editSlot];
      const note = selectedByIdx.get(nIdx)!;
      const local = perEditLocal[editSlot];
      const sampleOffset = local.startFrame * hop;
      const samples = new Float32Array(fullFrames * channels);
      let istftRmsSum = 0;
      let istftRmsCount = 0;
      for (let c = 0; c < channels; c++) {
        const localTime = cpuIstft(local.byChannel[c], local.localFrames, nFft, hop);
        istftRmsSum += rmsOfArray(localTime);
        istftRmsCount++;
        const dstBase = c * fullFrames + sampleOffset;
        const copyLen = Math.min(localTime.length, fullFrames - sampleOffset);
        if (copyLen > 0) samples.set(localTime.subarray(0, copyLen), dstBase);
      }
      const avgIstftRms = istftRmsSum / Math.max(1, istftRmsCount);
      console.log(`[polypitch.extract] note ${note.id} midi=${note.pitchMidi} t=[${note.startSec.toFixed(2)},${note.endSec.toFixed(2)}]s localFrames=${local.localFrames} istftRms(per-chan avg)=${avgIstftRms.toExponential(2)}`);
      out.set(note.id, { samples, channels, sampleRate: PUBLIC_SR, frames: fullFrames });
    }
    return out;
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

}

// ---- CPU ISTFT — Cooley-Tukey radix-2 iFFT + sqrt-Hann overlap-add ---------
// Replaces the GPU istft.wgsl path, which produced silent output for this
// use case despite non-zero inputs. Matches the STFT analysis window (sqrt-
// Hann on both sides → combined Hann) used by stft.wgsl.

function cpuIstft(
  rfftComplex: Float32Array, nFrames: number, nFft: number, hop: number,
): Float32Array {
  const rfftBins = nFft / 2 + 1;
  const outLen = nFft + (nFrames - 1) * hop;
  const out = new Float32Array(outLen);
  const winSum = new Float32Array(outLen);

  // sqrt-Hann window.
  const win = new Float32Array(nFft);
  for (let i = 0; i < nFft; i++) {
    win[i] = Math.sqrt(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nFft));
  }

  const re = new Float32Array(nFft);
  const im = new Float32Array(nFft);

  for (let t = 0; t < nFrames; t++) {
    // Unpack rfft-packed frame into full complex spectrum (mirror conjugate).
    for (let b = 0; b < rfftBins; b++) {
      re[b] = rfftComplex[(t * rfftBins + b) * 2];
      im[b] = rfftComplex[(t * rfftBins + b) * 2 + 1];
    }
    for (let b = 1; b < nFft / 2; b++) {
      re[nFft - b] = re[b];
      im[nFft - b] = -im[b];
    }

    cpuIfft(re, im);

    // sqrt-Hann synthesis window + overlap-add.
    const offset = t * hop;
    for (let i = 0; i < nFft; i++) {
      const w = win[i];
      out[offset + i] += re[i] * w;
      winSum[offset + i] += w * w;
    }
  }

  // Normalise by sum-of-squared-window so overlapping frames reconstruct.
  for (let i = 0; i < outLen; i++) {
    if (winSum[i] > 1e-12) out[i] /= winSum[i];
  }

  return out;
}

// Radix-2 in-place iFFT. Expects power-of-2 length. Leaves result in `re`/`im`.
function cpuIfft(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  // Bit reversal.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  // Cooley-Tukey, INVERSE direction — twiddle angle is POSITIVE 2π/len.
  // (Forward FFT uses -2π/len. The previous code had `angle = -2 * Math.PI / len`
  // which is the forward direction; combined with the N division at the end
  // that produced time-reversed output at ~1/N² amplitude — explained why
  // extract.rms was 10000× smaller than the Python reproduction.)
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = 2 * Math.PI / len;
    const wRe0 = Math.cos(angle), wIm0 = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let cRe = 1, cIm = 0;
      for (let j = 0; j < half; j++) {
        const a = i + j, b = a + half;
        const tRe = re[b] * cRe - im[b] * cIm;
        const tIm = re[b] * cIm + im[b] * cRe;
        re[b] = re[a] - tRe; im[b] = im[a] - tIm;
        re[a] += tRe; im[a] += tIm;
        const nRe = cRe * wRe0 - cIm * wIm0;
        cIm = cRe * wIm0 + cIm * wRe0;
        cRe = nRe;
      }
    }
  }
  // Divide by n for iFFT normalisation.
  for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

// ---- module-local helpers (unchanged from upstream) -----------------------

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}
function extractChannel(audio: AudioBuffer, c: number): Float32Array {
  const off = c * audio.frames;
  return audio.samples.subarray(off, off + audio.frames);
}
function compactComplexToRfftBins(full: Float32Array, nFrames: number, nFft: number, rfftBins: number): Float32Array {
  const out = new Float32Array(nFrames * rfftBins * 2);
  for (let t = 0; t < nFrames; t++) {
    const srcOff = t * nFft * 2, dstOff = t * rfftBins * 2;
    out.set(full.subarray(srcOff, srcOff + rfftBins * 2), dstOff);
  }
  return out;
}

function rmsOfArray(buf: Float32Array): number {
  if (buf.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
  return Math.sqrt(s / buf.length);
}

// RMS over a time window [startSec, endSec], averaged across channels.
// Used by render() to confirm each edited note actually carries signal in
// its extracted window. 0 here = extraction produced silence for that note.
function windowRms(audio: AudioBuffer, startSec: number, endSec: number, sr: number): number {
  const frames = audio.frames;
  const ch = audio.channels;
  const s0 = Math.max(0, Math.floor(startSec * sr));
  const s1 = Math.min(frames, Math.ceil(endSec * sr));
  if (s1 <= s0) return 0;
  let sum = 0;
  let count = 0;
  for (let c = 0; c < ch; c++) {
    const off = c * frames;
    for (let i = s0; i < s1; i++) {
      const v = audio.samples[off + i];
      sum += v * v;
      count++;
    }
  }
  return count === 0 ? 0 : Math.sqrt(sum / count);
}

// ---- spectral-portion + resample pitch shift (Python parity) --------------

interface PortionNote {
  startFrame: number;
  endFrame: number;
  f0Hz: number;
  energy: number;
}

/**
 * Pitch-shift a note's isolated time signal by resampling its [startSec, endSec]
 * window. Clean shift — no phase-vocoder artefacts — because the signal is
 * already single-note-isolated by spectral portion allocation.
 *
 * For pitch-up the resampled segment is shorter by 1/ratio, so the note tail
 * ends ~1/ratio earlier; the gap is inaudible at chord-edit granularity.
 * For pitch-down the segment is longer by ratio and the tail runs past the
 * original endSec into following silence.
 */
function pitchShiftByResample(
  note: Note,
  original: AudioBuffer,
  ratio: number,
  sampleRate: number,
): AudioBuffer {
  const frames = original.frames;
  const channels = original.channels;
  const startSample = Math.max(0, Math.floor(note.startSec * sampleRate));
  const endSample = Math.min(frames, Math.ceil(note.endSec * sampleRate));
  const segLen = Math.max(0, endSample - startSample);

  const out = new Float32Array(frames * channels);
  if (segLen === 0) {
    return { samples: out, channels, sampleRate, frames };
  }

  // Resampled segment length. Pitch-up (ratio > 1) compresses time.
  const shiftedLen = Math.max(1, Math.round(segLen / ratio));

  for (let c = 0; c < channels; c++) {
    const srcOff = c * frames;
    const dstOff = c * frames;
    // Linear-interp resample: out[n] = src[start + n*ratio]. Cheap and
    // sufficient — the note signal is already band-limited by the STFT/ISTFT.
    for (let n = 0; n < shiftedLen; n++) {
      const t = n * ratio;
      const i0 = Math.floor(t);
      const frac = t - i0;
      const s0 = startSample + i0;
      const s1 = s0 + 1;
      if (s0 >= endSample) break;
      const a = original.samples[srcOff + s0];
      const b = s1 < endSample ? original.samples[srcOff + s1] : 0;
      const writeIdx = dstOff + startSample + n;
      if (writeIdx >= dstOff + frames) break;
      out[writeIdx] = a + (b - a) * frac;
    }
  }

  return { samples: out, channels, sampleRate, frames };
}
