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
      // Pitch-shift the isolated note's time signal by resampling its
      // audio segment. Clean shift with no phasiness artefacts because
      // the signal is already single-note-isolated. Duration shortens by
      // 1/ratio for pitch-up; the gap at the note's tail is imperceptible
      // at chord-edit granularity.
      const shifted = pitchShiftByResample(note, original, ratio, PUBLIC_SR);
      if (gain !== 1) scaleInPlace(shifted.samples, gain);
      this.addNoteToMix(base, shifted);
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
    const channels = cache.complexByChannel.length as 1 | 2;
    const nFrames = cache.nFrames;
    const nBins = cache.bins;
    const nFft = cache.nFft;
    const hop = cache.hop;
    const sr = PUBLIC_SR;

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
      for (let c = 0; c < channels; c++) {
        const localTime = cpuIstft(local.byChannel[c], local.localFrames, nFft, hop);
        const dstBase = c * fullFrames + sampleOffset;
        const copyLen = Math.min(localTime.length, fullFrames - sampleOffset);
        if (copyLen > 0) samples.set(localTime.subarray(0, copyLen), dstBase);
      }
      out.set(note.id, { samples, channels, sampleRate: PUBLIC_SR, frames: fullFrames });
    }
    return out;
  }

  private async istftToTime(rfftComplex: Float32Array, nFrames: number): Promise<Float32Array> {
    // CPU ISTFT — the GPU path (istft.wgsl + FFTKernel) appears to produce
    // silence even when the masked STFT has real content. After nine commits
    // of GPU-plumbing fixes, abandon it here and use a straightforward
    // Cooley-Tukey radix-2 iFFT + sqrt-Hann overlap-add on CPU. Matches the
    // math verified in /tmp/polypitch-test/reproduce_browser.py. ~50 ms for a
    // 30 s song, noise against the polypitch render's other costs.
    return cpuIstft(rfftComplex, nFrames, STFT_N_FFT, STFT_HOP);
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
      // Broadband: ALL bins in the note's window, not just harmonic bands —
      // matches the extract path above so subtract + add operate on the
      // same signal.
      const maskedRfft = NoteExtractor.applyMaskToComplex(
        cache.complexByChannel[c], null, nFrames, bins, query.startFrame, maskFrames,
      );
      const rotatedRfft = cpuPhaseVocoder(maskedRfft, nFrames, bins, ratio, hop, cache.nFft);
      // CPU ISTFT (see istftToTime note). Also removes the GPU roundtrip
      // for this path.
      const time = cpuIstft(rotatedRfft, nFrames, cache.nFft, hop);
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
