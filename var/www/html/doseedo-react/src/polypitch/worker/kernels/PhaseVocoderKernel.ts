/**
 * PhaseVocoderKernel — Laroche-Dolson bin-wise phase vocoder (in-place).
 *
 * Applies per-frame pitch shift to a complex STFT. Because each frame's
 * output phase depends on the previous frame's state, we dispatch one
 * compute pass per frame with `frame_idx` in the uniform. Within a
 * frame, bins are processed in parallel.
 *
 * Recommended usage: clear the `prevInPhase` / `accumOutPhase` buffers
 * at the start of each pipeline render (call `reset()`).
 *
 * Caveat: this kernel does not re-bin (time-stretch → resample) — it
 * modifies phase only. That gives the correct pitch shift for stationary
 * signals up to ~3 semitones; larger shifts develop spectral smearing
 * and should route through the DDSP head.
 */

import shaderSource from "../wgsl/phase_vocoder";

export interface PhaseVocoderConfig {
  readonly nFft: number;
  readonly hop: number;
  readonly maxNFrames: number;
}

export class PhaseVocoderKernel {
  private readonly device: GPUDevice;
  private readonly cfg: PhaseVocoderConfig;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private readonly uniform: GPUBuffer;
  private readonly prevInPhase: GPUBuffer;
  private readonly accumOutPhase: GPUBuffer;
  private bgCache: {
    stft: GPUBuffer;
    ratios: GPUBuffer;
    bg: GPUBindGroup;
  } | null = null;
  private disposed = false;

  private constructor(
    device: GPUDevice,
    cfg: PhaseVocoderConfig,
    pipeline: GPUComputePipeline,
    bindGroupLayout: GPUBindGroupLayout,
    uniform: GPUBuffer,
    prevInPhase: GPUBuffer,
    accumOutPhase: GPUBuffer
  ) {
    this.device = device;
    this.cfg = cfg;
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;
    this.uniform = uniform;
    this.prevInPhase = prevInPhase;
    this.accumOutPhase = accumOutPhase;
  }

  static async create(
    device: GPUDevice,
    cfg: PhaseVocoderConfig
  ): Promise<PhaseVocoderKernel> {
    const module = device.createShaderModule({
      label: "PhaseVocoderKernel.shader",
      code: shaderSource,
    });

    const bindGroupLayout = device.createBindGroupLayout({
      label: "PhaseVocoderKernel.bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    const pipeline = device.createComputePipeline({
      label: "PhaseVocoderKernel.pipeline",
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: "PhaseVocoderKernel.pl",
      }),
      compute: { module, entryPoint: "main" },
    });

    const uniform = device.createBuffer({
      label: "PhaseVocoderKernel.uniform",
      size: 32, // 5 u32 padded to 16-byte boundary
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const nBins = Math.floor(cfg.nFft / 2) + 1;
    const phaseBytes = nBins * 4;
    const prevInPhase = device.createBuffer({
      label: "PhaseVocoderKernel.prevInPhase",
      size: phaseBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const accumOutPhase = device.createBuffer({
      label: "PhaseVocoderKernel.accumOutPhase",
      size: phaseBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return new PhaseVocoderKernel(
      device,
      cfg,
      pipeline,
      bindGroupLayout,
      uniform,
      prevInPhase,
      accumOutPhase
    );
  }

  /** Zero the phase accumulators. Call at the start of a render pass. */
  reset(): void {
    this.assertLive();
    const nBins = Math.floor(this.cfg.nFft / 2) + 1;
    const zeros = new Float32Array(nBins);
    this.device.queue.writeBuffer(this.prevInPhase, 0, zeros);
    this.device.queue.writeBuffer(this.accumOutPhase, 0, zeros);
  }

  /**
   * Enqueue phase-vocoder passes for all frames. `stftComplex` is the
   * `[nFrames, nBins, 2]` buffer, modified in-place. `pitchShiftRatios`
   * holds one float per frame: 2^(semitones/12).
   */
  run(
    stftComplex: GPUBuffer,
    pitchShiftRatios: GPUBuffer,
    nFrames: number,
    encoder?: GPUCommandEncoder
  ): void {
    this.assertLive();
    if (nFrames < 1 || nFrames > this.cfg.maxNFrames) {
      throw new Error(`PhaseVocoderKernel.run: nFrames=${nFrames} out of range`);
    }

    const nBins = Math.floor(this.cfg.nFft / 2) + 1;
    if (pitchShiftRatios.size < nFrames * 4) {
      throw new Error("PhaseVocoderKernel.run: ratios buffer too small");
    }

    let bg = this.bgCache?.bg;
    if (
      !bg ||
      this.bgCache?.stft !== stftComplex ||
      this.bgCache.ratios !== pitchShiftRatios
    ) {
      bg = this.device.createBindGroup({
        label: "PhaseVocoderKernel.bg",
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: { buffer: stftComplex } },
          { binding: 2, resource: { buffer: pitchShiftRatios } },
          { binding: 3, resource: { buffer: this.prevInPhase } },
          { binding: 4, resource: { buffer: this.accumOutPhase } },
        ],
      });
      this.bgCache = { stft: stftComplex, ratios: pitchShiftRatios, bg };
    }

    const enc = encoder ?? this.device.createCommandEncoder({ label: "PhaseVocoderKernel.enc" });
    const wgX = Math.ceil(nBins / 64);

    for (let f = 0; f < nFrames; f++) {
      const uData = new Uint32Array([nFrames, nBins, this.cfg.nFft, this.cfg.hop, f, 0, 0, 0]);
      this.device.queue.writeBuffer(this.uniform, 0, uData);
      const pass = enc.beginComputePass({ label: `PhaseVocoderKernel.pass.${f}` });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX, 1, 1);
      pass.end();
    }

    if (!encoder) {
      this.device.queue.submit([enc.finish()]);
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.uniform.destroy();
    this.prevInPhase.destroy();
    this.accumOutPhase.destroy();
  }

  private assertLive(): void {
    if (this.disposed) throw new Error("PhaseVocoderKernel used after dispose()");
  }
}
