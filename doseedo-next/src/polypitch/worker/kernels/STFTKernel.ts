/**
 * STFTKernel — windowed framing + forward FFT for mono PCM.
 *
 * Two-stage pipeline:
 *   1) `stft.wgsl` extracts and sqrt-Hann-windows `n_frames` frames from
 *      the input PCM into a complex (re, 0) scratch buffer.
 *   2) An embedded FFTKernel transforms the complex scratch in-place to
 *      produce the full complex spectrum. The caller then slices the
 *      first n_fft/2 + 1 bins (the second half is the conjugate mirror).
 *
 * The output layout the caller receives is the full `[n_frames, n_fft, 2]`
 * complex tensor; downstream consumers (MagSpec, HCQT) only look at the
 * first `n_fft/2 + 1` bins. This keeps the FFT shader simple — a separate
 * "to rfft bins" copy pass is not worth the extra dispatch.
 *
 * Patent citation: sqrt(Hann) window per US 8,022,286 B2 col.6 L.5-25.
 */

import { FFTKernel } from "./FFTKernel";
import shaderSource from "../wgsl/stft";

export interface STFTConfig {
  readonly nFft: number;
  readonly hop: number;
  readonly sampleRate: number;
  readonly maxNFrames: number;
}

const COMPLEX_F32_BYTES = 8;

function buildSqrtHann(nFft: number): Float32Array {
  const w = new Float32Array(nFft);
  for (let n = 0; n < nFft; n++) {
    // Periodic Hann (endpoint-excluded) for correct COLA.
    const hann = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / nFft);
    w[n] = Math.sqrt(Math.max(hann, 0));
  }
  return w;
}

export class STFTKernel {
  private readonly device: GPUDevice;
  private readonly cfg: STFTConfig;
  private readonly pipeline: GPUComputePipeline;
  private readonly uniform: GPUBuffer;
  private readonly windowBuffer: GPUBuffer;
  private readonly fft: FFTKernel;
  private readonly complexScratch: GPUBuffer;
  private bindGroupCache: { pcm: GPUBuffer; out: GPUBuffer; bg: GPUBindGroup } | null = null;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private disposed = false;

  private constructor(
    device: GPUDevice,
    cfg: STFTConfig,
    pipeline: GPUComputePipeline,
    uniform: GPUBuffer,
    windowBuffer: GPUBuffer,
    fft: FFTKernel,
    complexScratch: GPUBuffer,
    bindGroupLayout: GPUBindGroupLayout
  ) {
    this.device = device;
    this.cfg = cfg;
    this.pipeline = pipeline;
    this.uniform = uniform;
    this.windowBuffer = windowBuffer;
    this.fft = fft;
    this.complexScratch = complexScratch;
    this.bindGroupLayout = bindGroupLayout;
  }

  static async create(device: GPUDevice, cfg: STFTConfig): Promise<STFTKernel> {
    if (cfg.maxNFrames < 1) throw new Error("STFTKernel: maxNFrames must be >= 1");
    if (cfg.hop < 1 || cfg.hop > cfg.nFft) throw new Error("STFTKernel: invalid hop");

    const module = device.createShaderModule({
      label: `STFTKernel.shader(nFft=${cfg.nFft})`,
      code: shaderSource,
    });

    const bindGroupLayout = device.createBindGroupLayout({
      label: "STFTKernel.bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    const pipeline = device.createComputePipeline({
      label: "STFTKernel.pipeline",
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: "STFTKernel.pl",
      }),
      compute: { module, entryPoint: "main" },
    });

    const uniform = device.createBuffer({
      label: "STFTKernel.uniform",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const sqrtHann = buildSqrtHann(cfg.nFft);
    const windowBuffer = device.createBuffer({
      label: "STFTKernel.window",
      size: sqrtHann.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // `sqrtHann.buffer` is always an ArrayBuffer (we just allocated it with
    // `new Float32Array(n)`); the cast narrows the TS 5.9 `ArrayBufferLike`
    // union down to satisfy writeBuffer's `ArrayBuffer`-only parameter.
    device.queue.writeBuffer(windowBuffer, 0, sqrtHann.buffer as ArrayBuffer);

    const complexScratchBytes = cfg.maxNFrames * cfg.nFft * COMPLEX_F32_BYTES;
    const complexScratch = device.createBuffer({
      label: "STFTKernel.complexScratch",
      size: complexScratchBytes,
      usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const fft = await FFTKernel.create(device, {
      nFft: cfg.nFft,
      inverse: false,
      maxBatchSize: cfg.maxNFrames,
    });

    return new STFTKernel(
      device,
      cfg,
      pipeline,
      uniform,
      windowBuffer,
      fft,
      complexScratch,
      bindGroupLayout
    );
  }

  computeNFrames(nSamples: number): number {
    if (nSamples <= 0) return 0;
    if (nSamples < this.cfg.nFft) return 1;
    const remainder = (nSamples - this.cfg.nFft) % this.cfg.hop;
    const padded = remainder === 0 ? nSamples : nSamples + (this.cfg.hop - remainder);
    return 1 + Math.floor((padded - this.cfg.nFft) / this.cfg.hop);
  }

  requiredOutputBytes(nFrames: number): number {
    return nFrames * this.cfg.nFft * COMPLEX_F32_BYTES;
  }

  /**
   * Enqueue an STFT. `pcmInput` must hold at least `nSamples` float32s.
   * `outputComplex` receives `[nFrames, nFft, 2]` interleaved float32.
   */
  run(
    pcmInput: GPUBuffer,
    outputComplex: GPUBuffer,
    nFrames: number,
    nSamples: number,
    encoder?: GPUCommandEncoder
  ): void {
    this.assertLive();
    if (nFrames < 1 || nFrames > this.cfg.maxNFrames) {
      throw new Error(
        `STFTKernel.run: nFrames=${nFrames} out of range [1, ${this.cfg.maxNFrames}]`
      );
    }
    const byteLen = nFrames * this.cfg.nFft * COMPLEX_F32_BYTES;
    if (outputComplex.size < byteLen) {
      throw new Error(
        `STFTKernel.run: output buffer too small (need ${byteLen}, have ${outputComplex.size})`
      );
    }

    // Bind groups must reference the *current* pcmInput + outputComplex.
    // We cache the last one for hot reuse.
    let bg = this.bindGroupCache?.bg;
    if (
      !bg ||
      this.bindGroupCache?.pcm !== pcmInput ||
      this.bindGroupCache.out !== this.complexScratch
    ) {
      bg = this.device.createBindGroup({
        label: "STFTKernel.bg",
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: { buffer: pcmInput } },
          { binding: 2, resource: { buffer: this.windowBuffer } },
          { binding: 3, resource: { buffer: this.complexScratch } },
        ],
      });
      this.bindGroupCache = { pcm: pcmInput, out: this.complexScratch, bg };
    }

    const uniformData = new Uint32Array([this.cfg.nFft, this.cfg.hop, nFrames, nSamples]);
    this.device.queue.writeBuffer(this.uniform, 0, uniformData);

    const enc = encoder ?? this.device.createCommandEncoder({ label: "STFTKernel.enc" });

    // Stage 1: windowing.
    const pass = enc.beginComputePass({ label: "STFTKernel.window.pass" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bg);
    const wgx = Math.ceil(this.cfg.nFft / 64);
    pass.dispatchWorkgroups(wgx, nFrames, 1);
    pass.end();

    // Stage 2: FFT in place — use complexScratch as both input and output.
    // FFTKernel copies input→internal scratchA before running, so passing
    // the same buffer twice is safe.
    this.fft.run(this.complexScratch, outputComplex, nFrames, enc);

    if (!encoder) {
      this.device.queue.submit([enc.finish()]);
    }
  }

  get nFft(): number {
    return this.cfg.nFft;
  }

  get hop(): number {
    return this.cfg.hop;
  }

  get nBins(): number {
    return this.cfg.nFft / 2 + 1;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.uniform.destroy();
    this.windowBuffer.destroy();
    this.complexScratch.destroy();
    this.fft.dispose();
  }

  private assertLive(): void {
    if (this.disposed) throw new Error("STFTKernel used after dispose()");
  }
}
