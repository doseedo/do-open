/**
 * FFTKernel — WebGPU Stockham radix-2 complex FFT / inverse FFT.
 *
 * Each `run()` call enqueues a single compute pass that transforms a batch
 * of `batchSize` complex n_fft-point signals in parallel. The kernel is
 * reused across calls; we cache the pipeline + bind-group layout on
 * construction.
 *
 * Internal layout is Stockham "auto-sort" (no bit-reversal), ping-ponging
 * between two storage buffers inside the shader. The caller provides
 * `input` (read) and `output` (write) — we orchestrate the ping-pong
 * internally by copying input → scratchA at the top of every run, then
 * running log2(n_fft) stages, and finally ensuring the result lands in
 * scratchA before blitting to the caller's output.
 *
 * Patent citation: the FFT itself is not patented; the sqrt-Hann windowing
 * (US 8,022,286 B2 col.6 L.5-25) lives in stft.wgsl / istft.wgsl, not here.
 */

import shaderSource from "../wgsl/fft";

export interface FFTConfig {
  readonly nFft: number;
  readonly inverse?: boolean;
  /** Maximum batch size; used to size internal scratch buffers. */
  readonly maxBatchSize: number;
}

const COMPLEX_F32_BYTES = 8; // 2 * float32

function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

function log2Int(n: number): number {
  // Integer log2 for powers of two only.
  let r = 0;
  let v = n;
  while (v > 1) {
    v >>= 1;
    r += 1;
  }
  return r;
}

export class FFTKernel {
  private readonly device: GPUDevice;
  private readonly cfg: FFTConfig;
  private readonly pipeline: GPUComputePipeline;
  private readonly uniform: GPUBuffer;
  private readonly scratchA: GPUBuffer;
  private readonly scratchB: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private disposed = false;

  private constructor(
    device: GPUDevice,
    cfg: FFTConfig,
    pipeline: GPUComputePipeline,
    uniform: GPUBuffer,
    scratchA: GPUBuffer,
    scratchB: GPUBuffer,
    bindGroup: GPUBindGroup
  ) {
    this.device = device;
    this.cfg = cfg;
    this.pipeline = pipeline;
    this.uniform = uniform;
    this.scratchA = scratchA;
    this.scratchB = scratchB;
    this.bindGroup = bindGroup;
  }

  static async create(device: GPUDevice, cfg: FFTConfig): Promise<FFTKernel> {
    if (!isPowerOfTwo(cfg.nFft) || cfg.nFft < 4) {
      throw new Error(`FFTKernel: nFft must be a power of two >= 4; got ${cfg.nFft}`);
    }
    if (cfg.nFft > 4096) {
      throw new Error(`FFTKernel: nFft > 4096 not supported (got ${cfg.nFft})`);
    }
    if (cfg.maxBatchSize < 1) {
      throw new Error(`FFTKernel: maxBatchSize must be >= 1`);
    }

    const module = device.createShaderModule({
      label: `FFTKernel.shader(nFft=${cfg.nFft})`,
      code: shaderSource,
    });

    const bindGroupLayout = device.createBindGroupLayout({
      label: "FFTKernel.bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    const pipeline = device.createComputePipeline({
      label: `FFTKernel.pipeline(nFft=${cfg.nFft})`,
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: "FFTKernel.pl",
      }),
      compute: { module, entryPoint: "main" },
    });

    const uniform = device.createBuffer({
      label: "FFTKernel.uniform",
      size: 16, // 4 x u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const scratchBytes = cfg.maxBatchSize * cfg.nFft * COMPLEX_F32_BYTES;
    const scratchA = device.createBuffer({
      label: "FFTKernel.scratchA",
      size: scratchBytes,
      usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const scratchB = device.createBuffer({
      label: "FFTKernel.scratchB",
      size: scratchBytes,
      usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const bindGroup = device.createBindGroup({
      label: "FFTKernel.bg",
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: scratchA } },
        { binding: 2, resource: { buffer: scratchB } },
      ],
    });

    return new FFTKernel(device, cfg, pipeline, uniform, scratchA, scratchB, bindGroup);
  }

  /**
   * Enqueue an FFT. `input` and `output` are complex float32 buffers of
   * size at least `batchSize * nFft * 8` bytes.
   *
   * If `encoder` is provided, uses it (caller manages submission). Else
   * creates a fresh CommandEncoder and submits immediately.
   */
  run(
    input: GPUBuffer,
    output: GPUBuffer,
    batchSize: number,
    encoder?: GPUCommandEncoder
  ): void {
    this.assertLive();
    if (batchSize < 1 || batchSize > this.cfg.maxBatchSize) {
      throw new Error(
        `FFTKernel.run: batchSize=${batchSize} out of range [1, ${this.cfg.maxBatchSize}]`
      );
    }
    const byteLen = batchSize * this.cfg.nFft * COMPLEX_F32_BYTES;
    if (input.size < byteLen) {
      throw new Error(
        `FFTKernel.run: input buffer too small (need ${byteLen} bytes, have ${input.size})`
      );
    }
    if (output.size < byteLen) {
      throw new Error(
        `FFTKernel.run: output buffer too small (need ${byteLen} bytes, have ${output.size})`
      );
    }

    const uniformData = new Uint32Array([
      this.cfg.nFft,
      this.cfg.inverse ? 1 : 0,
      batchSize,
      log2Int(this.cfg.nFft),
    ]);
    this.device.queue.writeBuffer(this.uniform, 0, uniformData);

    const enc = encoder ?? this.device.createCommandEncoder({ label: "FFTKernel.enc" });

    // Copy caller input into scratchA (shader reads from buf_a on stage 0).
    enc.copyBufferToBuffer(input, 0, this.scratchA, 0, byteLen);

    const pass = enc.beginComputePass({ label: "FFTKernel.pass" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    const half = this.cfg.nFft >>> 1;
    const wgx = Math.ceil(half / 64);
    pass.dispatchWorkgroups(wgx, batchSize, 1);
    pass.end();

    // Shader guarantees final result lives in scratchA.
    enc.copyBufferToBuffer(this.scratchA, 0, output, 0, byteLen);

    if (!encoder) {
      this.device.queue.submit([enc.finish()]);
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.uniform.destroy();
    this.scratchA.destroy();
    this.scratchB.destroy();
  }

  private assertLive(): void {
    if (this.disposed) throw new Error("FFTKernel used after dispose()");
  }
}
