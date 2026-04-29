/**
 * MagSpecKernel — complex → magnitude (or magnitude^2) utility.
 *
 * Used by:
 *   - HCQT postprocess (but the HCQT kernel already emits magnitude)
 *   - Mask U-Net input (power=1)
 *   - Harmonic-prior path (power=2)
 */

import shaderSource from "../wgsl/magspec";

export interface MagSpecConfig {
  readonly nFft: number;
  readonly maxNFrames: number;
}

export class MagSpecKernel {
  private readonly device: GPUDevice;
  private readonly cfg: MagSpecConfig;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private readonly uniform: GPUBuffer;
  private bgCache: {
    input: GPUBuffer;
    output: GPUBuffer;
    bg: GPUBindGroup;
  } | null = null;
  private disposed = false;

  private constructor(
    device: GPUDevice,
    cfg: MagSpecConfig,
    pipeline: GPUComputePipeline,
    bindGroupLayout: GPUBindGroupLayout,
    uniform: GPUBuffer
  ) {
    this.device = device;
    this.cfg = cfg;
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;
    this.uniform = uniform;
  }

  static async create(device: GPUDevice, cfg: MagSpecConfig): Promise<MagSpecKernel> {
    const module = device.createShaderModule({
      label: "MagSpecKernel.shader",
      code: shaderSource,
    });
    const bindGroupLayout = device.createBindGroupLayout({
      label: "MagSpecKernel.bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const pipeline = device.createComputePipeline({
      label: "MagSpecKernel.pipeline",
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: "MagSpecKernel.pl",
      }),
      compute: { module, entryPoint: "main" },
    });
    const uniform = device.createBuffer({
      label: "MagSpecKernel.uniform",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    return new MagSpecKernel(device, cfg, pipeline, bindGroupLayout, uniform);
  }

  /**
   * `complex` is `[nFrames, nBins, 2]` interleaved. `mag` receives
   * `[nFrames, nBins]` float32 with either |X| (power=1) or |X|^2
   * (power=2). Here `nBins` is the FULL FFT bin count the caller used —
   * we take whatever length fits the output buffer.
   */
  run(
    complex: GPUBuffer,
    mag: GPUBuffer,
    nFrames: number,
    nBins: number,
    power: 1 | 2,
    encoder?: GPUCommandEncoder
  ): void {
    this.assertLive();
    if (nFrames < 1 || nFrames > this.cfg.maxNFrames) {
      throw new Error(`MagSpecKernel.run: nFrames=${nFrames} out of range`);
    }
    if (power !== 1 && power !== 2) {
      throw new Error(`MagSpecKernel.run: power must be 1 or 2 (got ${power})`);
    }
    const total = nFrames * nBins;
    if (mag.size < total * 4) {
      throw new Error(`MagSpecKernel.run: mag buffer too small (need ${total * 4})`);
    }
    if (complex.size < total * 8) {
      throw new Error(`MagSpecKernel.run: complex buffer too small (need ${total * 8})`);
    }

    const uData = new Uint32Array([nFrames, nBins, power, 0]);
    this.device.queue.writeBuffer(this.uniform, 0, uData);

    let bg = this.bgCache?.bg;
    if (!bg || this.bgCache?.input !== complex || this.bgCache.output !== mag) {
      bg = this.device.createBindGroup({
        label: "MagSpecKernel.bg",
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: { buffer: complex } },
          { binding: 2, resource: { buffer: mag } },
        ],
      });
      this.bgCache = { input: complex, output: mag, bg };
    }

    const enc = encoder ?? this.device.createCommandEncoder({ label: "MagSpecKernel.enc" });
    const pass = enc.beginComputePass({ label: "MagSpecKernel.pass" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(total / 64), 1, 1);
    pass.end();
    if (!encoder) {
      this.device.queue.submit([enc.finish()]);
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.uniform.destroy();
  }

  private assertLive(): void {
    if (this.disposed) throw new Error("MagSpecKernel used after dispose()");
  }
}
