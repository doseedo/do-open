/**
 * HCQTKernel — Harmonic Constant-Q Transform.
 *
 * Upload-side: precompute the sparse complex CQT kernel matrix spanning
 * all `n_harmonics` harmonic blocks, then upload to GPU as four storage
 * buffers (values_re, values_im, col_indices, row_start).
 *
 * Run-side: one dispatch with (n_harmonics * n_frames * cent_bins)
 * output elements, each doing a sparse dot-product of a complex STFT
 * row with the kernel row for its (harmonic, cent) pair. Output is the
 * magnitude.
 *
 * See hcqt.wgsl for the inner-loop math.
 */

import {
  buildCQTKernelMatrix,
  type CQTKernelMatrixConfig,
  type SparseCQTKernel,
} from "./cqtKernelMatrix";
import shaderSource from "../wgsl/hcqt";

export interface HCQTConfig {
  readonly binsPerOctave: number;
  readonly nOctaves: number;
  readonly nHarmonics: number;
  readonly harmonicScales: readonly number[];
  readonly fminHz: number;
  readonly sampleRate: number;
  readonly nFft: number;
  readonly maxNFrames: number;
}

export class HCQTKernel {
  private readonly device: GPUDevice;
  private readonly cfg: HCQTConfig;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private readonly uniform: GPUBuffer;
  private readonly kValuesRe: GPUBuffer;
  private readonly kValuesIm: GPUBuffer;
  private readonly kIndices: GPUBuffer;
  private readonly kRowStart: GPUBuffer;
  private readonly matrix: SparseCQTKernel;
  private bgCache: { stft: GPUBuffer; out: GPUBuffer; bg: GPUBindGroup } | null = null;
  private disposed = false;

  private constructor(
    device: GPUDevice,
    cfg: HCQTConfig,
    pipeline: GPUComputePipeline,
    bindGroupLayout: GPUBindGroupLayout,
    uniform: GPUBuffer,
    kValuesRe: GPUBuffer,
    kValuesIm: GPUBuffer,
    kIndices: GPUBuffer,
    kRowStart: GPUBuffer,
    matrix: SparseCQTKernel
  ) {
    this.device = device;
    this.cfg = cfg;
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;
    this.uniform = uniform;
    this.kValuesRe = kValuesRe;
    this.kValuesIm = kValuesIm;
    this.kIndices = kIndices;
    this.kRowStart = kRowStart;
    this.matrix = matrix;
  }

  static async create(device: GPUDevice, cfg: HCQTConfig): Promise<HCQTKernel> {
    const matrixCfg: CQTKernelMatrixConfig = {
      binsPerOctave: cfg.binsPerOctave,
      nOctaves: cfg.nOctaves,
      nHarmonics: cfg.nHarmonics,
      harmonicScales: cfg.harmonicScales,
      fminHz: cfg.fminHz,
      sampleRate: cfg.sampleRate,
      nFft: cfg.nFft,
    };
    const matrix = buildCQTKernelMatrix(matrixCfg);

    const module = device.createShaderModule({
      label: "HCQTKernel.shader",
      code: shaderSource,
    });

    const bindGroupLayout = device.createBindGroupLayout({
      label: "HCQTKernel.bgl",
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
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    const pipeline = device.createComputePipeline({
      label: "HCQTKernel.pipeline",
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: "HCQTKernel.pl",
      }),
      compute: { module, entryPoint: "main" },
    });

    const uniform = device.createBuffer({
      label: "HCQTKernel.uniform",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Matrix buffers — upload once and reuse.
    const mkStorage = (data: Float32Array | Uint32Array, label: string): GPUBuffer => {
      const buf = device.createBuffer({
        label,
        size: Math.max(data.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      if (data.byteLength > 0) {
        // Narrow TS 5.9 `ArrayBufferLike` → `ArrayBuffer` (we always
        // allocate these arrays with `new XArray(n)` in cqtKernelMatrix).
        device.queue.writeBuffer(buf, 0, data.buffer as ArrayBuffer);
      }
      return buf;
    };

    const kValuesRe = mkStorage(matrix.valuesRe, "HCQTKernel.kValuesRe");
    const kValuesIm = mkStorage(matrix.valuesIm, "HCQTKernel.kValuesIm");
    const kIndices = mkStorage(matrix.colIndices, "HCQTKernel.kIndices");
    const kRowStart = mkStorage(matrix.rowStart, "HCQTKernel.kRowStart");

    return new HCQTKernel(
      device,
      cfg,
      pipeline,
      bindGroupLayout,
      uniform,
      kValuesRe,
      kValuesIm,
      kIndices,
      kRowStart,
      matrix
    );
  }

  get sparseMatrix(): SparseCQTKernel {
    return this.matrix;
  }

  get binsPerHarmonic(): number {
    return this.matrix.binsPerHarmonic;
  }

  /**
   * Enqueue an HCQT. `stftComplex` is `[nFrames, nBinsStft, 2]` f32
   * interleaved. `outputHCQT` receives
   * `[nHarmonics, nFrames, binsPerHarmonic]` float32 (magnitudes).
   */
  run(
    stftComplex: GPUBuffer,
    outputHCQT: GPUBuffer,
    nFrames: number,
    encoder?: GPUCommandEncoder
  ): void {
    this.assertLive();
    if (nFrames < 1 || nFrames > this.cfg.maxNFrames) {
      throw new Error(`HCQTKernel.run: nFrames=${nFrames} out of range`);
    }

    const nBinsStft = Math.floor(this.cfg.nFft / 2) + 1;
    const outElements = this.cfg.nHarmonics * nFrames * this.matrix.binsPerHarmonic;
    if (outputHCQT.size < outElements * 4) {
      throw new Error(
        `HCQTKernel.run: output buffer too small (need ${outElements * 4}, have ${outputHCQT.size})`
      );
    }

    const uniformData = new Uint32Array([
      nFrames,
      nBinsStft,
      this.matrix.binsPerHarmonic,
      this.cfg.nHarmonics,
    ]);
    this.device.queue.writeBuffer(this.uniform, 0, uniformData);

    let bg = this.bgCache?.bg;
    if (!bg || this.bgCache?.stft !== stftComplex || this.bgCache.out !== outputHCQT) {
      bg = this.device.createBindGroup({
        label: "HCQTKernel.bg",
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: { buffer: stftComplex } },
          { binding: 2, resource: { buffer: this.kValuesRe } },
          { binding: 3, resource: { buffer: this.kValuesIm } },
          { binding: 4, resource: { buffer: this.kIndices } },
          { binding: 5, resource: { buffer: this.kRowStart } },
          { binding: 6, resource: { buffer: outputHCQT } },
        ],
      });
      this.bgCache = { stft: stftComplex, out: outputHCQT, bg };
    }

    const enc = encoder ?? this.device.createCommandEncoder({ label: "HCQTKernel.enc" });
    const pass = enc.beginComputePass({ label: "HCQTKernel.pass" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bg);
    const wgX = Math.ceil(this.matrix.binsPerHarmonic / 64);
    pass.dispatchWorkgroups(wgX, nFrames, this.cfg.nHarmonics);
    pass.end();

    if (!encoder) {
      this.device.queue.submit([enc.finish()]);
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.uniform.destroy();
    this.kValuesRe.destroy();
    this.kValuesIm.destroy();
    this.kIndices.destroy();
    this.kRowStart.destroy();
  }

  private assertLive(): void {
    if (this.disposed) throw new Error("HCQTKernel used after dispose()");
  }
}
