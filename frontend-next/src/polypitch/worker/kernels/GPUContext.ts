/**
 * GPUContext — shared WebGPU device acquisition for all kernels.
 *
 * Caches a single `GPUDevice` per module load. Safe to call from any
 * worker — the first caller initializes, later callers get the same
 * device handle.
 *
 * We request high-performance power preference and the limits we need
 * for the STFT/FFT kernels (max storage buffer size, at least 256 MB,
 * which covers 60 seconds of 48 kHz stereo complex STFT).
 */

export interface GPUContextOptions {
  readonly powerPreference?: GPUPowerPreference;
  readonly label?: string;
}

export class GPUContext {
  private static devicePromise: Promise<GPUDevice> | null = null;

  /** Returns a cached GPUDevice. Throws if WebGPU is unavailable. */
  static getDevice(opts: GPUContextOptions = {}): Promise<GPUDevice> {
    if (this.devicePromise !== null) return this.devicePromise;
    this.devicePromise = this.acquireDevice(opts);
    return this.devicePromise;
  }

  /** Test helper: drop the cached device. Not for production use. */
  static _resetForTests(): void {
    this.devicePromise = null;
  }

  private static async acquireDevice(opts: GPUContextOptions): Promise<GPUDevice> {
    // `navigator` is available in both window and worker global scopes.
    const gpu: GPU | undefined = (globalThis as unknown as { navigator?: { gpu?: GPU } })
      .navigator?.gpu;
    if (!gpu) {
      throw new Error(
        "WebGPU not available. polypitch-browser requires a WebGPU-capable browser (Chrome 121+, Edge 121+, Safari 18+, Firefox 135+)."
      );
    }
    const adapter = await gpu.requestAdapter({
      powerPreference: opts.powerPreference ?? "high-performance",
    });
    if (!adapter) {
      throw new Error("Failed to acquire GPUAdapter.");
    }
    // Request the limits we actually need; stay within widely-supported caps.
    const desiredLimits: Record<string, number> = {
      maxStorageBufferBindingSize: Math.min(
        adapter.limits.maxStorageBufferBindingSize,
        1 << 30 // 1 GiB, clamped by adapter.
      ),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 30),
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
    };
    const device = await adapter.requestDevice({
      label: opts.label ?? "polypitch-device",
      requiredLimits: desiredLimits,
    });
    device.lost.then((info) => {
      // Force re-acquire on next call.
      this.devicePromise = null;
      if ((process.env.NODE_ENV === 'development')) {
        console.warn("[polypitch] GPUDevice lost:", info.message);
      }
    });
    return device;
  }

  /** Convenience: create a labelled GPUBuffer. */
  static createBuffer(
    device: GPUDevice,
    size: number,
    usage: GPUBufferUsageFlags,
    label: string
  ): GPUBuffer {
    if (size <= 0) {
      throw new Error(`createBuffer(${label}): size must be > 0 (got ${size})`);
    }
    // WebGPU requires buffer sizes be multiples of 4.
    const aligned = (size + 3) & ~3;
    return device.createBuffer({ size: aligned, usage, label });
  }
}
