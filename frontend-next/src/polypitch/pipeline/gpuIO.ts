/**
 * gpuIO — tiny CPU↔GPU buffer helpers used by `Pipeline` to upload analysis
 * data to Agent A's kernels and read back the results. Deliberately small:
 * `Pipeline` owns its own buffers per-analysis so we can accurately size them
 * for the cached `nFrames` before dispatch.
 *
 * All helpers in this file assume an already-acquired `GPUDevice`.
 */

export function createStorageBuffer(
  device: GPUDevice,
  byteLength: number,
  label: string,
  extraUsage = 0,
): GPUBuffer {
  const aligned = (Math.max(1, byteLength) + 3) & ~3;
  return device.createBuffer({
    label,
    size: aligned,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST |
      extraUsage,
  });
}

export function uploadFloat32(
  device: GPUDevice,
  buf: GPUBuffer,
  data: Float32Array,
  byteOffset = 0,
): void {
  // WebGPU's writeBuffer takes an ArrayBufferLike — narrow to the underlying
  // ArrayBuffer so TS doesn't complain under verbatimModuleSyntax-adjacent
  // strictness.
  device.queue.writeBuffer(buf, byteOffset, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
}

/**
 * Copy `byteLength` bytes of `src` into a fresh MAP_READ-able buffer, map it,
 * and return a Float32Array view that owns its own ArrayBuffer (so callers can
 * safely transfer it across a postMessage boundary).
 */
export async function readbackFloat32(
  device: GPUDevice,
  src: GPUBuffer,
  byteLength: number,
): Promise<Float32Array> {
  const aligned = (byteLength + 3) & ~3;
  const staging = device.createBuffer({
    label: "gpuIO.staging",
    size: aligned,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder({ label: "gpuIO.readback.enc" });
  enc.copyBufferToBuffer(src, 0, staging, 0, aligned);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const mapped = staging.getMappedRange(0, aligned);
  // Copy out, then unmap + destroy the staging buffer.
  const out = new Float32Array(byteLength / 4);
  out.set(new Float32Array(mapped, 0, byteLength / 4));
  staging.unmap();
  staging.destroy();
  return out;
}
