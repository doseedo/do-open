/**
 * Global ONNX-Runtime inference queue.
 *
 * ORT-web's jsep layer wraps every session.run with a module-global `$c`
 * lock. If a second session.run fires while the first holds the lock,
 * ORT-Web throws "Session already started" / "Session mismatch". The lock
 * is ONE instance on the WASM module — it doesn't care whether the EP is
 * webgpu, webnn, or wasm. So anything the jsep decorator wraps has to go
 * through this queue.
 *
 * In practice every WebGPU session (latentEncoder, sem4Decoder,
 * latentDemucsV4, latentPitch, …) AND basic-pitch's WASM session belong
 * here. The name is historical; treat it as the process-wide ORT queue.
 * rmsDemucs still has its own queue — that's fine, it's additional
 * serialization inside a single model.
 *
 * Usage:
 *   import { ortWebGPURun } from './webgpuOrtQueue';
 *   const out = await ortWebGPURun(() => sess.run(feeds));
 */

let _queue = Promise.resolve();

export function ortWebGPURun(fn) {
  const work = _queue.then(() => fn());
  // keep the queue alive on failure — one bad run shouldn't wedge the whole
  // pipeline. The caller still sees the rejection from `work`.
  _queue = work.catch(() => {});
  return work;
}
