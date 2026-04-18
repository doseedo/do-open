/**
 * Global WebGPU ONNX-Runtime inference queue.
 *
 * ORT-web's WebGPU EP shares one `GPUDevice` across every InferenceSession
 * and is not safe for concurrent `.run()` calls from different sessions.
 * Two overlapping runs surface as "Session mismatch" — we saw this the
 * first time sem4Decoder's `sem.run()` raced the analyze pipeline's
 * `latentEncoder.run()` after a file drop.
 *
 * Every WebGPU session.run call on the page must go through this queue
 * (latentEncoder, sem4Decoder, latentDemucsV4, latentDecoder, …).
 * WASM sessions (rmsDemucs) are NOT affected — they have their own
 * serializer inside the ORT WASM runtime.
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
