/**
 * Web Worker for semantic encoder + decoder ONNX inference.
 *
 * v2: Loads two small packed models (63 MB total) instead of one
 * large split model (323 MB). For each decode request:
 *   1. Run sem_encoder: [1, 64, T] → [1, 128] per stem
 *   2. Run sem_decoder: [B, 64, T] + [B, 128] → [B, 2, T*1920] batched
 *
 * Protocol (unchanged from v1):
 *   main → worker:  { type: 'init' }
 *   worker → main:  { type: 'initDone' } | { type: 'initFailed', error }
 *   main → worker:  { type: 'decode', id, input, shape }
 *   worker → main:  { type: 'decoded', id, audio } (audio is Transferable)
 *   worker → main:  { type: 'decodeFailed', id, error }
 *   main → worker:  { type: 'envelope', id, input, T }
 *   worker → main:  { type: 'envelopeDone', id, envelope }
 *   worker → main:  { type: 'progress', pct, mb, elapsed }
 */

/* eslint-disable no-restricted-globals */

const ENC_MODEL_URL = '/static/models/sem_encoder_packed.onnx';
// Temp-warmstart decoder — same input shape as the old sem_decoder for
// (latent, sem_emb), plus a third stft_mask input. Warmstart design
// zero-initializes the mask path, so passing zeros reduces the model
// to sem_v1 behavior exactly. Real per-stem masks from v4-small will
// get piped through in a follow-up commit.
const DEC_MODEL_URL = '/static/models/sem_mask_temp_decoder_fp16.onnx';
const VIS_MODEL_URL = '/static/models/latent_visual.onnx';
const VIS_MODEL_DATA_URL = '/static/models/latent_visual.onnx.data';

const SEM_DIM = 128;
const LATENT_CHANNELS = 64;

let ort = null;
let encSession = null;  // semantic encoder
let decSession = null;  // semantic decoder
let visSession = null;  // tiny envelope model

async function fetchPacked(url, label) {
  const resp = await fetch(url, { cache: 'force-cache' });
  if (!resp.ok) throw new Error(`${label} HTTP ${resp.status}`);
  const total = parseInt(resp.headers.get('content-length') || '0', 10);
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
  }
  const bytes = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { bytes.set(c, off); off += c.byteLength; }
  return { bytes, loaded };
}

async function init() {
  ort = await import('onnxruntime-web');

  if (ort.env?.wasm) {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    ort.env.wasm.numThreads = 1; // workers are already off-main-thread
    ort.env.wasm.simd = true;
  }

  const t0 = performance.now();

  // Fetch both packed models (no external .data files needed)
  self.postMessage({ type: 'progress', pct: 0, mb: '0', elapsed: '0' });
  const enc = await fetchPacked(ENC_MODEL_URL, 'sem_encoder');
  const mb1 = (enc.loaded / 1e6).toFixed(0);
  const el1 = ((performance.now() - t0) / 1000).toFixed(1);
  self.postMessage({ type: 'progress', pct: 35, mb: mb1, elapsed: el1 });

  const dec = await fetchPacked(DEC_MODEL_URL, 'sem_decoder');
  // New decoder ships with external .onnx.data (fp16 weights).
  const decDataResp = await fetch(DEC_MODEL_URL + '.data', { cache: 'force-cache' });
  const decDataBytes = new Uint8Array(await decDataResp.arrayBuffer());
  const totalLoaded = enc.loaded + dec.loaded + decDataBytes.byteLength;
  const mb2 = (totalLoaded / 1e6).toFixed(0);
  const el2 = ((performance.now() - t0) / 1000).toFixed(1);
  self.postMessage({ type: 'progress', pct: 100, mb: mb2, elapsed: el2 });

  // Try WebGPU first, fall back to WASM
  const backends = [];
  if (ort.env?.webgpu) backends.push('webgpu');
  backends.push('wasm');
  let lastErr = null;
  for (const ep of backends) {
    try {
      encSession = await ort.InferenceSession.create(enc.bytes, {
        executionProviders: [ep],
        graphOptimizationLevel: 'all',
      });
      decSession = await ort.InferenceSession.create(dec.bytes, {
        executionProviders: [ep],
        graphOptimizationLevel: 'all',
        externalData: [{ path: 'sem_mask_temp_decoder_fp16.onnx.data', data: decDataBytes.buffer }],
      });

      // Also load the tiny latent_visual model (244 KB)
      try {
        const vgResp = await fetch(VIS_MODEL_URL, { cache: 'force-cache' });
        const vgBytes = new Uint8Array(await vgResp.arrayBuffer());
        const vdResp = await fetch(VIS_MODEL_DATA_URL, { cache: 'force-cache' });
        const vdBytes = new Uint8Array(await vdResp.arrayBuffer());
        visSession = await ort.InferenceSession.create(vgBytes, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
          externalData: [{ path: 'latent_visual.onnx.data', data: vdBytes.buffer }],
        });
      } catch (visErr) {
        visSession = null;
      }

      self.postMessage({ type: 'initDone', backend: ep });
      return;
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr || new Error('no ORT backend');
}

/**
 * Run the 2-stage decode pipeline.
 * input: ArrayBuffer containing [B, 64, T] channels-first float32 latent
 * shape: [B, 64, T]
 *
 * Returns the decoded audio Float32Array [B, 2, T*1920].
 */
async function decode(input, shape) {
  const [B, , T] = shape;
  const latent = new Float32Array(input);
  const perStem = LATENT_CHANNELS * T;

  // Step 1: Run encoder per-stem (B=1 fixed for encoder)
  const embeddings = new Float32Array(B * SEM_DIM);
  for (let b = 0; b < B; b++) {
    const stemLatent = latent.subarray(b * perStem, (b + 1) * perStem);
    const encInput = new ort.Tensor('float32', stemLatent, [1, LATENT_CHANNELS, T]);
    const encOut = await encSession.run({ latent: encInput });
    const embT = encOut.embedding || encOut[Object.keys(encOut)[0]];
    embeddings.set(embT.data, b * SEM_DIM);
  }

  // Step 2: Run decoder batched [B, 64, T] + [B, 128] + [B, 1025, T_m]
  //         → [B, 2, T*1920]. New input stft_mask defaults to zeros
  // here — the warmstart-trained model reduces to sem_v1 behavior when
  // the mask path is all zeros, so this is a safe no-op until the
  // caller wires real per-stem masks from v4-small. Use T_m=1 to
  // minimize the zero-tensor size; the compressor interpolates
  // internally to whatever time resolution it needs.
  const latInput = new ort.Tensor('float32', latent, shape);
  const embInput = new ort.Tensor('float32', embeddings, [B, SEM_DIM]);
  const N_FREQ = 1025;
  const zeroMask = new Float32Array(B * N_FREQ * 1);
  const maskInput = new ort.Tensor('float32', zeroMask, [B, N_FREQ, 1]);
  const decOut = await decSession.run({
    latent: latInput,
    sem_emb: embInput,
    stft_mask: maskInput,
  });
  const audioT = decOut.audio || decOut[Object.keys(decOut)[0]];
  // ORT-Web WebGPU: outputs stay on GPU unless explicitly copied.
  if (typeof audioT.getData === 'function') {
    return await audioT.getData();
  }
  return audioT.data;
}

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === 'init') {
    try {
      await init();
    } catch (err) {
      self.postMessage({
        type: 'initFailed',
        error: err?.message || err?.toString?.() || 'unknown',
      });
    }
    return;
  }

  if (msg.type === 'envelope') {
    const { id, input, T } = msg;
    try {
      if (!visSession) throw new Error('latent_visual not loaded');
      const flatTD = new Float32Array(input);
      const dt = new Float32Array(T * 64);
      for (let t = 0; t < T; t++) {
        for (let d = 0; d < 64; d++) {
          dt[d * T + t] = flatTD[t * 64 + d];
        }
      }
      const tensor = new ort.Tensor('float32', dt, [1, 64, T]);
      const out = await visSession.run({ latent: tensor });
      const envT = out.envelope || out[Object.keys(out)[0]];
      const envData = envT.data;
      self.postMessage(
        { type: 'envelopeDone', id, envelope: envData.buffer },
        [envData.buffer],
      );
    } catch (err) {
      self.postMessage({
        type: 'envelopeFailed',
        id,
        error: err?.message || err?.toString?.() || 'unknown',
      });
    }
    return;
  }

  if (msg.type === 'decode') {
    const { id, input, shape } = msg;
    try {
      const audioData = await decode(input, shape);
      self.postMessage(
        { type: 'decoded', id, audio: audioData.buffer },
        [audioData.buffer],
      );
    } catch (err) {
      self.postMessage({
        type: 'decodeFailed',
        id,
        error: err?.message || err?.toString?.() || 'unknown',
      });
    }
    return;
  }
};
