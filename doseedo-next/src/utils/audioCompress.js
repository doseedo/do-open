/**
 * Browser-side audio compressor for upload size reduction.
 *
 * Cloud Run + Cloudflare cap upload bodies at ~32 MB and 100 MB respectively,
 * so any track longer than ~3 min @ 44.1k stereo PCM will get rejected as 413.
 * This helper decodes any input audio (Blob/File/URL → Web Audio buffer),
 * downsamples to a configurable sample rate, and re-encodes as 16-bit PCM
 * WAV. For 22050 Hz stereo, 1 min ≈ 5 MB. A 6-minute track ≈ 30 MB.
 *
 * Demucs upsamples internally and is robust to source rate changes, so the
 * separation quality loss from this is small (and was the only fix that
 * actually unblocks long uploads through Cloud Run).
 */

const DEFAULT_TARGET_SR = 22050;

async function blobToAudioBuffer(blob, ctx) {
  const arrayBuf = await blob.arrayBuffer();
  return ctx.decodeAudioData(arrayBuf);
}

function downsample(audioBuffer, targetSampleRate, ctx) {
  if (audioBuffer.sampleRate === targetSampleRate) return Promise.resolve(audioBuffer);
  const offlineCtx = new OfflineAudioContext(
    audioBuffer.numberOfChannels,
    Math.ceil(audioBuffer.duration * targetSampleRate),
    targetSampleRate,
  );
  const src = offlineCtx.createBufferSource();
  src.buffer = audioBuffer;
  src.connect(offlineCtx.destination);
  src.start(0);
  return offlineCtx.startRendering();
}

function encodeWav16(audioBuffer) {
  const numCh = audioBuffer.numberOfChannels;
  const sr = audioBuffer.sampleRate;
  const numFrames = audioBuffer.length;
  const bytesPerSample = 2;
  const dataSize = numFrames * numCh * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  function writeStr(off, s) { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); }

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);              // fmt chunk size
  view.setUint16(20, 1, true);               // PCM
  view.setUint16(22, numCh, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * numCh * bytesPerSample, true);
  view.setUint16(32, numCh * bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, dataSize, true);

  // Interleave channels and convert float32 → int16
  const channels = [];
  for (let c = 0; c < numCh; c++) channels.push(audioBuffer.getChannelData(c));
  let off = 44;
  for (let i = 0; i < numFrames; i++) {
    for (let c = 0; c < numCh; c++) {
      let s = Math.max(-1, Math.min(1, channels[c][i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      off += 2;
    }
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Compress an audio Blob/File for upload.
 * Returns a new Blob (audio/wav) downsampled to targetSr.
 * If the input is already smaller than maxBytes, returns it unchanged.
 *
 * @param {Blob|File} blob       — input audio
 * @param {object}    opts
 * @param {number}    opts.targetSr — sample rate (default 22050)
 * @param {number}    opts.maxBytes — skip compression if input is smaller (default 25 MB)
 */
export async function compressAudioForUpload(blob, opts = {}) {
  const targetSr = opts.targetSr || DEFAULT_TARGET_SR;
  const maxBytes = opts.maxBytes || 25 * 1024 * 1024;

  if (blob.size <= maxBytes) {
    return blob;
  }

  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  try {
    const decoded = await blobToAudioBuffer(blob, ctx);
    const down = await downsample(decoded, targetSr, ctx);
    const out = encodeWav16(down);
    console.log(
      `🎚️ compressAudioForUpload: ${(blob.size / 1024 / 1024).toFixed(1)}MB → ${(out.size / 1024 / 1024).toFixed(1)}MB (${decoded.sampleRate}Hz → ${targetSr}Hz, ${decoded.numberOfChannels}ch, ${decoded.duration.toFixed(1)}s)`,
    );
    return out;
  } finally {
    try { ctx.close(); } catch (_) {}
  }
}
