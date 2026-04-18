/**
 * Decode a server-side STFT mask bundle into per-stem audio URLs.
 *
 * Server (see stemphonic_server.py `_compute_and_save_mask_bundle`)
 * writes a 128-byte header + uint8 magnitude masks, 1 per stem, mono.
 * Client STFTs the master audio once, multiplies each stem's mask into
 * the complex master STFT, iSTFTs back to time-domain stereo, and
 * encodes a WAV blob URL. The master's phase is reused for every stem —
 * perceptually clean, and the wire payload (~1 mask bundle) is ~4× smaller
 * than 6 WAV downloads for a 30 s clip.
 *
 * Trade-offs vs. WAV download:
 *   + Bandwidth: ~32 MB (6 × 5.3 MB WAV) → ~8 MB (1 mask bundle).
 *   + One HTTP request instead of six.
 *   − Browser CPU: ~500 ms of STFT/iSTFT per 30 s clip.
 *   − Requires the client to have decoded the master AudioBuffer already
 *     (it does — useWaveform's audioBufferCache stashes it on drop).
 *
 * Usage:
 *   const { stems } = await decodeStemMasks({
 *     bundleUrl, masterAudioBuffer, headers
 *   });
 *   // stems = [{ stemName, wavUrl }, ...]
 */

const HEADER_SIZE = 128;

// ── FFT (radix-2 Cooley-Tukey, complex in/out, in-place) ──────────────
function _fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k], uIm = im[i + k];
        const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k]             = uRe + vRe; im[i + k]             = uIm + vIm;
        re[i + k + len / 2]   = uRe - vRe; im[i + k + len / 2]   = uIm - vIm;
        const nr = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nr;
      }
    }
  }
}

function _ifft(re, im) {
  // Conjugate, FFT, conjugate, divide by n.
  const n = re.length;
  for (let i = 0; i < n; i++) im[i] = -im[i];
  _fft(re, im);
  const inv = 1 / n;
  for (let i = 0; i < n; i++) { re[i] *= inv; im[i] = -im[i] * inv; }
}

function _hann(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));
  return w;
}

// ── STFT / iSTFT (centered, reflect-padded — matches torch.stft default) ─
/** Returns { re, im }, each Float32Array of length nFrames * nBins. */
function _stft(mono, nFft, hop, window, nFrames, nBins) {
  const pad = nFft >> 1;
  const padded = new Float32Array(mono.length + 2 * pad);
  padded.set(mono, pad);
  // reflect padding (matches pad_mode='reflect')
  for (let i = 0; i < pad; i++) {
    padded[pad - 1 - i] = mono[i + 1] || 0;
    padded[pad + mono.length + i] = mono[mono.length - 2 - i] || 0;
  }
  const re = new Float32Array(nFft);
  const im = new Float32Array(nFft);
  const outRe = new Float32Array(nFrames * nBins);
  const outIm = new Float32Array(nFrames * nBins);
  for (let t = 0; t < nFrames; t++) {
    const off = t * hop;
    for (let k = 0; k < nFft; k++) {
      re[k] = padded[off + k] * window[k];
      im[k] = 0;
    }
    _fft(re, im);
    const base = t * nBins;
    for (let k = 0; k < nBins; k++) {
      outRe[base + k] = re[k];
      outIm[base + k] = im[k];
    }
  }
  return { re: outRe, im: outIm };
}

/** Inverse of _stft with matching center/window. */
function _istft(stftRe, stftIm, nFft, hop, window, nFrames, nBins, outLen) {
  const pad = nFft >> 1;
  const paddedLen = outLen + 2 * pad;
  const out = new Float32Array(paddedLen);
  const winSum = new Float32Array(paddedLen);

  const re = new Float32Array(nFft);
  const im = new Float32Array(nFft);
  for (let t = 0; t < nFrames; t++) {
    const base = t * nBins;
    // Reconstruct full spectrum from one-sided (hermitian symmetry).
    for (let k = 0; k < nBins; k++) {
      re[k] = stftRe[base + k];
      im[k] = stftIm[base + k];
    }
    for (let k = 1; k < nBins - 1; k++) {
      re[nFft - k] = stftRe[base + k];
      im[nFft - k] = -stftIm[base + k];
    }
    _ifft(re, im);
    const off = t * hop;
    for (let k = 0; k < nFft; k++) {
      out[off + k] += re[k] * window[k];
      winSum[off + k] += window[k] * window[k];
    }
  }
  // Normalize by overlap-add window-sum
  const trimmed = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const ws = winSum[pad + i];
    trimmed[i] = ws > 1e-8 ? out[pad + i] / ws : 0;
  }
  return trimmed;
}

// ── WAV writer (16-bit PCM stereo) ─────────────────────────────────────
function _stereoToWavBlobUrl(L, R, sampleRate) {
  const N = Math.min(L.length, R.length);
  const byteLen = 44 + N * 2 * 2;
  const ab = new ArrayBuffer(byteLen);
  const view = new DataView(ab);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, 'RIFF');
  view.setUint32(4, byteLen - 8, true);
  writeStr(8, 'WAVE'); writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 2, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 4, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, N * 4, true);
  let off = 44;
  for (let i = 0; i < N; i++) {
    let l = Math.max(-1, Math.min(1, L[i]));
    let r = Math.max(-1, Math.min(1, R[i]));
    view.setInt16(off,     (l * 0x7fff) | 0, true); off += 2;
    view.setInt16(off,     (r * 0x7fff) | 0, true); off += 2;
  }
  return URL.createObjectURL(new Blob([ab], { type: 'audio/wav' }));
}

// ── Bundle header parser ──────────────────────────────────────────────
function _parseHeader(ab) {
  const dv = new DataView(ab);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'DMSK') throw new Error(`bad mask bundle magic "${magic}"`);
  const version = dv.getUint8(4);
  if (version !== 1) throw new Error(`unsupported mask bundle version ${version}`);
  const nStems   = dv.getUint8(5);
  const nChans   = dv.getUint8(6);
  const dtype    = dv.getUint8(7);
  const nFft     = dv.getUint16(8, true);
  const hop      = dv.getUint16(10, true);
  const sr       = dv.getUint32(12, true);
  const nBins    = dv.getUint16(16, true);
  const nFrames  = dv.getUint32(18, true);
  const nSamples = dv.getUint32(22, true);
  if (dtype !== 0) throw new Error(`unsupported mask dtype ${dtype}`);
  if (nChans !== 1) throw new Error(`only mono masks supported (got ${nChans})`);
  const stemNames = [];
  for (let i = 0; i < nStems; i++) {
    const off = 32 + i * 16;
    let name = '';
    for (let k = 0; k < 16; k++) {
      const c = dv.getUint8(off + k);
      if (c === 0) break;
      name += String.fromCharCode(c);
    }
    stemNames.push(name);
  }
  return { nStems, nChans, nFft, hop, sr, nBins, nFrames, nSamples, stemNames };
}

// ── main ──────────────────────────────────────────────────────────────
export async function decodeStemMasks({ bundleUrl, masterAudioBuffer, headers = {}, signal }) {
  const tFetch0 = performance.now();
  const resp = await fetch(bundleUrl, { headers, signal, cache: 'no-store' });
  if (!resp.ok) throw new Error(`mask bundle fetch HTTP ${resp.status}`);
  const ab = await resp.arrayBuffer();
  const tFetch = performance.now() - tFetch0;

  const meta = _parseHeader(ab);
  const payload = new Uint8Array(ab, HEADER_SIZE);
  const expectedPayload = meta.nStems * meta.nBins * meta.nFrames;
  if (payload.length !== expectedPayload) {
    throw new Error(`mask payload size ${payload.length} != expected ${expectedPayload}`);
  }

  const masterSr = masterAudioBuffer.sampleRate;
  if (masterSr !== meta.sr) {
    // We don't resample — the master should already be at meta.sr (48 kHz).
    // If not, bail and let the caller fall back to WAV URLs.
    throw new Error(`master sr ${masterSr} != bundle sr ${meta.sr} — no client-side resample`);
  }
  const nFft = meta.nFft, hop = meta.hop, nBins = meta.nBins, nFrames = meta.nFrames;
  const window = _hann(nFft);

  // STFT master ONCE per channel — reused for every stem.
  const tStft0 = performance.now();
  const nCh = masterAudioBuffer.numberOfChannels;
  const chL = masterAudioBuffer.getChannelData(0);
  const chR = nCh > 1 ? masterAudioBuffer.getChannelData(1) : chL;
  const stftL = _stft(chL, nFft, hop, window, nFrames, nBins);
  const stftR = _stft(chR, nFft, hop, window, nFrames, nBins);
  const tStft = performance.now() - tStft0;

  const outLen = Math.min(meta.nSamples, chL.length);
  const stems = [];
  const tDecode0 = performance.now();
  // Scratch buffers reused per stem
  const maskedReL = new Float32Array(nFrames * nBins);
  const maskedImL = new Float32Array(nFrames * nBins);
  const maskedReR = new Float32Array(nFrames * nBins);
  const maskedImR = new Float32Array(nFrames * nBins);
  for (let s = 0; s < meta.nStems; s++) {
    const maskBase = s * nBins * nFrames;
    // Apply mask per (bin, frame): mask stored [stem][bin][frame] row-major,
    // which is maskBase + bin*nFrames + frame. STFT is stored [frame][bin]
    // row-major: base = frame*nBins + bin.
    for (let t = 0; t < nFrames; t++) {
      const stBase = t * nBins;
      for (let k = 0; k < nBins; k++) {
        const m = payload[maskBase + k * nFrames + t] / 255;
        maskedReL[stBase + k] = stftL.re[stBase + k] * m;
        maskedImL[stBase + k] = stftL.im[stBase + k] * m;
        maskedReR[stBase + k] = stftR.re[stBase + k] * m;
        maskedImR[stBase + k] = stftR.im[stBase + k] * m;
      }
    }
    const stemL = _istft(maskedReL, maskedImL, nFft, hop, window, nFrames, nBins, outLen);
    const stemR = _istft(maskedReR, maskedImR, nFft, hop, window, nFrames, nBins, outLen);
    const wavUrl = _stereoToWavBlobUrl(stemL, stemR, meta.sr);
    stems.push({ stemName: meta.stemNames[s], wavUrl });
  }
  const tDecode = performance.now() - tDecode0;

  console.log(
    `[stemMasks] ${meta.nStems} stems from ${(ab.byteLength / 1e6).toFixed(2)}MB bundle — ` +
    `fetch ${tFetch.toFixed(0)}ms · stft ${tStft.toFixed(0)}ms · decode ${tDecode.toFixed(0)}ms`
  );
  return { stems, meta, bytes: ab.byteLength, tFetch, tStft, tDecode };
}
