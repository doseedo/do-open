/**
 * sessionAudioAPI — talk to /api/sessions/{id}/audio.
 *
 * Default path: encode-to-Opus-128 then POST. Pro+ users can pass
 * `preserveLossless: true` to upload the original WAV/FLAC bytes
 * (the server enforces tier; we just don't transcode locally).
 */

import { encodeOpus128, inferAudioMime, shouldEncodeForTier } from './audioEncode';

const _AUTH_HEADERS = (clerkToken) =>
  clerkToken ? { Authorization: `Bearer ${clerkToken}` } : {};

/**
 * Upload a source audio File to a session.
 *
 * @param {object} args
 * @param {string} args.sessionId
 * @param {File}   args.file
 * @param {string|null} [args.trackId]
 * @param {string} [args.tier='free']    'free' | 'pro' | 'pro_plus'
 * @param {boolean} [args.preserveLossless=false]   pro+ only
 * @param {string|null} [args.shareToken]
 * @param {string|null} [args.clerkToken]
 * @param {(p:number)=>void} [args.onProgress]   0..1
 * @returns {Promise<{ id, blob_sha256, download_url, stored_mime, original_mime, is_lossless, bytes }>}
 */
export async function uploadSessionAudio({
  sessionId,
  file,
  trackId = null,
  tier = 'free',
  preserveLossless = false,
  shareToken = null,
  clerkToken = null,
  onProgress = null,
}) {
  const originalMime = inferAudioMime(file);
  let blob = file;
  let storedMime = originalMime;
  let filename = file.name;

  if (shouldEncodeForTier(tier, preserveLossless)) {
    onProgress?.(0.05);
    const enc = await encodeOpus128(file);
    blob = enc.blob;
    storedMime = 'audio/ogg';            // server normalises to audio/ogg
    if (!filename.toLowerCase().endsWith('.opus')) {
      filename = filename.replace(/\.[^.]+$/, '') + '.opus';
    }
    onProgress?.(0.45);
  }

  const fd = new FormData();
  fd.append('file', new File([blob], filename, { type: storedMime }));
  if (trackId) fd.append('track_id', trackId);
  if (originalMime) fd.append('original_mime', originalMime);

  const url = `/api/sessions/${encodeURIComponent(sessionId)}/audio${
    shareToken ? `?share_token=${encodeURIComponent(shareToken)}` : ''
  }`;

  const res = await fetch(url, {
    method: 'POST',
    body: fd,
    headers: { ..._AUTH_HEADERS(clerkToken) },
    credentials: 'include',
  });
  onProgress?.(1.0);

  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try { const j = await res.json(); msg = j.detail || msg; } catch {}
    if (res.status === 402) throw new Error(`UPGRADE_REQUIRED: ${msg}`);
    throw new Error(msg);
  }
  return res.json();
}

export async function listSessionAudio({ sessionId, shareToken = null, clerkToken = null }) {
  const url = `/api/sessions/${encodeURIComponent(sessionId)}/audio${
    shareToken ? `?share_token=${encodeURIComponent(shareToken)}` : ''
  }`;
  const res = await fetch(url, {
    headers: { ..._AUTH_HEADERS(clerkToken) },
    credentials: 'include',
  });
  if (!res.ok) throw new Error(`list audio: HTTP ${res.status}`);
  return res.json();
}

/**
 * Attach a non-audio companion blob (latent or midi) to an existing
 * SessionAudio row. Idempotent — re-uploading the same bytes hits the
 * server's BlobIndex dedup. The studio fires this from inside the
 * background analyze pipeline once the encoder finishes.
 *
 * @param {object} args
 * @param {'latent'|'midi'} args.kind
 * @param {ArrayBuffer|Uint8Array|Blob} args.payload   raw bytes
 * @param {object} [args.meta]   {n_frames, fps, vae_version} for latent;
 *                               {n_notes} for midi
 */
export async function attachTrackAsset({
  sessionId,
  audioId,
  kind,
  payload,
  meta = {},
  shareToken = null,
  clerkToken = null,
}) {
  if (kind !== 'latent' && kind !== 'midi') {
    throw new Error(`attachTrackAsset: kind must be 'latent' or 'midi', got '${kind}'`);
  }
  const blob = payload instanceof Blob
    ? payload
    : new Blob([payload], { type: kind === 'midi' ? 'audio/midi' : 'application/octet-stream' });
  const filename = kind === 'latent' ? 'latent.doae' : 'transcription.mid';

  const fd = new FormData();
  fd.append('file', new File([blob], filename, { type: blob.type }));
  fd.append('kind', kind);
  if (meta.n_frames != null) fd.append('n_frames', String(meta.n_frames));
  if (meta.fps != null) fd.append('fps', String(meta.fps));
  if (meta.vae_version) fd.append('vae_version', meta.vae_version);
  if (meta.n_notes != null) fd.append('n_notes', String(meta.n_notes));

  const url = `/api/sessions/${encodeURIComponent(sessionId)}/audio/${encodeURIComponent(audioId)}/asset${
    shareToken ? `?share_token=${encodeURIComponent(shareToken)}` : ''
  }`;
  const res = await fetch(url, {
    method: 'PATCH',
    body: fd,
    headers: { ..._AUTH_HEADERS(clerkToken) },
    credentials: 'include',
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try { const j = await res.json(); msg = j.detail || msg; } catch {}
    throw new Error(msg);
  }
  return res.json();
}

export async function deleteSessionAudio({ sessionId, audioId, shareToken = null, clerkToken = null }) {
  const url = `/api/sessions/${encodeURIComponent(sessionId)}/audio/${encodeURIComponent(audioId)}${
    shareToken ? `?share_token=${encodeURIComponent(shareToken)}` : ''
  }`;
  const res = await fetch(url, {
    method: 'DELETE',
    headers: { ..._AUTH_HEADERS(clerkToken) },
    credentials: 'include',
  });
  if (!res.ok && res.status !== 204) throw new Error(`delete audio: HTTP ${res.status}`);
}
