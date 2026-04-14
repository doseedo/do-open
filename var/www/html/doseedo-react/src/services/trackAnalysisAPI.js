import { compressAudioForUpload } from '../utils/audioCompress';
import { initLatentEncoder, encodeToLatent, audioFileToStereo48k } from './latentEncoder';
import { uploadLatent } from './latentDemucs';

/**
 * trackAnalysisAPI — per-upload analysis pipeline
 *
 * Every audio file dropped on the timeline goes through:
 *   1. /api/extract-midi      → basic-pitch transcription, returns midi_url
 *   2. /api/classify-instrument → PANNs CNN14 → track type (guitar/bass/...)
 *
 * Both run in parallel. Results are stored on track.metadata so the
 * chord-aware regen pipeline can use them later.
 */

export async function extractMidi(audioFile) {
  const compressed = await compressAudioForUpload(audioFile);
  const fd = new FormData();
  fd.append('audioFile', compressed, audioFile.name || 'audio.wav');
  const r = await fetch('/api/extract-midi', { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`extract-midi HTTP ${r.status}`);
  const data = await r.json();
  if (data.error) throw new Error(data.error);
  return data; // { midi_url, n_notes, duration }
}

export async function classifyInstrument(audioFile) {
  const compressed = await compressAudioForUpload(audioFile);
  const fd = new FormData();
  fd.append('audioFile', compressed, audioFile.name || 'audio.wav');
  const r = await fetch('/api/classify-instrument', { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`classify-instrument HTTP ${r.status}`);
  const data = await r.json();
  if (data.error) throw new Error(data.error);
  return data; // { type, label, score, top5 }
}

export async function encodeAudioLatent(audioFile) {
  // Client-side only — raw audio never leaves the browser for latent extraction.
  // Throws (fail-closed) if WebGPU/WASM encoder is unavailable.
  await initLatentEncoder();
  const { flat, numFrames } = await audioFileToStereo48k(audioFile);
  const flatTD = await encodeToLatent(flat, numFrames);
  const T = flatTD.length / 64;
  const { latent_id } = await uploadLatent(flatTD, T);
  return {
    latent_id,
    latent_url: `/api/latents/download/${latent_id}.pt`,
    n_frames: T,
    duration: T / 25,
  };
}

export async function detectChordsAndTempo(audioFile) {
  // Returns { bpm, beats_per_bar, downbeat_offset, mode, chords }
  const compressed = await compressAudioForUpload(audioFile);
  const fd = new FormData();
  fd.append('audioFile', compressed, audioFile.name || 'audio.wav');
  fd.append('mode', 'master');
  const r = await fetch('/api/detect-chords', { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`detect-chords HTTP ${r.status}`);
  const data = await r.json();
  if (data.error) throw new Error(data.error);
  return data;
}

export async function encodeLatentsBulk(urls) {
  // Client-side only — fetches each stem WAV, encodes in-browser, uploads latent.
  // Throws (fail-closed) if encoder is unavailable. Per-URL errors are logged and skipped.
  await initLatentEncoder();
  const results = [];
  for (const audioUrl of urls) {
    try {
      const resp = await fetch(audioUrl);
      if (!resp.ok) { console.error(`encodeLatentsBulk: fetch ${audioUrl} → ${resp.status}`); continue; }
      const blob = await resp.blob();
      const { flat, numFrames } = await audioFileToStereo48k(blob);
      const flatTD = await encodeToLatent(flat, numFrames);
      const T = flatTD.length / 64;
      const { latent_id } = await uploadLatent(flatTD, T);
      results.push({ url: audioUrl, latent_id, latent_url: `/api/latents/download/${latent_id}.pt`, n_frames: T });
    } catch (err) {
      console.error(`encodeLatentsBulk: ${audioUrl} failed:`, err);
    }
  }
  return { results };
}

export async function repaintMeter({ stems, srcMeter, tgtMeter, srcBpm, tgtBpm, coverNoise = 0.55, prompt, downbeatOffset = 0 }) {
  const r = await fetch('/api/repaint-meter', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      stems,
      src_meter: srcMeter,
      tgt_meter: tgtMeter,
      src_bpm: srcBpm,
      tgt_bpm: tgtBpm,
      cover_noise: coverNoise,
      prompt,
      downbeat_offset: downbeatOffset,
    }),
  });
  if (!r.ok) throw new Error(`repaint-meter HTTP ${r.status}`);
  return r.json();
}

export async function separateStemsAuto(audioFile) {
  // Calls /separate-stems (background task), polls until done, returns
  // { task_id, stems: { drums, bass, other, vocals, guitar, piano } }
  const compressed = await compressAudioForUpload(audioFile);
  const fd = new FormData();
  fd.append('audioFile', compressed, audioFile.name || 'audio.wav');
  const r = await fetch('/separate-stems', { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`separate-stems HTTP ${r.status}`);
  const start = await r.json();
  // Capture the task_id ONCE — the status endpoint doesn't echo it back,
  // so re-reading from `result` after the first poll would give undefined
  // and we'd start polling /status/undefined.
  const taskId = start.task_id;
  let result = start;
  if (result.status === 'processing' && taskId) {
    for (let i = 0; i < 90; i++) {
      await new Promise((res) => setTimeout(res, 2000));
      // Cache-bust the GET so Cloudflare/CDN doesn't serve a stale
      // "processing" response for the entire poll loop.
      const sr = await fetch(`/separate-stems/status/${taskId}?t=${Date.now()}`, {
        cache: 'no-store',
      });
      if (!sr.ok) continue;
      result = await sr.json();
      if (result.status === 'completed') {
        // Status payload doesn't echo task_id; merge it back so callers
        // (and downstream metadata) can reference the task.
        result.task_id = taskId;
        break;
      }
      if (result.status === 'failed') throw new Error(result.error || 'separation failed');
    }
  }
  return result;
}

export async function analyzeAudio(audioFile) {
  // 3 parallel calls: basic-pitch MIDI extraction, PANNs classifier,
  // and VAE latent encode. All cached on the track for download/repaint.
  const [midi, cls, latent] = await Promise.allSettled([
    extractMidi(audioFile),
    classifyInstrument(audioFile),
    encodeAudioLatent(audioFile),
  ]);
  return {
    midi: midi.status === 'fulfilled' ? midi.value : null,
    classification: cls.status === 'fulfilled' ? cls.value : null,
    latent: latent.status === 'fulfilled' ? latent.value : null,
    midiError: midi.status === 'rejected' ? midi.reason?.message : null,
    classifyError: cls.status === 'rejected' ? cls.reason?.message : null,
    latentError: latent.status === 'rejected' ? latent.reason?.message : null,
  };
}

/**
 * Regenerate a single stem to fit a chord change.
 * Returns { task_id } — poll /api/generate-stemphonic/task/<task_id>.
 */
export async function regenStemForChord({
  audioFile, midiFile, role, oldChord, newChord,
  regionStart = 0, regionEnd = null, coverNoise = 0.7, prompt = null, duration = null,
}) {
  const compressed = await compressAudioForUpload(audioFile);
  const fd = new FormData();
  fd.append('audioFile', compressed, audioFile.name || 'audio.wav');
  if (midiFile) fd.append('midiFile', midiFile);
  fd.append('role', role || 'harmony');
  fd.append('old_chord', oldChord || '');
  fd.append('new_chord', newChord || '');
  fd.append('region_start', String(regionStart));
  if (regionEnd != null) fd.append('region_end', String(regionEnd));
  fd.append('cover_noise', String(coverNoise));
  if (prompt) fd.append('prompt', prompt);
  if (duration != null) fd.append('duration', String(duration));
  const r = await fetch('/api/regen-stem-for-chord', { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`regen-stem-for-chord HTTP ${r.status}`);
  return r.json();
}

// AudioSet → FontAwesome icon mapping
export const TRACK_TYPE_ICONS = {
  guitar:  'fa-guitar',
  bass:    'fa-guitar',
  piano:   'fa-music',
  drums:   'fa-drum',
  vocals:  'fa-microphone',
  synth:   'fa-wave-square',
  strings: 'fa-music',
  brass:   'fa-music',
  winds:   'fa-music',
  other:   'fa-music',
};

export function iconForType(type) {
  return TRACK_TYPE_ICONS[type] || TRACK_TYPE_ICONS.other;
}
