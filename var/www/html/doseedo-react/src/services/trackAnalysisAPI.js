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

/**
 * Rich per-bar tempo/meter analysis. Returns:
 *   {
 *     bpm, beatsPerBar, meterDenominator, grouping,
 *     downbeat_offset,
 *     beat_map: [{t, pos}, ...],
 *     tempoMap: [{bar, t, bpm, meter: [n,d], grouping?}, ...],
 *     duration,
 *   }
 *
 * Populated by the backend /api/analyze-rhythm endpoint using beat_this +
 * librosa. Supports in-song tempo and meter changes — each entry in
 * tempoMap represents a run of bars sharing local tempo+meter.
 *
 * Independent of /api/detect-chords (which is currently gated off for
 * chord extraction). Safe to call on every audio upload.
 */
export async function analyzeRhythm(audioFile) {
  const compressed = await compressAudioForUpload(audioFile);
  const fd = new FormData();
  fd.append('audioFile', compressed, audioFile.name || 'audio.wav');
  const r = await fetch('/api/analyze-rhythm', { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`analyze-rhythm HTTP ${r.status}`);
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
  // Clerk Bearer token — same pattern as separateStemsAuto. Without it
  // the Modal generation gate (_GATED_ROUTES in modal_stemphonic.py) 401s
  // with "Authentication required for generation".
  const token = (typeof window !== 'undefined' && typeof window.__clerkGetToken === 'function'
      ? await window.__clerkGetToken().catch(() => null) : null)
    || (typeof localStorage !== 'undefined' ? localStorage.getItem('token') : null);
  const authHeaders = token ? { 'Authorization': `Bearer ${token}` } : {};
  const r = await fetch('/api/repaint-meter', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders },
    credentials: 'include',
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
  if (!r.ok) {
    if (r.status === 429) {
      const { handleMaybeCreditGate } = await import('./outOfCreditsSignal');
      await handleMaybeCreditGate(r);
    }
    throw new Error(`repaint-meter HTTP ${r.status}`);
  }
  return r.json();
}

export async function separateStemsAuto(audioFile, opts = {}) {
  // Calls /separate-stems (background task), polls until stems are ready
  // (`status === 'stems_ready'`), returns immediately so the DAW can load
  // audio + placeholder (latent_pitch) MIDI, then continues polling in the
  // background for per-stem BasicPitch MIDI as each finishes and fires
  // `opts.onMidiReady({ taskId, midi_urls })` whenever new URLs arrive.
  //
  //   const res = await separateStemsAuto(file, {
  //     onMidiReady: ({ midi_urls }) => { ...replace per-stem MIDI... },
  //   });
  //
  // Returns { task_id, stems, stem_latents, mask_bundle } (no blocking
  // on BasicPitch). Old callers that just awaited the result still work —
  // they'll get stems_ready payload + the final `status === 'completed'`
  // is still reachable via polling status if they need it.
  // Auth: always ask Clerk for a live token via window.__clerkGetToken
  // (installed by AppShell.tsx's ClerkTokenBridge). NEVER read the cached
  // localStorage['clerk_token'] snapshot — on a dev→prod Clerk cutover the
  // snapshot can be a stale dev-instance JWT whose signing kid isn't in
  // prod's JWKS, which auth-service rejects with 401. getToken() itself
  // returns null under the prod instance if the user has no valid session,
  // which is the correct signal to surface instead of a bogus token.
  const token = (typeof window !== 'undefined' && typeof window.__clerkGetToken === 'function'
      ? await window.__clerkGetToken().catch(() => null) : null)
    || localStorage.getItem('token');
  const authHeaders = token ? { 'Authorization': `Bearer ${token}` } : {};
  const compressed = await compressAudioForUpload(audioFile);
  const fd = new FormData();
  fd.append('audioFile', compressed, audioFile.name || 'audio.wav');
  const r = await fetch('/separate-stems', { method: 'POST', body: fd, headers: authHeaders, credentials: 'include' });
  if (!r.ok) {
    if (r.status === 429) {
      const { handleMaybeCreditGate } = await import('./outOfCreditsSignal');
      await handleMaybeCreditGate(r);
    }
    throw new Error(`separate-stems HTTP ${r.status}`);
  }
  const start = await r.json();
  const taskId = start.task_id;
  let result = start;

  // Phase 1: wait for stems.
  if (result.status === 'processing' && taskId) {
    for (let i = 0; i < 90; i++) {
      await new Promise((res) => setTimeout(res, 2000));
      const sr = await fetch(`/separate-stems/status/${taskId}?t=${Date.now()}`, {
        cache: 'no-store',
        headers: authHeaders,
      });
      if (!sr.ok) continue;
      result = await sr.json();
      if (result.status === 'stems_ready' || result.status === 'completed') {
        result.task_id = taskId;
        break;
      }
      if (result.status === 'failed') throw new Error(result.error || 'separation failed');
    }
  }

  // Phase 2: background-poll for BasicPitch MIDI + drum-teacher substems
  // (non-blocking). Both arrive after `stems_ready` on their own threads;
  // we surface them via callbacks and also accumulate into `result` so a
  // caller that passed no callback can still read them via the resolved
  // promise once polling has caught up.
  const { onMidiReady, onDrumTeacher, onLyrics } = opts;
  const wantsBackground = (onMidiReady || onDrumTeacher || onLyrics)
    && taskId && (result.status === 'stems_ready' || result.status === 'completed');
  if (wantsBackground) {
    (async () => {
      const seenMidi = new Set(Object.keys(result.midi_urls || {}));
      let drumDelivered = !!result.drum_substem_urls && Object.keys(result.drum_substem_urls).length > 0;
      let lyricsDelivered = !!result.vocals_lyrics;
      if (seenMidi.size && onMidiReady) onMidiReady({ task_id: taskId, midi_urls: { ...result.midi_urls } });
      if (drumDelivered && onDrumTeacher) {
        onDrumTeacher({
          task_id: taskId,
          drum_substem_urls: { ...result.drum_substem_urls },
          drum_substem_onsets: { ...(result.drum_substem_onsets || {}) },
          drum_substem_onset_strengths: { ...(result.drum_substem_onset_strengths || {}) },
        });
      }
      if (lyricsDelivered && onLyrics) {
        onLyrics({
          task_id: taskId,
          vocals_lyrics: result.vocals_lyrics,
          vocals_lyrics_language: result.vocals_lyrics_language,
        });
      }
      for (let i = 0; i < 180; i++) {    // up to ~6 minutes of polling
        await new Promise((res) => setTimeout(res, 2000));
        let poll;
        try {
          const sr = await fetch(`/separate-stems/status/${taskId}?t=${Date.now()}`, {
            cache: 'no-store', headers: authHeaders,
          });
          if (!sr.ok) continue;
          poll = await sr.json();
        } catch (_) { continue; }

        // BasicPitch MIDI — incremental, fire per new stem.
        if (onMidiReady) {
          const urls = poll?.midi_urls || {};
          const added = {};
          for (const [stem, url] of Object.entries(urls)) {
            if (!seenMidi.has(stem)) { seenMidi.add(stem); added[stem] = url; }
          }
          if (Object.keys(added).length) {
            onMidiReady({ task_id: taskId, midi_urls: added });
          }
        }

        // Drum teacher (MDX23C-DrumSep) — kick/snare/hh/toms/ride/crash WAVs
        // + librosa onsets per substem. Arrives in one shot once the
        // separator finishes; fire the callback exactly once.
        if (!drumDelivered && poll?.drum_substem_urls && Object.keys(poll.drum_substem_urls).length > 0) {
          drumDelivered = true;
          result.drum_substem_urls = { ...poll.drum_substem_urls };
          result.drum_substem_onsets = { ...(poll.drum_substem_onsets || {}) };
          result.drum_substem_onset_strengths = { ...(poll.drum_substem_onset_strengths || {}) };
          if (onDrumTeacher) {
            onDrumTeacher({
              task_id: taskId,
              drum_substem_urls: result.drum_substem_urls,
              drum_substem_onsets: result.drum_substem_onsets,
              drum_substem_onset_strengths: result.drum_substem_onset_strengths,
            });
          }
        }

        // Whisper lyrics — fire once when present.
        if (!lyricsDelivered && poll?.vocals_lyrics) {
          lyricsDelivered = true;
          result.vocals_lyrics = poll.vocals_lyrics;
          result.vocals_lyrics_language = poll.vocals_lyrics_language;
          if (onLyrics) {
            onLyrics({
              task_id: taskId,
              vocals_lyrics: result.vocals_lyrics,
              vocals_lyrics_language: result.vocals_lyrics_language,
            });
          }
        }

        if (poll?.status === 'completed' || poll?.status === 'failed') break;
      }
    })().catch(() => {});   // swallow background errors — main path already returned
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
