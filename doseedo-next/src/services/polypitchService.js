/**
 * polypitchService — bridge between doseedo's existing basic-pitch MIDI and
 * the polypitch WebGPU resample pipeline (ported from
 * ~/Downloads/doseedo-desktop/polypitch-browser under src/polypitch/).
 *
 * The upstream Pipeline did its own note detection via @spotify/basic-pitch;
 * we already have a WebGPU ONNX basic-pitch loaded in basicPitchOnnx.js and
 * the per-stem MIDI from latentPitch, so the adapted Pipeline here takes
 * `ingest(audio, notes)` instead.
 *
 * Public surface (all idempotent, singleton pipeline):
 *
 *   getPipeline(): Promise<Pipeline>
 *     Lazily boot the WebGPU pipeline. First call acquires the GPU device
 *     and compiles kernels. Subsequent calls return the same instance.
 *
 *   renderWithNewPitches({ audioUrl, notes, newPitches }): Promise<Blob>
 *     notes:       Note[] — the basic-pitch output (id, startSec, endSec,
 *                  pitchMidi, pitchCents, velocity, confidence)
 *     newPitches:  Map<noteId, newMidi> — target pitch for each note that
 *                  should move. Missing ids = unchanged.
 *     Returns a WAV Blob of the re-rendered audio.
 *
 *   notesFromMidiData(midiData, trackDurationSec): Note[]
 *     Converts the {notes: [...]} shape produced by basicPitchOnnx /
 *     latentPitch into the polypitch Note[] shape, minting stable ids.
 */

import { Pipeline } from '../polypitch/pipeline/Pipeline';
import { encodeWav } from '../polypitch/utils/wav';
import { logPipeline } from './pipelineStatus';

let _pipelinePromise = null;

// A/B flag for the learned mask UNet.
// - default (no query param, no localStorage key) → Hann4 classical fallback
// - `?maskUnet=step15000` or `localStorage.polypitchMask = 'step15000'`
//     → load /static/models/mask_unet_v1_step15000.onnx
// Any other truthy value is treated as a direct URL.
function resolveMaskUnetUrl() {
  if (typeof window === 'undefined') return null;
  const qs = new URLSearchParams(window.location.search).get('maskUnet');
  const ls = (() => { try { return window.localStorage.getItem('polypitchMask'); } catch { return null; } })();
  const v = qs ?? ls;
  if (!v || v === 'off' || v === 'null' || v === 'hann4') return null;
  if (v === 'step15000') return '/static/models/mask_unet_v1_step15000.onnx';
  return v; // treat as direct URL
}

export function getPipeline() {
  if (!_pipelinePromise) {
    _pipelinePromise = (async () => {
      const maskUrl = resolveMaskUnetUrl();
      logPipeline('polypitch', `booting WebGPU kernels… (mask=${maskUrl ? 'unet:' + maskUrl : 'hann4'})`);
      const pipeline = await Pipeline.init({
        maskUNetUrl: maskUrl,
      });
      logPipeline(
        'polypitch',
        `ready (mask=${maskUrl ? 'unet' : 'hann4'}, python-parity resample pitch-shift)`,
        'ok',
      );
      return pipeline;
    })().catch((err) => {
      _pipelinePromise = null; // retry on next call
      logPipeline('polypitch', `init failed: ${err?.message || err}`, 'error');
      throw err;
    });
  }
  return _pipelinePromise;
}

/**
 * Convert the {notes: [{note, time, duration, velocity}]} shape produced by
 * basicPitchOnnx / latentPitch into the polypitch Note[] shape.
 *
 * @param {{notes:Array<{note:number, time:number, duration:number, velocity:number}>}} midiData
 * @returns {Array<{id:string, startSec:number, endSec:number, pitchMidi:number, pitchCents:number, velocity:number, confidence:number}>}
 */
export function notesFromMidiData(midiData) {
  if (!midiData || !Array.isArray(midiData.notes)) return [];
  // IDs are deterministic on pitch+time so the same note produces the
  // same id whether it came through a windowed slice or the full track
  // array. polypitchChordSync computes newPitches from a ~2-note window
  // and then hands renderWithNewPitches the full ~200-note array — if
  // we index-prefixed the id, "n0-69-10.234" in newPitches wouldn't
  // match "n47-69-10.234" in the full array and every render silently
  // noop'd with "no pitch changes".
  return midiData.notes.map((n) => ({
    id: `${n.note | 0}@${n.time.toFixed(3)}`,
    startSec: n.time,
    endSec: n.time + Math.max(0.01, n.duration || 0.1),
    pitchMidi: n.note | 0,
    pitchCents: 0,
    velocity: Math.min(1, Math.max(0, (n.velocity ?? 100) / 127)),
    confidence: 1.0,
  }));
}

/**
 * Fetch audioUrl, decode through the browser's AudioContext, and return the
 * planar-stereo `AudioBuffer` shape polypitch expects. Uses
 * fetchAudioWithCache so we share the IndexedDB cache + in-flight dedupe
 * with the playback prewarm — without this, polypitch's raw fetch was
 * queueing behind the prewarm's repeated re-fetches of the same stem
 * URLs (4-stem chord edits silently hung at "loading audio…" because
 * the browser's per-origin connection limit pushed the polypitch fetch
 * to the back of the queue).
 */
async function loadAudioBuffer(audioUrl) {
  const { fetchAudioWithCache } = await import('./audioCacheService');
  const { blob } = await fetchAudioWithCache(audioUrl);
  const ab = await blob.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(ab.slice(0));
  } finally {
    try { ctx.close(); } catch (_) {}
  }
  const channels = Math.min(2, decoded.numberOfChannels);
  const frames = decoded.length;
  const samples = new Float32Array(frames * channels);
  for (let c = 0; c < channels; c++) {
    samples.set(decoded.getChannelData(c), c * frames);
  }
  return { samples, channels, sampleRate: decoded.sampleRate, frames };
}

/**
 * Render a new take of the track where each note in `newPitches` has been
 * pitch-shifted to its target MIDI via polypitch's resample renderer. Notes not
 * present in `newPitches` stay at their original pitch (they stay in the mix
 * because `includeUnedited=true`, the subtract-then-add path inside render).
 *
 * Returns a 16-bit WAV Blob ready to wire to a track's audioUrl via
 * URL.createObjectURL.
 */
export async function renderWithNewPitches({ audioUrl, notes, newPitches }) {
  if (!audioUrl) throw new Error('polypitch: audioUrl required');
  if (!Array.isArray(notes) || notes.length === 0) {
    throw new Error('polypitch: notes[] required — run basic-pitch first');
  }
  const edits = [];
  for (const note of notes) {
    const target = newPitches?.get?.(note.id);
    if (typeof target !== 'number' || target === note.pitchMidi) continue;
    edits.push({
      noteId: note.id,
      semitones: target - note.pitchMidi,
      cents: 0,
      gainDb: 0,
      muted: false,
    });
  }
  if (edits.length === 0) {
    logPipeline('polypitch', 'no pitch changes — skipping render', 'warn');
    return null;
  }

  // Show the actual semitone deltas so we can see whether a chord edit
  // produces audible shifts or just tiny ±1 semitone nudges (e.g. the
  // new chord shares most of its tones with the underlying audio).
  const deltaHist = edits.reduce((acc, e) => {
    acc[e.semitones] = (acc[e.semitones] || 0) + 1;
    return acc;
  }, {});
  const deltaSummary = Object.entries(deltaHist)
    .sort((a, b) => parseInt(a[0], 10) - parseInt(b[0], 10))
    .map(([s, n]) => `${s > 0 ? '+' : ''}${s}st×${n}`)
    .join(' ');
  logPipeline('polypitch', `deltas: ${deltaSummary}`);

  logPipeline('polypitch', `loading audio…`);
  const audio = await loadAudioBuffer(audioUrl);

  const pipeline = await getPipeline();
  logPipeline('polypitch', `ingesting ${notes.length} notes…`);
  await pipeline.ingest(audio, notes);

  logPipeline('polypitch', `rendering ${edits.length} pitch edits…`);
  const rendered = await pipeline.render(edits, /* includeUnedited */ true);

  const blob = encodeWav(rendered);
  logPipeline('polypitch', `rendered ${(blob.size / 1024).toFixed(0)} KB WAV`, 'ok');
  return blob;
}
