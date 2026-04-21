/**
 * polypitchService — bridge between doseedo's existing basic-pitch MIDI and
 * the polypitch WebGPU phase-vocoder pipeline (ported from
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

export function getPipeline() {
  if (!_pipelinePromise) {
    _pipelinePromise = (async () => {
      logPipeline('polypitch', 'booting WebGPU kernels…');
      const pipeline = await Pipeline.init({
        // maskUNetUrl null → classical Hann4 harmonic mask (no ONNX).
        // Wire a learned mask_unet.onnx URL here once it's trained.
        maskUNetUrl: null,
      });
      logPipeline('polypitch', 'ready (phase-vocoder pitch-shift)', 'ok');
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
  return midiData.notes.map((n, i) => ({
    id: `n${i}-${n.note}-${n.time.toFixed(3)}`,
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
 * planar-stereo `AudioBuffer` shape polypitch expects.
 */
async function loadAudioBuffer(audioUrl) {
  const res = await fetch(audioUrl);
  const ab = await res.arrayBuffer();
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
 * pitch-shifted to its target MIDI via polypitch's phase vocoder. Notes not
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
