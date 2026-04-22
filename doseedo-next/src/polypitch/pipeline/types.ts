// Shared type contract across pipeline stages, model wrappers, workers, and UI.
// Every public boundary in polypitch-browser should import from this file.

/**
 * Mono or stereo PCM audio. Samples are planar per-channel in row-major order,
 * i.e. for stereo the buffer is [left..., right...]. Length === frames * channels.
 * Non-interleaved makes WGSL/ONNX ingestion trivial and stays out of the way of
 * Web Audio API ingestion, which exposes planar `getChannelData(i)` anyway.
 */
export interface AudioBuffer {
  samples: Float32Array;
  channels: 1 | 2;
  sampleRate: number; // canonical pipeline SR is 44100 for Basic Pitch, 48000 for our mask+DDSP
  frames: number;     // samples.length === frames * channels
}

/**
 * A single extractable note.
 * `id` is a stable string so UI can track per-note state across re-analyses.
 */
export interface Note {
  id: string;
  startSec: number;
  endSec: number;
  startFrame?: number;            // detector-native frame boundary, if available
  endFrame?: number;              // detector-native frame boundary, if available
  pitchMidi: number;              // rounded MIDI pitch (integer)
  pitchCents: number;             // cents offset from pitchMidi center (-50 .. +50)
  pitchTrack?: Float32Array;      // sub-sample pitch contour in MIDI, 100 Hz grid
  energyCurve?: Float32Array;     // detector-native per-frame energy curve
  velocity: number;               // 0..1
  confidence: number;             // 0..1 from detector head
  instrumentClass?: string;       // "piano", "strings", "voice", ... or undefined
  // Populated by extractor when the user asks for the isolated audio:
  extractedAudio?: AudioBuffer;
}

/**
 * Output of the note-identification stage (Basic Pitch + postprocess).
 * Does NOT include extracted audio — that's computed lazily on demand.
 */
export interface AnalysisResult {
  notes: Note[];
  durationSec: number;
  sampleRate: number;
}

/**
 * Per-note edit request from the UI.
 * `semitones`=0 and `gainDb`=0 means "extract without modification".
 */
export interface NoteEdit {
  noteId: string;
  semitones: number;
  cents: number;     // fine pitch, -100..+100; added on top of `semitones`
  gainDb: number;
  muted: boolean;
}

/**
 * Runtime status signalled over the worker channel so the UI can render
 * progress bars / disabled states.
 */
export type PipelineStage =
  | "idle"
  | "loading-models"
  | "decoding-audio"
  | "analyzing"
  | "extracting"
  | "synthesizing"
  | "ready"
  | "error";

export interface PipelineStatus {
  stage: PipelineStage;
  progress: number;       // 0..1
  message?: string;
  error?: string;
}

/**
 * Worker RPC message envelopes. Single-request/response, correlated by `id`.
 */
export type WorkerRequest =
  | { kind: "init"; id: string; modelBaseUrl: string }
  | { kind: "analyze"; id: string; audio: TransferableAudio }
  | { kind: "extract"; id: string; noteIds: string[] }
  | {
      kind: "render";
      id: string;
      edits: NoteEdit[];
      includeUnedited: boolean;
    }
  | { kind: "dispose"; id: string };

export type WorkerResponse =
  | { kind: "status"; id: string; status: PipelineStatus }
  | { kind: "analyzed"; id: string; result: AnalysisResult }
  | { kind: "extracted"; id: string; notes: Array<Pick<Note, "id"> & { audio: TransferableAudio }> }
  | { kind: "rendered"; id: string; audio: TransferableAudio }
  | { kind: "error"; id: string; error: string }
  | { kind: "ready"; id: string };

/**
 * AudioBuffer with the underlying Float32Array marked for zero-copy transfer
 * across the worker postMessage boundary.
 */
export interface TransferableAudio {
  samples: Float32Array;
  channels: 1 | 2;
  sampleRate: number;
  frames: number;
}

/**
 * Build a stable hash-id for a note from its frequency signature. Avoids
 * collisions between re-runs of the analyzer.
 */
export function noteIdFrom(startSec: number, pitchMidi: number, durationSec: number): string {
  return `n_${pitchMidi}_${Math.round(startSec * 1000)}_${Math.round(durationSec * 1000)}`;
}

/**
 * Canonical sample rates. Basic Pitch expects 22.05 kHz mono internally; the
 * mask U-Net operates on 48 kHz stereo STFT; the public interface is 48 kHz.
 */
export const PUBLIC_SR = 48000 as const;
export const BASIC_PITCH_SR = 22050 as const;
export const STFT_N_FFT = 2048 as const;
export const STFT_HOP = 512 as const;

/**
 * HCQT configuration — six harmonic scalings, 60 bins/octave (20-cent resolution),
 * matches Bittner 2017 / 2022 defaults.
 */
export const HCQT_BINS_PER_OCTAVE = 60 as const;
export const HCQT_N_OCTAVES = 6 as const;
export const HCQT_HARMONICS = [0.5, 1, 2, 3, 4, 5] as const;
export const HCQT_FMIN_HZ = 32.7 as const;  // C1
