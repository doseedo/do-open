/**
 * wav.ts — minimal 16-bit PCM WAV encoder for the "download mix" button.
 *
 * Fully self-contained. Reads our planar `AudioBuffer` and emits a RIFF/WAV
 * blob that every browser and DAW can import.
 */

import { AudioBuffer } from "../pipeline/types";

/**
 * Encode `audio` as a 16-bit PCM WAV. Samples are clamped to [-1, 1] before
 * quantisation so a slightly hot mix doesn't wrap.
 */
export function encodeWav(audio: AudioBuffer): Blob {
  const { samples, channels, sampleRate, frames } = audio;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = channels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = frames * blockAlign;
  const headerSize = 44;
  const buffer = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(buffer);

  // RIFF header
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");

  // fmt chunk
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true); // PCM subchunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data chunk
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);

  // Interleave planar → LRLR and quantise.
  let offset = headerSize;
  if (channels === 1) {
    for (let i = 0; i < frames; i++) {
      view.setInt16(offset, floatToInt16(samples[i]), true);
      offset += 2;
    }
  } else {
    const right = frames;
    for (let i = 0; i < frames; i++) {
      view.setInt16(offset, floatToInt16(samples[i]), true);
      view.setInt16(offset + 2, floatToInt16(samples[right + i]), true);
      offset += 4;
    }
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function floatToInt16(s: number): number {
  const clamped = Math.max(-1, Math.min(1, s));
  return clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
}

function writeAscii(view: DataView, offset: number, s: string): void {
  for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
}
