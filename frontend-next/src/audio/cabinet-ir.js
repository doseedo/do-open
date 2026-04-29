/**
 * Synthetic 4x10 guitar/bass cabinet impulse response
 *
 * Crude bandpass-shaped impulse (~40ms) used as a fallback when no real cabinet
 * IR file is loaded. Designed to give a Bassman-ish frequency curve:
 *   • aggressive rolloff above ~4kHz (speaker cone breakup)
 *   • highpass around ~80Hz (cabinet resonance / port tuning)
 *   • a small peak around 2-3kHz (presence range)
 *
 * The shape is generated procedurally so it adds zero binary weight to the
 * bundle. Real users can override by loading a .wav and writing it into the
 * ConvolverNode's buffer.
 *
 * Author: Agent R4 (Circuit Models)
 */

/**
 * Build a cabinet IR AudioBuffer.
 * @param {BaseAudioContext} ctx
 * @param {object} [opts]
 * @param {number} [opts.durationSec=0.04]   IR length in seconds (40 ms default)
 * @param {number} [opts.lowCutHz=80]        Highpass corner
 * @param {number} [opts.highCutHz=4000]     Lowpass corner (speaker cone breakup)
 * @param {number} [opts.presencePeakHz=2500]
 * @returns {AudioBuffer}
 */
export function buildCabinetIR(ctx, opts = {}) {
  const sampleRate = ctx.sampleRate;
  const duration = opts.durationSec ?? 0.04;
  const lowCutHz = opts.lowCutHz ?? 80;
  const highCutHz = opts.highCutHz ?? 4000;
  const presencePeakHz = opts.presencePeakHz ?? 2500;

  const length = Math.max(64, Math.floor(sampleRate * duration));
  const buffer = ctx.createBuffer(1, length, sampleRate);
  const data = buffer.getChannelData(0);

  // 1. Start with a shaped impulse: dense early reflections then exponential decay
  for (let i = 0; i < length; i++) {
    const t = i / sampleRate;
    // Decay envelope (fast, like a closed-back cab)
    const env = Math.exp(-t * 80);
    // Add a couple of early reflections to thicken the IR
    const refl1 = (i === 0 ? 1 : 0);
    const refl2 = (i === Math.floor(sampleRate * 0.0008) ? -0.45 : 0);
    const refl3 = (i === Math.floor(sampleRate * 0.0021) ? 0.28 : 0);
    const noise = (Math.random() * 2 - 1) * 0.15 * env;
    data[i] = refl1 + refl2 + refl3 + noise;
  }

  // 2. Apply a simple biquad-like cascade in the time domain to shape the magnitude
  //    (we can't run an OfflineAudioContext synchronously here, so we do a cheap
  //    1-pole HP + 1-pole LP + a fixed peak).
  const lowCutA = Math.exp(-2 * Math.PI * lowCutHz / sampleRate);
  const highCutA = Math.exp(-2 * Math.PI * highCutHz / sampleRate);

  // 1-pole highpass
  let prevIn = 0, prevOut = 0;
  for (let i = 0; i < length; i++) {
    const x = data[i];
    const y = lowCutA * (prevOut + x - prevIn);
    prevIn = x;
    prevOut = y;
    data[i] = y;
  }

  // 1-pole lowpass
  let lpState = 0;
  for (let i = 0; i < length; i++) {
    lpState = highCutA * lpState + (1 - highCutA) * data[i];
    data[i] = lpState;
  }

  // Presence bump: very cheap resonant peak via a single biquad-style allocation
  const wp = 2 * Math.PI * presencePeakHz / sampleRate;
  const Q = 2.0;
  const alpha = Math.sin(wp) / (2 * Q);
  const cosw = Math.cos(wp);
  const b0 = 1 + alpha;
  const b1 = -2 * cosw;
  const b2 = 1 - alpha;
  const a0 = 1;
  const a1 = -2 * cosw;
  const a2 = 1 - alpha;
  let z1 = 0, z2 = 0, w1 = 0, w2 = 0;
  for (let i = 0; i < length; i++) {
    const x = data[i];
    const y = (b0 * x + b1 * z1 + b2 * z2 - a1 * w1 - a2 * w2) / a0;
    z2 = z1; z1 = x;
    w2 = w1; w1 = y;
    // Mix presence peak in lightly — full replacement would be too peaky
    data[i] = data[i] * 0.7 + y * 0.3;
  }

  // 3. Normalize so peak is at 0 dBFS (ConvolverNode does its own gain compensation
  //    via `normalize=true`, but we still want a sane peak).
  let peak = 0;
  for (let i = 0; i < length; i++) {
    const a = Math.abs(data[i]);
    if (a > peak) peak = a;
  }
  if (peak > 0) {
    const scale = 0.95 / peak;
    for (let i = 0; i < length; i++) data[i] *= scale;
  }

  return buffer;
}

export default buildCabinetIR;
