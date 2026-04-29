/**
 * r1-multitap-delay-processor — true multitap delay with per-tap delay/gain/pan
 *
 * Implements a circular delay buffer with up to MAX_TAPS independent taps.
 * Each tap has its own delay (ms), gain (linear), and pan (-1..+1).
 * Output is stereo (constant-power pan law).
 * Feedback is taken from the SUM of taps and fed back into the line — so
 * feedback decays exponentially through repeated applications of the tap pattern.
 *
 * Tap configuration is sent via port message {type:'setTaps', taps:[{delay,gain,pan},...]}.
 * Defaults to 4 evenly spaced taps with decay if no message has been received.
 *
 * @author Doseedo R1
 */

const MAX_TAPS = 16;
const MAX_DELAY_SECONDS = 5.0;

class MultitapDelayProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'feedback', defaultValue: 0.3, minValue: 0,    maxValue: 0.99, automationRate: 'k-rate' },
      { name: 'mix',      defaultValue: 0.3, minValue: 0,    maxValue: 1.0,  automationRate: 'k-rate' },
      // baseTime in ms — used to seed default tap pattern when no explicit taps set
      { name: 'baseTime', defaultValue: 250, minValue: 0,    maxValue: 5000, automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    this._bufLen = Math.ceil(MAX_DELAY_SECONDS * sampleRate);
    // Mono delay line — input is summed L+R before write
    this._buf = new Float32Array(this._bufLen);
    this._writeIdx = 0;

    // Default tap pattern: 4 taps at multiples of baseTime, decaying gain, alternating pan
    this._taps = this._defaultTaps(250);
    this._userTapsSet = false;

    this.port.onmessage = (e) => {
      const msg = e.data;
      if (msg && msg.type === 'setTaps' && Array.isArray(msg.taps)) {
        this._taps = msg.taps.slice(0, MAX_TAPS).map((t) => ({
          delay: Math.max(0, Math.min(MAX_DELAY_SECONDS * 1000, +t.delay || 0)), // ms
          gain:  Math.max(0, Math.min(2, +t.gain ?? 1)),
          pan:   Math.max(-1, Math.min(1, +t.pan ?? 0)),
        }));
        this._userTapsSet = true;
      }
    };
  }

  _defaultTaps(baseMs) {
    const out = [];
    const count = 4;
    for (let i = 0; i < count; i++) {
      out.push({
        delay: baseMs * (i + 1),
        gain:  Math.pow(0.7, i),
        pan:   (i % 2 === 0) ? -0.6 : 0.6,
      });
    }
    return out;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!output || output.length < 1) return true;

    const feedback = parameters.feedback[0];
    const mix = parameters.mix[0];
    const baseTime = parameters.baseTime[0];

    // Regenerate default taps if user hasn't set them and baseTime changed
    if (!this._userTapsSet && Math.abs(this._taps[0]?.delay - baseTime) > 0.5) {
      this._taps = this._defaultTaps(baseTime);
    }

    const blockSize = output[0].length;
    const inL = input && input[0] ? input[0] : null;
    const inR = input && input[1] ? input[1] : inL;
    const outL = output[0];
    const outR = output[1] || output[0];

    const fbGain = Math.min(0.95, feedback);
    const wet = mix;
    const dry = 1 - mix;
    const buf = this._buf;
    const bufLen = this._bufLen;
    const taps = this._taps;
    const numTaps = taps.length;

    // Pre-compute tap delay in samples + pan gains
    const tapDelaySamps = new Array(numTaps);
    const tapGainL = new Array(numTaps);
    const tapGainR = new Array(numTaps);
    let tapSumGain = 0;
    for (let t = 0; t < numTaps; t++) {
      tapDelaySamps[t] = (taps[t].delay / 1000) * sampleRate;
      // Constant-power pan
      const pan = taps[t].pan;
      const angle = (pan + 1) * 0.25 * Math.PI; // 0..π/2
      tapGainL[t] = Math.cos(angle) * taps[t].gain;
      tapGainR[t] = Math.sin(angle) * taps[t].gain;
      tapSumGain += taps[t].gain;
    }
    // Normalize feedback by tap-sum to keep stable across tap counts
    const fbNorm = numTaps > 0 ? fbGain / Math.max(1, tapSumGain) : 0;

    let widx = this._writeIdx;

    for (let i = 0; i < blockSize; i++) {
      const dryL = inL ? inL[i] : 0;
      const dryR = inR ? inR[i] : dryL;
      const drySample = (dryL + dryR) * 0.5;

      // Read all taps & accumulate L/R
      let wetL = 0, wetR = 0;
      let fbAccum = 0;
      for (let t = 0; t < numTaps; t++) {
        const d = tapDelaySamps[t];
        // Linear interpolation for fractional delay
        let ridx = widx - d;
        while (ridx < 0) ridx += bufLen;
        while (ridx >= bufLen) ridx -= bufLen;
        const i0 = Math.floor(ridx);
        const i1 = (i0 + 1) % bufLen;
        const frac = ridx - i0;
        const s = buf[i0] * (1 - frac) + buf[i1] * frac;

        wetL += s * tapGainL[t];
        wetR += s * tapGainR[t];
        fbAccum += s * taps[t].gain;
      }

      // Write input + feedback into delay buffer
      buf[widx] = drySample + fbAccum * fbNorm;
      widx += 1;
      if (widx >= bufLen) widx = 0;

      outL[i] = dryL * dry + wetL * wet;
      outR[i] = dryR * dry + wetR * wet;
    }

    this._writeIdx = widx;
    return true;
  }
}

registerProcessor('r1-multitap-delay-processor', MultitapDelayProcessor);
