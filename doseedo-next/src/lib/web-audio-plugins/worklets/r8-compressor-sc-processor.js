/**
 * r8-compressor-sc-processor.js
 *
 * SIDECHAIN-CAPABLE compressor variant. Created by Agent R8.
 *
 * Why this exists (vs editing compressor-processor.js in-place):
 *   The stock compressor-processor.js only ever reads inputs[0] for both the
 *   detector path AND the audio path. Logic Pro's Compressor / Noise Gate /
 *   Enveloper can have their detector ("key input") driven by an external
 *   signal — a kick on a bass channel, a vocal on a delay return, etc.
 *
 *   Rather than mutate the existing processor (which is referenced by every
 *   compressor instance in the engine and would change behavior for all of
 *   them) this is a NEW worklet exposing TWO inputs:
 *     inputs[0] = main audio (the signal being compressed)
 *     inputs[1] = sidechain key input (the detector signal)
 *
 *   The processor uses inputs[1] for envelope detection when present and
 *   `sidechain_active` parameter > 0.5; otherwise it falls back to inputs[0]
 *   (effectively the same behavior as the stock processor). This lets the
 *   builder always allocate two inputs at AudioWorkletNode-construction time
 *   and the user can wire — or leave unwired — the key input dynamically.
 *
 * Registered name: 'r8-compressor-sc-processor'
 *
 * @author Agent R8 — Sidechain Routing
 * @version 1.0.0
 */

importScripts('../core/dsp-utils.js');

class R8CompressorSCProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'threshold',  defaultValue: -24,  minValue: -60, maxValue: 0,    automationRate: 'k-rate' },
      { name: 'ratio',      defaultValue: 4,    minValue: 1,   maxValue: 20,   automationRate: 'k-rate' },
      { name: 'attack',     defaultValue: 0.010, minValue: 0.0001, maxValue: 0.5,  automationRate: 'k-rate' },
      { name: 'release',    defaultValue: 0.100, minValue: 0.001,  maxValue: 2.0,  automationRate: 'k-rate' },
      { name: 'knee',       defaultValue: 0,    minValue: 0,   maxValue: 12,   automationRate: 'k-rate' },
      { name: 'makeupGain', defaultValue: 0,    minValue: 0,   maxValue: 24,   automationRate: 'k-rate' },
      { name: 'mix',        defaultValue: 1.0,  minValue: 0,   maxValue: 1.0,  automationRate: 'k-rate' },
      // 0/1 flag (k-rate). When 1, detector reads inputs[1]; when 0, falls
      // back to inputs[0] — same behavior as stock compressor-processor.
      { name: 'sidechain_active', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: 'k-rate' }
    ];
  }

  constructor(options) {
    super();

    this.envelopes = [
      new EnvelopeFollower(0.010, 0.100, sampleRate),
      new EnvelopeFollower(0.010, 0.100, sampleRate)
    ];
    this.gainSmoothers = [
      new OnePoleFilter(10, sampleRate),
      new OnePoleFilter(10, sampleRate)
    ];

    this.threshold = -24;
    this.ratio = 4;
    this.attack = 0.010;
    this.release = 0.100;
    this.knee = 0;
    this.makeupGain = 0;
    this.mix = 1.0;
    this.currentGainReduction = 0;
  }

  calculateGainReduction(levelDb) {
    if (levelDb <= this.threshold - this.knee / 2) return 0;
    if (this.knee > 0 && levelDb < this.threshold + this.knee / 2) {
      const delta = levelDb - this.threshold;
      const slope = 1 / this.ratio - 1;
      return slope * delta * delta / (2 * this.knee);
    }
    return (levelDb - this.threshold) * (1 - 1 / this.ratio);
  }

  process(inputs, outputs, parameters) {
    const audio = inputs[0];
    const sidechain = inputs[1]; // may be undefined / empty
    const output = outputs[0];

    if (!audio || !audio.length) return true;

    const threshold = parameters.threshold;
    const ratio = parameters.ratio;
    const attack = parameters.attack;
    const release = parameters.release;
    const knee = parameters.knee;
    const makeupGain = parameters.makeupGain;
    const mix = parameters.mix;
    const sidechainActive = parameters.sidechain_active;

    const isThresholdArray = threshold.length > 1;
    const blockSize = audio[0].length;

    // Decide which signal feeds the detector. Sidechain is "live" only if
    // the flag is set AND inputs[1] actually has channel data this block.
    const scLive =
      (isThresholdArray ? sidechainActive[0] : sidechainActive[0]) > 0.5 &&
      sidechain && sidechain.length > 0 && sidechain[0] && sidechain[0].length > 0;

    for (let channel = 0; channel < audio.length; channel++) {
      const inputChannel = audio[channel];
      const outputChannel = output[channel];
      const envelope = this.envelopes[channel];
      const gainSmoother = this.gainSmoothers[channel];

      // Pick detector channel: try matching channel from sidechain, fall
      // back to its channel 0, fall back to audio channel itself.
      let detectorChannel = inputChannel;
      if (scLive) {
        detectorChannel = sidechain[channel] || sidechain[0] || inputChannel;
      }

      for (let i = 0; i < blockSize; i++) {
        this.threshold  = isThresholdArray ? threshold[i]  : threshold[0];
        this.ratio      = isThresholdArray ? ratio[i]      : ratio[0];
        this.attack     = isThresholdArray ? attack[i]     : attack[0];
        this.release    = isThresholdArray ? release[i]    : release[0];
        this.knee       = isThresholdArray ? knee[i]       : knee[0];
        this.makeupGain = isThresholdArray ? makeupGain[i] : makeupGain[0];
        this.mix        = isThresholdArray ? mix[i]        : mix[0];

        envelope.setAttack(this.attack);
        envelope.setRelease(this.release);

        const audioSample = inputChannel[i];
        const detectorSample = detectorChannel[i];

        // 1. Envelope detection runs on the DETECTOR sample (sidechain or self)
        const envelopeLevel = envelope.process(detectorSample);
        const levelDb = gainToDb(envelopeLevel);

        // 2. Gain reduction is computed from detector level
        const gainReductionDb = this.calculateGainReduction(levelDb);
        this.currentGainReduction = gainReductionDb;

        // 3. Smooth the linear gain
        const gainReduction = dbToGain(-gainReductionDb);
        const smoothedGain = gainSmoother.process(gainReduction);

        const makeupGainLinear = dbToGain(this.makeupGain);

        // 4. Apply gain reduction to the AUDIO sample (NOT the detector)
        const compressed = audioSample * smoothedGain * makeupGainLinear;

        // 5. Wet/dry mix on audio
        outputChannel[i] = audioSample * (1 - this.mix) + compressed * this.mix;
      }
    }

    this.port.postMessage({
      type: 'gainReduction',
      value: this.currentGainReduction,
      sidechainActive: scLive
    });

    return true;
  }
}

registerProcessor('r8-compressor-sc-processor', R8CompressorSCProcessor);
