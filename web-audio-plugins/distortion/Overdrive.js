/**
 * Overdrive Plugin
 * Tube-style soft clipping for warm, musical distortion
 *
 * Features:
 * - Soft clipping (tanh, atan, or custom curves)
 * - Asymmetric distortion for even harmonics
 * - Tone stack (post-distortion EQ)
 * - Auto gain compensation
 * - Dry/wet mix control
 *
 * @class Overdrive
 */
class Overdrive {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Pre-gain (drive)
    this.preGain = audioContext.createGain();
    this.preGain.gain.value = 1.0;

    // Waveshaper (soft clipping)
    this.shaper = audioContext.createWaveShaper();
    this.shaper.oversample = '4x'; // Reduce aliasing

    // Tone control (post-distortion lowshelf filter)
    this.toneFilter = audioContext.createBiquadFilter();
    this.toneFilter.type = 'lowshelf';
    this.toneFilter.frequency.value = 1000;
    this.toneFilter.gain.value = 0;

    // Output gain (makeup gain)
    this.outputGain = audioContext.createGain();
    this.outputGain.gain.value = 1.0;

    // Dry/wet mix
    this.dryGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();
    this.dryGain.gain.value = 0; // 100% wet by default
    this.wetGain.gain.value = 1;

    // Parameters
    this.params = {
      drive: 30,      // 0-100%
      tone: 50,       // 0-100%
      bias: 0,        // -100 to +100%
      output: 0,      // -24 to +24 dB
      mix: 100        // 0-100%
    };

    // Curve generation parameters
    this.curveType = 'tanh'; // 'tanh', 'atan', 'softClip'

    this.setupRouting();
    this.initialize(options);
  }

  /**
   * Setup audio routing
   */
  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path: input → preGain → shaper → tone → outputGain → wetGain → output
    this.input.connect(this.preGain);
    this.preGain.connect(this.shaper);
    this.shaper.connect(this.toneFilter);
    this.toneFilter.connect(this.outputGain);
    this.outputGain.connect(this.wetGain);
    this.wetGain.connect(this.output);
  }

  /**
   * Initialize with options
   */
  initialize(options) {
    if (options.drive !== undefined) this.setDrive(options.drive);
    if (options.tone !== undefined) this.setTone(options.tone);
    if (options.bias !== undefined) this.setBias(options.bias);
    if (options.output !== undefined) this.setOutput(options.output);
    if (options.mix !== undefined) this.setMix(options.mix);
    if (options.curveType !== undefined) this.curveType = options.curveType;

    // Generate initial curve
    this.updateCurve();
  }

  /**
   * Set drive amount (0-100%)
   * Maps to gain 1-20
   */
  setDrive(percent) {
    this.params.drive = Math.max(0, Math.min(100, percent));
    const drive = 1 + (this.params.drive / 100) * 19;
    this.preGain.gain.value = drive;
    this.updateCurve();
  }

  /**
   * Set tone control (0-100%)
   * 0 = dark, 100 = bright
   */
  setTone(percent) {
    this.params.tone = Math.max(0, Math.min(100, percent));

    // Tone control: 200 Hz to 5000 Hz lowshelf
    const freq = 200 + (this.params.tone / 100) * 4800;
    this.toneFilter.frequency.value = freq;

    // Gain: -6 dB to +6 dB
    const gain = (this.params.tone / 100) * 12 - 6;
    this.toneFilter.gain.value = gain;
  }

  /**
   * Set bias for asymmetric distortion (-100 to +100%)
   * Creates even harmonics
   */
  setBias(percent) {
    this.params.bias = Math.max(-100, Math.min(100, percent));
    this.updateCurve();
  }

  /**
   * Set output gain (-24 to +24 dB)
   */
  setOutput(db) {
    this.params.output = Math.max(-24, Math.min(24, db));
    const gain = Math.pow(10, this.params.output / 20);
    this.outputGain.gain.value = gain;
  }

  /**
   * Set dry/wet mix (0-100%)
   */
  setMix(percent) {
    this.params.mix = Math.max(0, Math.min(100, percent));
    const wet = this.params.mix / 100;
    const dry = 1 - wet;

    this.wetGain.gain.value = wet;
    this.dryGain.gain.value = dry;
  }

  /**
   * Set curve type ('tanh', 'atan', 'softClip')
   */
  setCurveType(type) {
    this.curveType = type;
    this.updateCurve();
  }

  /**
   * Update waveshaper curve with current drive and bias
   */
  updateCurve() {
    const curveLength = 4096;
    const curve = new Float32Array(curveLength);
    const drive = 1 + (this.params.drive / 100) * 19;
    const bias = this.params.bias / 100; // -1 to 1

    for (let i = 0; i < curveLength; i++) {
      let x = (i * 2 / curveLength) - 1; // -1 to 1

      // Apply bias for asymmetric distortion
      x += bias;

      // Apply different clipping curves
      switch (this.curveType) {
        case 'tanh':
          // Hyperbolic tangent - smooth, warm saturation
          curve[i] = Math.tanh(x * drive);
          break;

        case 'atan':
          // Arctangent - softer than tanh
          curve[i] = (2 / Math.PI) * Math.atan(x * drive * Math.PI / 2);
          break;

        case 'softClip':
          // Algebraic soft clipping
          const val = x * drive;
          if (Math.abs(val) < 1) {
            curve[i] = val;
          } else {
            curve[i] = val / (1 + Math.abs(val));
          }
          break;

        default:
          curve[i] = Math.tanh(x * drive);
      }
    }

    this.shaper.curve = curve;
  }

  /**
   * Get current parameters
   */
  getParams() {
    return { ...this.params };
  }

  /**
   * Set all parameters at once
   */
  setParams(params) {
    if (params.drive !== undefined) this.setDrive(params.drive);
    if (params.tone !== undefined) this.setTone(params.tone);
    if (params.bias !== undefined) this.setBias(params.bias);
    if (params.output !== undefined) this.setOutput(params.output);
    if (params.mix !== undefined) this.setMix(params.mix);
  }

  /**
   * Connect to destination
   */
  connect(destination) {
    this.output.connect(destination);
    return this;
  }

  /**
   * Disconnect from all destinations
   */
  disconnect() {
    this.output.disconnect();
  }

  /**
   * Dispose and clean up resources
   */
  dispose() {
    this.disconnect();
    this.input.disconnect();
    this.preGain.disconnect();
    this.shaper.disconnect();
    this.toneFilter.disconnect();
    this.outputGain.disconnect();
    this.dryGain.disconnect();
    this.wetGain.disconnect();
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Overdrive;
}
