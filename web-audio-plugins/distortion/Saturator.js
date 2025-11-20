/**
 * Saturator Plugin
 * Multi-mode saturation from subtle warmth to heavy distortion
 *
 * Features:
 * - Multiple saturation algorithms (warm, digital, analog, clip, foldback, sine-fold)
 * - Pre/post filtering
 * - DC offset removal
 * - Oversampling to reduce aliasing
 * - Color parameter for harmonic emphasis
 *
 * @class Saturator
 */
class Saturator {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Pre-gain (drive)
    this.preGain = audioContext.createGain();
    this.preGain.gain.value = 1.0;

    // Waveshaper for saturation
    this.shaper = audioContext.createWaveShaper();
    this.shaper.oversample = '4x'; // Reduce aliasing

    // DC filter (highpass at 5 Hz to remove DC offset)
    this.dcFilter = audioContext.createBiquadFilter();
    this.dcFilter.type = 'highpass';
    this.dcFilter.frequency.value = 5;
    this.dcFilter.Q.value = 0.7071; // Butterworth response

    // Color filter (harmonic emphasis)
    this.colorFilter = audioContext.createBiquadFilter();
    this.colorFilter.type = 'peaking';
    this.colorFilter.frequency.value = 2000;
    this.colorFilter.Q.value = 2;
    this.colorFilter.gain.value = 0;

    // Output gain
    this.outputGain = audioContext.createGain();
    this.outputGain.gain.value = 1.0;

    // Dry/wet mix
    this.dryGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();
    this.dryGain.gain.value = 0; // 100% wet by default
    this.wetGain.gain.value = 1;

    // Parameters
    this.params = {
      drive: 0,           // 0-100%
      type: 'warm',       // 'warm', 'digital', 'analog', 'clip', 'foldback', 'sine-fold'
      color: 0,           // 0-100%
      depth: 100,         // 0-100% (wet/dry character)
      dcFilter: true,     // boolean
      output: 0,          // -24 to +24 dB
      mix: 100            // 0-100%
    };

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

    // Wet path: input → preGain → shaper → dcFilter → colorFilter → outputGain → wetGain → output
    this.input.connect(this.preGain);
    this.preGain.connect(this.shaper);
    this.shaper.connect(this.dcFilter);
    this.dcFilter.connect(this.colorFilter);
    this.colorFilter.connect(this.outputGain);
    this.outputGain.connect(this.wetGain);
    this.wetGain.connect(this.output);
  }

  /**
   * Initialize with options
   */
  initialize(options) {
    if (options.drive !== undefined) this.setDrive(options.drive);
    if (options.type !== undefined) this.setType(options.type);
    if (options.color !== undefined) this.setColor(options.color);
    if (options.depth !== undefined) this.setDepth(options.depth);
    if (options.dcFilter !== undefined) this.setDCFilter(options.dcFilter);
    if (options.output !== undefined) this.setOutput(options.output);
    if (options.mix !== undefined) this.setMix(options.mix);

    // Generate initial curve
    this.updateCurve();
  }

  /**
   * Set drive amount (0-100%)
   */
  setDrive(percent) {
    this.params.drive = Math.max(0, Math.min(100, percent));
    const drive = 1 + (this.params.drive / 100) * 9; // 1 to 10
    this.preGain.gain.value = drive;
    this.updateCurve();
  }

  /**
   * Set saturation type
   */
  setType(type) {
    const validTypes = ['warm', 'digital', 'analog', 'clip', 'foldback', 'sine-fold'];
    if (!validTypes.includes(type)) {
      console.warn(`Invalid saturation type: ${type}. Using 'warm'.`);
      type = 'warm';
    }
    this.params.type = type;
    this.updateCurve();
  }

  /**
   * Set color (harmonic emphasis, 0-100%)
   */
  setColor(percent) {
    this.params.color = Math.max(0, Math.min(100, percent));

    // Frequency range: 2000 Hz to 8000 Hz
    const freq = 2000 + (this.params.color / 100) * 6000;
    this.colorFilter.frequency.value = freq;

    // Gain: 0 dB to +6 dB
    const gain = (this.params.color / 100) * 6;
    this.colorFilter.gain.value = gain;
  }

  /**
   * Set depth (wet/dry character, 0-100%)
   */
  setDepth(percent) {
    this.params.depth = Math.max(0, Math.min(100, percent));
    // Depth affects internal wet/dry balance
    // This is different from the main mix parameter
    // For now, we'll use it to affect the drive curve intensity
    this.updateCurve();
  }

  /**
   * Enable/disable DC filter
   */
  setDCFilter(enabled) {
    this.params.dcFilter = enabled;
    // We can't bypass a BiquadFilter directly, so we'll adjust its frequency
    if (enabled) {
      this.dcFilter.frequency.value = 5; // 5 Hz highpass
    } else {
      this.dcFilter.frequency.value = 0.1; // Essentially bypass
    }
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
   * Update waveshaper curve based on current type and drive
   */
  updateCurve() {
    const curveLength = 4096;
    const curve = new Float32Array(curveLength);
    const drive = 1 + (this.params.drive / 100) * 9;
    const depthFactor = this.params.depth / 100;

    for (let i = 0; i < curveLength; i++) {
      const x = (i * 2 / curveLength) - 1; // -1 to 1
      let y;

      switch (this.params.type) {
        case 'warm':
          // Soft tanh saturation - warm, musical
          y = Math.tanh(x * drive);
          break;

        case 'digital':
          // Hard clipping - aggressive digital sound
          y = Math.max(-1, Math.min(1, x * drive));
          break;

        case 'analog':
          // Asymmetric soft clip - simulates analog circuits
          const biasedX = x + 0.1; // Slight asymmetry
          y = Math.tanh(biasedX * drive);
          break;

        case 'clip':
          // Very hard clip - extreme distortion
          const clipped = x * drive;
          y = clipped > 0.1 ? 1 : (clipped < -0.1 ? -1 : clipped * 10);
          break;

        case 'foldback':
          // Foldback distortion - complex harmonics
          const folded = x * drive;
          y = Math.abs((folded + 1) % 4 - 2) - 1;
          break;

        case 'sine-fold':
          // Sine folding - musical harmonics
          y = Math.sin(x * drive * Math.PI);
          break;

        default:
          y = Math.tanh(x * drive);
      }

      // Apply depth (blend between dry and saturated)
      curve[i] = x + (y - x) * depthFactor;
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
    if (params.type !== undefined) this.setType(params.type);
    if (params.color !== undefined) this.setColor(params.color);
    if (params.depth !== undefined) this.setDepth(params.depth);
    if (params.dcFilter !== undefined) this.setDCFilter(params.dcFilter);
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
    this.dcFilter.disconnect();
    this.colorFilter.disconnect();
    this.outputGain.disconnect();
    this.dryGain.disconnect();
    this.wetGain.disconnect();
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Saturator;
}
