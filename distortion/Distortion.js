/**
 * Distortion Plugin
 * Hard clipping distortion with aggressive harmonic generation
 *
 * Features:
 * - Hard clipping with pre/post filtering
 * - Tone stack for shaping distortion character
 * - Multiple clipping algorithms
 * - High gain capability
 * - Pre/post filter positioning
 *
 * @class Distortion
 */
class Distortion {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Pre-gain (drive)
    this.preGain = audioContext.createGain();
    this.preGain.gain.value = 1.0;

    // Waveshaper for distortion
    this.shaper = audioContext.createWaveShaper();
    this.shaper.oversample = '4x'; // Reduce aliasing

    // Tone filter (bandpass or peaking)
    this.toneFilter = audioContext.createBiquadFilter();
    this.toneFilter.type = 'peaking';
    this.toneFilter.frequency.value = 1000;
    this.toneFilter.Q.value = 1;
    this.toneFilter.gain.value = 0;

    // Pre-filter (goes before distortion if filterPosition is 'pre')
    this.preFilter = audioContext.createBiquadFilter();
    this.preFilter.type = 'peaking';
    this.preFilter.frequency.value = 1000;
    this.preFilter.Q.value = 1;
    this.preFilter.gain.value = 0;

    // Output gain
    this.outputGain = audioContext.createGain();
    this.outputGain.gain.value = 0.5; // Compensate for high gain

    // Dry/wet mix
    this.dryGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();
    this.dryGain.gain.value = 0; // 100% wet by default
    this.wetGain.gain.value = 1;

    // Parameters
    this.params = {
      drive: 50,              // 0-100%
      tone: 1000,             // 20 Hz to 20 kHz
      toneWidth: 1,           // 0.1 to 10 (Q factor)
      filterPosition: 'post', // 'pre' or 'post'
      clipType: 'hard',       // 'hard', 'soft', 'asymmetric', 'foldback'
      output: 0,              // -24 to +24 dB
      mix: 100                // 0-100%
    };

    this.setupRouting();
    this.initialize(options);
  }

  /**
   * Setup audio routing (default: post-filtering)
   */
  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // We'll reconfigure wet path based on filterPosition
    this.configureFilterPosition();
  }

  /**
   * Configure filter position (pre or post distortion)
   */
  configureFilterPosition() {
    // Disconnect everything first
    try {
      this.input.disconnect(this.preGain);
      this.input.disconnect(this.preFilter);
      this.preGain.disconnect();
      this.preFilter.disconnect();
      this.shaper.disconnect();
      this.toneFilter.disconnect();
    } catch (e) {
      // Ignore errors if nodes weren't connected
    }

    if (this.params.filterPosition === 'pre') {
      // Pre-distortion filtering: input → preFilter → preGain → shaper → outputGain → wetGain → output
      this.input.connect(this.preFilter);
      this.preFilter.connect(this.preGain);
      this.preGain.connect(this.shaper);
      this.shaper.connect(this.outputGain);
    } else {
      // Post-distortion filtering: input → preGain → shaper → toneFilter → outputGain → wetGain → output
      this.input.connect(this.preGain);
      this.preGain.connect(this.shaper);
      this.shaper.connect(this.toneFilter);
      this.toneFilter.connect(this.outputGain);
    }

    this.outputGain.connect(this.wetGain);
    this.wetGain.connect(this.output);
  }

  /**
   * Initialize with options
   */
  initialize(options) {
    if (options.drive !== undefined) this.setDrive(options.drive);
    if (options.tone !== undefined) this.setTone(options.tone);
    if (options.toneWidth !== undefined) this.setToneWidth(options.toneWidth);
    if (options.filterPosition !== undefined) this.setFilterPosition(options.filterPosition);
    if (options.clipType !== undefined) this.setClipType(options.clipType);
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
    // Higher drive range for distortion: 1 to 50
    const drive = 1 + (this.params.drive / 100) * 49;
    this.preGain.gain.value = drive;
    this.updateCurve();
  }

  /**
   * Set tone center frequency (20 Hz to 20 kHz)
   */
  setTone(frequency) {
    this.params.tone = Math.max(20, Math.min(20000, frequency));
    this.toneFilter.frequency.value = this.params.tone;
    this.preFilter.frequency.value = this.params.tone;
  }

  /**
   * Set tone width (Q factor, 0.1 to 10)
   */
  setToneWidth(q) {
    this.params.toneWidth = Math.max(0.1, Math.min(10, q));
    this.toneFilter.Q.value = this.params.toneWidth;
    this.preFilter.Q.value = this.params.toneWidth;
  }

  /**
   * Set filter position ('pre' or 'post')
   */
  setFilterPosition(position) {
    if (position !== 'pre' && position !== 'post') {
      console.warn(`Invalid filter position: ${position}. Using 'post'.`);
      position = 'post';
    }
    this.params.filterPosition = position;
    this.configureFilterPosition();
  }

  /**
   * Set clipping type
   */
  setClipType(type) {
    const validTypes = ['hard', 'soft', 'asymmetric', 'foldback'];
    if (!validTypes.includes(type)) {
      console.warn(`Invalid clip type: ${type}. Using 'hard'.`);
      type = 'hard';
    }
    this.params.clipType = type;
    this.updateCurve();
  }

  /**
   * Set output gain (-24 to +24 dB)
   */
  setOutput(db) {
    this.params.output = Math.max(-24, Math.min(24, db));
    const gain = Math.pow(10, this.params.output / 20);
    // Base gain is 0.5 to compensate for high drive
    this.outputGain.gain.value = 0.5 * gain;
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
   * Update waveshaper curve based on current clip type and drive
   */
  updateCurve() {
    const curveLength = 4096;
    const curve = new Float32Array(curveLength);
    const drive = 1 + (this.params.drive / 100) * 49;

    for (let i = 0; i < curveLength; i++) {
      const x = (i * 2 / curveLength) - 1; // -1 to 1
      const input = x * drive;
      let y;

      switch (this.params.clipType) {
        case 'hard':
          // Hard clipping - brick wall limiting
          y = Math.max(-1, Math.min(1, input));
          break;

        case 'soft':
          // Soft clipping - tanh for comparison
          y = Math.tanh(input);
          break;

        case 'asymmetric':
          // Asymmetric clipping - different threshold for positive/negative
          if (input > 0) {
            y = Math.min(1, input * 1.5);
          } else {
            y = Math.max(-1, input * 0.8);
          }
          break;

        case 'foldback':
          // Foldback distortion - signal folds back at threshold
          const threshold = 1.0;
          if (Math.abs(input) > threshold) {
            const excess = Math.abs(input) - threshold;
            const folded = threshold - (excess % (2 * threshold));
            y = input > 0 ? folded : -folded;
          } else {
            y = input;
          }
          break;

        default:
          y = Math.max(-1, Math.min(1, input));
      }

      curve[i] = y;
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
    if (params.toneWidth !== undefined) this.setToneWidth(params.toneWidth);
    if (params.filterPosition !== undefined) this.setFilterPosition(params.filterPosition);
    if (params.clipType !== undefined) this.setClipType(params.clipType);
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
    this.preFilter.disconnect();
    this.outputGain.disconnect();
    this.dryGain.disconnect();
    this.wetGain.disconnect();
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Distortion;
}
