/**
 * Erosion Effect
 *
 * Noise-based distortion for aggressive digital artifacts and ring modulation effects.
 * Inspired by Ableton Live's Erosion device.
 *
 * Features:
 * - Noise modulation of signal
 * - Multiple erosion algorithms/modes
 * - Bandwidth control for character
 * - Can create bit-crushed or ring-mod like effects
 * - Four distinct noise character modes
 *
 * @author Agent 7: Creative Effects
 */

class Erosion {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = this.context.createGain();
    this.output = this.context.createGain();
    this.dryGain = this.context.createGain();
    this.wetGain = this.context.createGain();

    // Noise source and filtering
    this.noiseBuffer = null;
    this.noiseSource = null;
    this.noiseGain = this.context.createGain();
    this.noiseFilter = this.context.createBiquadFilter();
    this.noiseFilter.type = 'bandpass';

    // Modulation processing (using ScriptProcessor for ring modulation)
    this.processor = null;

    // Parameters with defaults
    this.params = {
      mode: 1,                 // I (1), II (2), III (3), IV (4) - noise character/algorithm
      frequency: 1000,         // 20 Hz to 18 kHz - noise frequency
      width: 50,               // 0 to 100% - noise bandwidth (Q factor)
      amount: 0,               // 0 to 100% - distortion intensity
      dryWet: 50               // 0 to 100% - dry/wet mix
    };

    // Initialize
    this.setupRouting();
    this.createNoise();

    // Apply user options
    if (options) {
      Object.keys(options).forEach(key => {
        if (this.params.hasOwnProperty(key)) {
          this.params[key] = options[key];
        }
      });
    }

    this.updateNoiseFilter();
    this.updateMix();
  }

  /**
   * Setup audio routing
   */
  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Noise filter chain
    this.noiseFilter.connect(this.noiseGain);
  }

  /**
   * Create noise buffer and source
   */
  createNoise() {
    const bufferSize = 2 * this.context.sampleRate; // 2 seconds of noise
    this.noiseBuffer = this.context.createBuffer(1, bufferSize, this.context.sampleRate);
    const output = this.noiseBuffer.getChannelData(0);

    // Generate white noise
    for (let i = 0; i < bufferSize; i++) {
      output[i] = Math.random() * 2 - 1;
    }

    // Create looping noise source
    this.startNoiseSource();

    // Setup ring modulation processor
    this.setupModulation();
  }

  /**
   * Start or restart noise source
   */
  startNoiseSource() {
    // Stop existing source if any
    if (this.noiseSource) {
      try {
        this.noiseSource.stop();
        this.noiseSource.disconnect();
      } catch (e) {
        // Already stopped
      }
    }

    // Create new source
    this.noiseSource = this.context.createBufferSource();
    this.noiseSource.buffer = this.noiseBuffer;
    this.noiseSource.loop = true;

    // Connect to filter
    this.noiseSource.connect(this.noiseFilter);

    // Start immediately
    this.noiseSource.start();
  }

  /**
   * Setup ScriptProcessor for ring modulation
   * Multiplies input signal with filtered noise
   */
  setupModulation() {
    const bufferSize = 4096;
    this.processor = this.context.createScriptProcessor(bufferSize, 2, 2);

    // Temporary buffer to capture noise
    this.noiseOutputBuffer = new Float32Array(bufferSize);
    let noiseWritePos = 0;

    // Create an analyzer to capture noise signal
    const noiseCapture = this.context.createScriptProcessor(bufferSize, 1, 1);
    this.noiseGain.connect(noiseCapture);
    noiseCapture.connect(this.context.createGain()); // Dummy connection

    noiseCapture.onaudioprocess = (e) => {
      const noise = e.inputBuffer.getChannelData(0);
      this.noiseOutputBuffer.set(noise);
    };

    // Main processing
    this.processor.onaudioprocess = (e) => {
      const inputL = e.inputBuffer.getChannelData(0);
      const inputR = e.inputBuffer.getChannelData(1);
      const outputL = e.outputBuffer.getChannelData(0);
      const outputR = e.outputBuffer.getChannelData(1);

      const amount = this.params.amount / 100;
      const mode = this.params.mode;

      for (let i = 0; i < bufferSize; i++) {
        const noise = this.noiseOutputBuffer[i] || 0;

        // Apply different modulation algorithms based on mode
        let modulatedL, modulatedR;

        switch (mode) {
          case 1: // Mode I - Classic ring modulation
            modulatedL = inputL[i] * (1 + noise * amount);
            modulatedR = inputR[i] * (1 + noise * amount);
            break;

          case 2: // Mode II - Asymmetric modulation
            modulatedL = inputL[i] * (1 + Math.abs(noise) * amount);
            modulatedR = inputR[i] * (1 + Math.abs(noise) * amount);
            break;

          case 3: // Mode III - Sample & hold style
            const threshold = 1 - amount;
            modulatedL = Math.abs(noise) > threshold ? inputL[i] * Math.sign(noise) : inputL[i];
            modulatedR = Math.abs(noise) > threshold ? inputR[i] * Math.sign(noise) : inputR[i];
            break;

          case 4: // Mode IV - Bit crushing style
            const noiseBits = Math.floor((1 - amount) * 8 + 1); // 1-8 bits
            const step = Math.pow(2, noiseBits);
            const noiseQuantized = Math.floor(noise * step) / step;
            modulatedL = inputL[i] * (1 + noiseQuantized * amount);
            modulatedR = inputR[i] * (1 + noiseQuantized * amount);
            break;

          default:
            modulatedL = inputL[i];
            modulatedR = inputR[i];
        }

        // Apply soft clipping to prevent extreme values
        outputL[i] = this.softClip(modulatedL);
        outputR[i] = this.softClip(modulatedR);
      }
    };

    // Connect processor
    this.input.connect(this.processor);
    this.processor.connect(this.wetGain);
    this.wetGain.connect(this.output);
  }

  /**
   * Soft clipping function to prevent harsh distortion
   */
  softClip(value) {
    if (value > 1) {
      return 1;
    } else if (value < -1) {
      return -1;
    } else {
      // Soft knee clipping using tanh-like curve
      return value * (1.5 - 0.5 * value * value);
    }
  }

  /**
   * Update noise filter parameters
   */
  updateNoiseFilter() {
    // Set center frequency
    this.noiseFilter.frequency.value = this.params.frequency;

    // Set bandwidth (Q factor)
    // width: 0% = narrow (high Q), 100% = wide (low Q)
    const minQ = 0.1;
    const maxQ = 20;
    const normalizedWidth = this.params.width / 100;
    const Q = maxQ - (normalizedWidth * (maxQ - minQ));
    this.noiseFilter.Q.value = Q;
  }

  /**
   * Update dry/wet mix
   */
  updateMix() {
    const wet = this.params.dryWet / 100;
    const dry = 1 - wet;

    this.dryGain.gain.value = dry;
    this.wetGain.gain.value = wet;
  }

  // ========== Public Parameter Methods ==========

  setMode(mode) {
    const validMode = Math.max(1, Math.min(4, Math.floor(mode)));
    this.params.mode = validMode;
  }

  setFrequency(hz) {
    this.params.frequency = Math.max(20, Math.min(18000, hz));
    this.updateNoiseFilter();
  }

  setWidth(percent) {
    this.params.width = Math.max(0, Math.min(100, percent));
    this.updateNoiseFilter();
  }

  setAmount(percent) {
    this.params.amount = Math.max(0, Math.min(100, percent));
  }

  setDryWet(percent) {
    this.params.dryWet = Math.max(0, Math.min(100, percent));
    this.updateMix();
  }

  /**
   * Get current parameters
   */
  getParams() {
    return { ...this.params };
  }

  /**
   * Connect to another audio node
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
    return this;
  }

  /**
   * Cleanup and destroy effect
   */
  destroy() {
    // Stop noise source
    if (this.noiseSource) {
      try {
        this.noiseSource.stop();
        this.noiseSource.disconnect();
      } catch (e) {
        // Already stopped
      }
      this.noiseSource = null;
    }

    // Disconnect processor
    if (this.processor) {
      this.processor.disconnect();
      this.input.disconnect(this.processor);
      this.processor = null;
    }

    // Disconnect nodes
    this.input.disconnect();
    this.dryGain.disconnect();
    this.wetGain.disconnect();
    this.noiseGain.disconnect();
    this.noiseFilter.disconnect();
    this.output.disconnect();
  }
}

export default Erosion;
