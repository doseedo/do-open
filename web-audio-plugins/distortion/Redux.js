/**
 * Redux Plugin
 * Bit crushing and sample rate reduction for lo-fi digital artifacts
 *
 * Features:
 * - Bit depth reduction (quantization)
 * - Sample rate reduction simulation
 * - Dithering to reduce harsh artifacts
 * - Jitter for analog-style instability
 * - Hardness parameter for quantization curve
 *
 * @class Redux
 */
class Redux {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // ScriptProcessorNode for bit crushing (will be replaced by AudioWorklet in production)
    this.processor = null;
    this.bufferSize = 4096;

    // Dry/wet mix
    this.dryGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();
    this.dryGain.gain.value = 0; // 100% wet by default
    this.wetGain.gain.value = 1;

    // Parameters
    this.params = {
      bitDepth: 8,        // 1 to 16 bits
      sampleRate: 22050,  // 50 to 44100 Hz
      hardness: 50,       // 0 to 100% (quantization curve)
      dither: 0,          // 0 to 100%
      jitter: 0,          // 0 to 100%
      mix: 100            // 0-100%
    };

    // Processing state
    this.sampleCounter = 0;
    this.lastSample = [0, 0]; // For stereo

    this.setupProcessor();
    this.initialize(options);
  }

  /**
   * Setup ScriptProcessorNode for bit crushing
   * Note: In production, use AudioWorklet for better performance
   */
  setupProcessor() {
    // Use ScriptProcessorNode (deprecated but widely supported)
    // For modern browsers, use the AudioWorklet version
    this.processor = this.context.createScriptProcessor(this.bufferSize, 2, 2);

    this.processor.onaudioprocess = (e) => {
      const inputL = e.inputBuffer.getChannelData(0);
      const inputR = e.inputBuffer.numberOfChannels > 1 ? e.inputBuffer.getChannelData(1) : inputL;
      const outputL = e.outputBuffer.getChannelData(0);
      const outputR = e.outputBuffer.getChannelData(1);

      const reduction = this.context.sampleRate / this.params.sampleRate;
      const levels = Math.pow(2, this.params.bitDepth);
      const ditherAmount = this.params.dither / 100;
      const jitterAmount = this.params.jitter / 100;
      const hardness = this.params.hardness / 100;

      for (let i = 0; i < inputL.length; i++) {
        this.sampleCounter++;

        // Calculate jitter (random timing variation)
        const jitter = (Math.random() * 2 - 1) * jitterAmount * reduction;
        const threshold = reduction + jitter;

        // Sample and hold (sample rate reduction)
        if (this.sampleCounter >= threshold) {
          // Process left channel
          outputL[i] = this.processSample(inputL[i], levels, ditherAmount, hardness);
          this.lastSample[0] = outputL[i];

          // Process right channel
          outputR[i] = this.processSample(inputR[i], levels, ditherAmount, hardness);
          this.lastSample[1] = outputR[i];

          this.sampleCounter = 0;
        } else {
          // Hold previous sample
          outputL[i] = this.lastSample[0];
          outputR[i] = this.lastSample[1];
        }
      }
    };

    // Setup routing
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    this.input.connect(this.processor);
    this.processor.connect(this.wetGain);
    this.wetGain.connect(this.output);
  }

  /**
   * Process a single sample with bit crushing
   */
  processSample(sample, levels, ditherAmount, hardness) {
    // Add dither (TPDF - Triangular Probability Density Function)
    const dither1 = Math.random();
    const dither2 = Math.random();
    const tpdf = (dither1 + dither2 - 1) * ditherAmount / levels;
    const ditheredSample = sample + tpdf;

    // Quantization with hardness curve
    let quantized;
    if (hardness === 1) {
      // Hard quantization (standard)
      quantized = Math.round(ditheredSample * levels) / levels;
    } else {
      // Soft quantization (blend between original and quantized)
      const hard = Math.round(ditheredSample * levels) / levels;
      quantized = sample + (hard - sample) * hardness;
    }

    // Clamp to -1 to 1
    return Math.max(-1, Math.min(1, quantized));
  }

  /**
   * Initialize with options
   */
  initialize(options) {
    if (options.bitDepth !== undefined) this.setBitDepth(options.bitDepth);
    if (options.sampleRate !== undefined) this.setSampleRate(options.sampleRate);
    if (options.hardness !== undefined) this.setHardness(options.hardness);
    if (options.dither !== undefined) this.setDither(options.dither);
    if (options.jitter !== undefined) this.setJitter(options.jitter);
    if (options.mix !== undefined) this.setMix(options.mix);
  }

  /**
   * Set bit depth (1 to 16 bits)
   */
  setBitDepth(bits) {
    this.params.bitDepth = Math.max(1, Math.min(16, Math.round(bits)));
  }

  /**
   * Set sample rate (50 to 44100 Hz)
   */
  setSampleRate(rate) {
    this.params.sampleRate = Math.max(50, Math.min(this.context.sampleRate, rate));
  }

  /**
   * Set hardness (0 to 100%)
   * Controls the quantization curve from soft to hard
   */
  setHardness(percent) {
    this.params.hardness = Math.max(0, Math.min(100, percent));
  }

  /**
   * Set dither amount (0 to 100%)
   * Adds noise to reduce quantization artifacts
   */
  setDither(percent) {
    this.params.dither = Math.max(0, Math.min(100, percent));
  }

  /**
   * Set jitter amount (0 to 100%)
   * Adds sample timing variation for analog-style instability
   */
  setJitter(percent) {
    this.params.jitter = Math.max(0, Math.min(100, percent));
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
   * Get current parameters
   */
  getParams() {
    return { ...this.params };
  }

  /**
   * Set all parameters at once
   */
  setParams(params) {
    if (params.bitDepth !== undefined) this.setBitDepth(params.bitDepth);
    if (params.sampleRate !== undefined) this.setSampleRate(params.sampleRate);
    if (params.hardness !== undefined) this.setHardness(params.hardness);
    if (params.dither !== undefined) this.setDither(params.dither);
    if (params.jitter !== undefined) this.setJitter(params.jitter);
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
    if (this.processor) {
      this.processor.disconnect();
      this.processor.onaudioprocess = null;
    }
    this.disconnect();
    this.input.disconnect();
    this.dryGain.disconnect();
    this.wetGain.disconnect();
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Redux;
}
