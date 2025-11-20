/**
 * Granular Synthesis Audio Worklet Processor
 *
 * High-performance granular synthesis processor for Web Audio API.
 * Runs on audio rendering thread for low-latency grain playback.
 *
 * Features:
 * - Real-time grain generation and playback
 * - Windowing (Hann, Hamming, Gaussian)
 * - Pitch shifting via grain playback rate
 * - Density control
 * - Position and pitch spray (randomization)
 *
 * Usage:
 * await audioContext.audioWorklet.addModule('granular-processor.js');
 * const granularNode = new AudioWorkletNode(audioContext, 'granular-processor');
 *
 * @author Agent 7: Creative Effects
 */

class GranularProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Grain parameters
    this.grainSize = 0.1; // seconds
    this.grainRate = 10; // grains per second
    this.grainPitch = 1.0; // playback rate
    this.grainSpray = 0.0; // position randomization (0-1)
    this.pitchSpray = 0.0; // pitch randomization (0-1)
    this.windowType = 'hann'; // hann, hamming, gaussian

    // Delay buffer (5 seconds)
    this.maxDelayTime = 5.0;
    this.bufferSize = Math.floor(this.maxDelayTime * sampleRate);
    this.delayBuffer = new Float32Array(this.bufferSize);
    this.writePosition = 0;

    // Grain scheduling
    this.samplesSinceLastGrain = 0;
    this.samplesPerGrain = sampleRate / this.grainRate;

    // Active grains (max 64 concurrent)
    this.maxGrains = 64;
    this.activeGrains = [];

    // Message handling
    this.port.onmessage = (event) => {
      this.handleMessage(event.data);
    };
  }

  /**
   * Handle parameter messages from main thread
   */
  handleMessage(data) {
    switch (data.type) {
      case 'grainSize':
        this.grainSize = data.value;
        break;
      case 'grainRate':
        this.grainRate = data.value;
        this.samplesPerGrain = sampleRate / this.grainRate;
        break;
      case 'grainPitch':
        this.grainPitch = data.value;
        break;
      case 'grainSpray':
        this.grainSpray = data.value;
        break;
      case 'pitchSpray':
        this.pitchSpray = data.value;
        break;
      case 'windowType':
        this.windowType = data.value;
        break;
    }
  }

  /**
   * Main audio processing function
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    if (!input || !input[0] || !output || !output[0]) {
      return true;
    }

    const inputChannel = input[0];
    const outputChannel = output[0];
    const blockSize = outputChannel.length;

    // Write input to delay buffer
    for (let i = 0; i < blockSize; i++) {
      this.delayBuffer[this.writePosition] = inputChannel[i];
      this.writePosition = (this.writePosition + 1) % this.bufferSize;
    }

    // Clear output
    outputChannel.fill(0);

    // Schedule new grains
    this.samplesSinceLastGrain += blockSize;
    while (this.samplesSinceLastGrain >= this.samplesPerGrain) {
      this.scheduleGrain();
      this.samplesSinceLastGrain -= this.samplesPerGrain;
    }

    // Render active grains
    this.renderGrains(outputChannel);

    return true;
  }

  /**
   * Schedule a new grain
   */
  scheduleGrain() {
    if (this.activeGrains.length >= this.maxGrains) {
      return; // Too many active grains
    }

    // Calculate grain parameters with spray
    const sprayOffset = (Math.random() - 0.5) * 2 * this.grainSpray;
    const grainDelay = 0.5 + sprayOffset; // Delay time in seconds

    const pitchVariation = (Math.random() - 0.5) * 2 * this.pitchSpray;
    const grainPitch = this.grainPitch * (1 + pitchVariation);

    // Calculate read position
    const delaySamples = Math.floor(grainDelay * sampleRate);
    let readPosition = this.writePosition - delaySamples;
    if (readPosition < 0) {
      readPosition += this.bufferSize;
    }

    // Create grain
    const grain = {
      readPosition: readPosition,
      playbackRate: grainPitch,
      currentSample: 0,
      totalSamples: Math.floor(this.grainSize * sampleRate)
    };

    this.activeGrains.push(grain);
  }

  /**
   * Render all active grains to output buffer
   */
  renderGrains(outputBuffer) {
    const blockSize = outputBuffer.length;
    const grainsToRemove = [];

    for (let g = 0; g < this.activeGrains.length; g++) {
      const grain = this.activeGrains[g];

      for (let i = 0; i < blockSize; i++) {
        if (grain.currentSample >= grain.totalSamples) {
          grainsToRemove.push(g);
          break;
        }

        // Calculate read position with pitch shift
        const sourcePosition = grain.currentSample * grain.playbackRate;
        const sourceIndex = Math.floor(sourcePosition);

        if (sourceIndex >= grain.totalSamples) {
          grainsToRemove.push(g);
          break;
        }

        // Read from delay buffer with interpolation
        const bufferPosition = (grain.readPosition + sourceIndex) % this.bufferSize;
        const sample = this.delayBuffer[bufferPosition];

        // Apply window function
        const windowValue = this.getWindowValue(grain.currentSample, grain.totalSamples);

        // Add to output
        outputBuffer[i] += sample * windowValue;

        grain.currentSample++;
      }
    }

    // Remove finished grains (in reverse order to maintain indices)
    for (let i = grainsToRemove.length - 1; i >= 0; i--) {
      this.activeGrains.splice(grainsToRemove[i], 1);
    }
  }

  /**
   * Calculate window function value
   */
  getWindowValue(sampleIndex, totalSamples) {
    const phase = sampleIndex / totalSamples; // 0 to 1

    switch (this.windowType) {
      case 'hann':
        return 0.5 * (1 - Math.cos(2 * Math.PI * phase));

      case 'hamming':
        return 0.54 - 0.46 * Math.cos(2 * Math.PI * phase);

      case 'gaussian':
        const sigma = 0.4;
        const x = (phase - 0.5) / sigma;
        return Math.exp(-0.5 * x * x);

      case 'triangle':
        return phase < 0.5 ? phase * 2 : (1 - phase) * 2;

      default:
        return 1.0; // Rectangular window
    }
  }
}

// Register the processor
registerProcessor('granular-processor', GranularProcessor);
