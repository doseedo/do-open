/**
 * Meter Processor
 * AudioWorklet processor for level metering with RMS and peak detection
 *
 * Features:
 * - RMS (Root Mean Square) level measurement
 * - Peak level detection with configurable hold time
 * - Per-channel metering (stereo support)
 * - Configurable update rate
 * - Pass-through audio (no processing, just analysis)
 *
 * @author Agent 8 (Analyzer Plugins)
 * @version 1.0.0
 */

class MeterProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Parameters
    this.holdTime = 1.5; // seconds
    this.updateRate = 60; // updates per second

    // State per channel
    this.rmsLevel = [0, 0];
    this.peakLevel = [0, 0];
    this.peakHold = [0, 0];
    this.holdCounter = [0, 0];

    // Update timing
    this.samplesPerUpdate = Math.floor(sampleRate / this.updateRate);
    this.sampleCounter = 0;

    // RMS accumulator
    this.rmsAccumulator = [0, 0];
    this.rmsWindowSize = Math.floor(sampleRate * 0.3); // 300ms RMS window
    this.rmsSampleCount = 0;

    // Message handling
    this.port.onmessage = (e) => {
      this.handleMessage(e.data);
    };
  }

  handleMessage(data) {
    switch (data.type) {
      case 'holdTime':
        this.holdTime = Math.max(0.1, Math.min(10, data.value));
        break;

      case 'updateRate':
        this.updateRate = Math.max(1, Math.min(120, data.value));
        this.samplesPerUpdate = Math.floor(sampleRate / this.updateRate);
        break;

      case 'reset':
        // Reset all peaks
        this.peakLevel = [0, 0];
        this.peakHold = [0, 0];
        this.holdCounter = [0, 0];
        this.rmsLevel = [0, 0];
        this.rmsAccumulator = [0, 0];
        break;
    }
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    // If no input, return silence
    if (!input || !input[0]) {
      return true;
    }

    const channelCount = Math.min(input.length, 2); // Support up to stereo

    // Pass through audio unmodified while measuring
    for (let channel = 0; channel < channelCount; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];

      let sumSquares = 0;
      let peak = 0;

      // Process each sample
      for (let i = 0; i < inputChannel.length; i++) {
        const sample = inputChannel[i];

        // Pass through
        outputChannel[i] = sample;

        // Calculate RMS (sum of squares)
        sumSquares += sample * sample;

        // Find peak in this block
        const absSample = Math.abs(sample);
        if (absSample > peak) {
          peak = absSample;
        }
      }

      // Update RMS accumulator
      this.rmsAccumulator[channel] += sumSquares;
      this.rmsSampleCount += inputChannel.length;

      // Calculate RMS level for this channel
      if (this.rmsSampleCount >= this.rmsWindowSize) {
        this.rmsLevel[channel] = Math.sqrt(
          this.rmsAccumulator[channel] / this.rmsSampleCount
        );
        // Decay RMS accumulator but keep some history
        this.rmsAccumulator[channel] *= 0.5;
        this.rmsSampleCount = Math.floor(this.rmsSampleCount / 2);
      }

      // Update peak level with hold
      if (peak > this.peakLevel[channel]) {
        this.peakLevel[channel] = peak;
        this.peakHold[channel] = peak;
        this.holdCounter[channel] = this.holdTime * sampleRate;
      } else {
        // Decay hold counter
        this.holdCounter[channel] -= inputChannel.length;

        if (this.holdCounter[channel] <= 0) {
          // Reset peak hold
          this.peakHold[channel] = this.peakLevel[channel];
        }
      }

      // Fast peak decay (peaks decay quickly but hold stays)
      this.peakLevel[channel] *= Math.pow(0.999, inputChannel.length);
    }

    // Send levels to main thread periodically
    this.sampleCounter += input[0].length;
    if (this.sampleCounter >= this.samplesPerUpdate) {
      // Convert to dB for display
      const rmsDb = this.rmsLevel.map(level =>
        level > 0 ? 20 * Math.log10(level) : -Infinity
      );
      const peakDb = this.peakLevel.map(level =>
        level > 0 ? 20 * Math.log10(level) : -Infinity
      );
      const peakHoldDb = this.peakHold.map(level =>
        level > 0 ? 20 * Math.log10(level) : -Infinity
      );

      this.port.postMessage({
        type: 'meter-update',
        rms: [...this.rmsLevel],
        rmsDb: rmsDb,
        peak: [...this.peakLevel],
        peakDb: peakDb,
        peakHold: [...this.peakHold],
        peakHoldDb: peakHoldDb,
        timestamp: currentTime
      });

      this.sampleCounter = 0;
    }

    return true;
  }

  static get parameterDescriptors() {
    return [];
  }
}

registerProcessor('meter-processor', MeterProcessor);
