/**
 * Oscilloscope Processor
 * AudioWorklet processor for real-time waveform capture and display
 *
 * Features:
 * - Waveform capture with configurable buffer size
 * - Trigger modes (auto, normal, single)
 * - Configurable trigger level and edge
 * - Dual channel support (X-Y mode or dual trace)
 * - Configurable update rate
 * - Pass-through audio (no processing, just analysis)
 *
 * @author Agent 8 (Analyzer Plugins)
 * @version 1.0.0
 */

class OscilloscopeProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Parameters
    this.bufferSize = 2048;
    this.updateRate = 60; // updates per second
    this.triggerMode = 'auto'; // 'auto', 'normal', 'single'
    this.triggerLevel = 0.0; // -1 to 1
    this.triggerEdge = 'rising'; // 'rising', 'falling'
    this.triggerChannel = 0; // 0 or 1

    // Waveform buffers (stereo)
    this.waveformBufferL = new Float32Array(this.bufferSize);
    this.waveformBufferR = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;

    // Trigger state
    this.triggered = false;
    this.triggerPosition = 0;
    this.lastSample = 0;
    this.singleTriggerFired = false;

    // Update timing
    this.samplesPerUpdate = Math.floor(sampleRate / this.updateRate);
    this.sampleCounter = 0;
    this.updateCounter = 0;

    // Auto trigger timeout
    this.autoTriggerTimeout = sampleRate / 10; // 100ms
    this.autoTriggerCounter = 0;

    // Message handling
    this.port.onmessage = (e) => {
      this.handleMessage(e.data);
    };
  }

  handleMessage(data) {
    switch (data.type) {
      case 'bufferSize':
        this.setBufferSize(data.value);
        break;

      case 'updateRate':
        this.updateRate = Math.max(1, Math.min(120, data.value));
        this.samplesPerUpdate = Math.floor(sampleRate / this.updateRate);
        break;

      case 'triggerMode':
        this.triggerMode = data.value;
        this.singleTriggerFired = false;
        break;

      case 'triggerLevel':
        this.triggerLevel = Math.max(-1, Math.min(1, data.value));
        break;

      case 'triggerEdge':
        this.triggerEdge = data.value;
        break;

      case 'triggerChannel':
        this.triggerChannel = data.value;
        break;

      case 'reset':
        this.triggered = false;
        this.bufferIndex = 0;
        this.singleTriggerFired = false;
        break;
    }
  }

  /**
   * Set buffer size
   */
  setBufferSize(size) {
    // Clamp to reasonable range
    size = Math.max(128, Math.min(16384, size));

    this.bufferSize = size;
    this.waveformBufferL = new Float32Array(size);
    this.waveformBufferR = new Float32Array(size);
    this.bufferIndex = 0;
    this.triggered = false;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    // If no input, return silence
    if (!input || !input[0]) {
      return true;
    }

    const channelCount = Math.min(input.length, output.length, 2);

    // Pass through audio unmodified
    for (let channel = 0; channel < channelCount; channel++) {
      output[channel].set(input[channel]);
    }

    const inputL = input[0];
    const inputR = input.length > 1 ? input[1] : input[0];

    // Process each sample
    for (let i = 0; i < inputL.length; i++) {
      const sampleL = inputL[i];
      const sampleR = inputR[i];

      // Determine trigger sample
      const triggerSample = this.triggerChannel === 0 ? sampleL : sampleR;

      // Check for trigger condition
      if (!this.triggered) {
        const triggered = this.checkTrigger(triggerSample);

        if (triggered) {
          this.triggered = true;
          this.triggerPosition = this.bufferIndex;
          this.autoTriggerCounter = 0;

          if (this.triggerMode === 'single') {
            this.singleTriggerFired = true;
          }
        } else {
          // Auto trigger timeout
          this.autoTriggerCounter++;
          if (this.triggerMode === 'auto' && this.autoTriggerCounter >= this.autoTriggerTimeout) {
            this.triggered = true;
            this.triggerPosition = this.bufferIndex;
            this.autoTriggerCounter = 0;
          }
        }
      }

      // Capture waveform
      if (this.triggered && !(this.triggerMode === 'single' && this.singleTriggerFired && this.bufferIndex >= this.bufferSize)) {
        this.waveformBufferL[this.bufferIndex] = sampleL;
        this.waveformBufferR[this.bufferIndex] = sampleR;
        this.bufferIndex++;

        // Buffer filled
        if (this.bufferIndex >= this.bufferSize) {
          this.bufferIndex = 0;

          // For normal and auto modes, reset trigger
          if (this.triggerMode !== 'single') {
            this.triggered = false;
          }
        }
      }

      this.lastSample = triggerSample;
    }

    // Send waveform data periodically
    this.sampleCounter += inputL.length;
    if (this.sampleCounter >= this.samplesPerUpdate) {
      // Only send if we have valid data
      if (this.bufferIndex > 0 || this.updateCounter > 0) {
        this.port.postMessage({
          type: 'waveform-update',
          waveformL: Array.from(this.waveformBufferL),
          waveformR: Array.from(this.waveformBufferR),
          bufferSize: this.bufferSize,
          triggerPosition: this.triggerPosition,
          triggered: this.triggered,
          timestamp: currentTime
        });
        this.updateCounter++;
      }

      this.sampleCounter = 0;
    }

    return true;
  }

  /**
   * Check if trigger condition is met
   */
  checkTrigger(sample) {
    if (this.triggerEdge === 'rising') {
      // Rising edge: last sample below threshold, current sample above
      return this.lastSample < this.triggerLevel && sample >= this.triggerLevel;
    } else {
      // Falling edge: last sample above threshold, current sample below
      return this.lastSample > this.triggerLevel && sample <= this.triggerLevel;
    }
  }

  static get parameterDescriptors() {
    return [];
  }
}

registerProcessor('oscilloscope-processor', OscilloscopeProcessor);
