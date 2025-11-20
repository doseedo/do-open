/**
 * Beat Repeat Effect
 *
 * Captures and repeats slices of audio for stuttering, glitch effects.
 * Inspired by Ableton Live's Beat Repeat device.
 *
 * Features:
 * - Buffer recording at intervals
 * - Probability-based triggering (gate)
 * - Loop playback with repetition
 * - Pitch shifting per repeat
 * - Filter modulation per repeat
 * - Decay envelopes
 * - Multiple variation modes
 *
 * @author Agent 7: Creative Effects
 */

class BeatRepeat {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = this.context.createGain();
    this.output = this.context.createGain();
    this.dryGain = this.context.createGain();
    this.wetGain = this.context.createGain();

    // Recording buffer (4 seconds)
    this.bufferDuration = 4;
    this.recordBuffer = this.context.createBuffer(
      2,
      this.bufferDuration * this.context.sampleRate,
      this.context.sampleRate
    );

    // Parameters with defaults
    this.params = {
      interval: 1.0,           // seconds (1/32 to 4 bars)
      offset: 0,               // 0 to 100% offset from interval start
      gate: 50,                // 0 to 100% probability of triggering
      variation: 'trigger',    // off, trigger, loop, reverse
      repeat: 4,               // 1 to 32 repetitions
      grid: 0.25,              // 1/32 to 1 bar (slice length)
      decay: 90,               // 0 to 100% volume decay per repeat
      pitch: 0,                // -24 to +24 semitones
      pitchDecay: 0,           // -24 to +24 semitones per repeat
      filterFreq: 20000,       // 20 Hz to 20 kHz
      filterDecay: 0,          // 0 to 100% filter decay
      volume: 100,             // 0 to 100%
      mix: 50                  // 0 to 100% dry/wet
    };

    // State
    this.writePosition = 0;
    this.lastTriggerTime = 0;
    this.isRecording = false;
    this.processor = null;
    this.activeSlices = [];

    // Initialize
    this.setupRouting();
    this.setupRecording();

    // Apply user options
    if (options) {
      Object.keys(options).forEach(key => {
        if (this.params.hasOwnProperty(key)) {
          this.params[key] = options[key];
        }
      });
    }

    this.updateMix();
  }

  /**
   * Setup audio routing
   */
  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path will be created dynamically for each slice
    this.wetGain.connect(this.output);
  }

  /**
   * Setup ScriptProcessor for buffer recording
   * Note: In production, use AudioWorklet for better performance
   */
  setupRecording() {
    const bufferSize = 4096;
    this.processor = this.context.createScriptProcessor(bufferSize, 2, 2);

    this.processor.onaudioprocess = (e) => {
      const inputL = e.inputBuffer.getChannelData(0);
      const inputR = e.inputBuffer.getChannelData(1);
      const outputL = e.outputBuffer.getChannelData(0);
      const outputR = e.outputBuffer.getChannelData(1);

      // Copy input to output (pass-through)
      outputL.set(inputL);
      outputR.set(inputR);

      // Record to circular buffer
      const bufferL = this.recordBuffer.getChannelData(0);
      const bufferR = this.recordBuffer.getChannelData(1);

      for (let i = 0; i < inputL.length; i++) {
        bufferL[this.writePosition] = inputL[i];
        bufferR[this.writePosition] = inputR[i];

        this.writePosition++;
        if (this.writePosition >= this.recordBuffer.length) {
          this.writePosition = 0;
        }
      }

      // Check if we should trigger repeat
      if (this.isRecording) {
        this.checkTrigger();
      }
    };

    // Connect processor
    this.input.connect(this.processor);
    this.processor.connect(this.context.createGain()); // Connect to dummy node (required)

    this.isRecording = true;
  }

  /**
   * Check if it's time to trigger a beat repeat
   */
  checkTrigger() {
    const now = this.context.currentTime;

    if (now - this.lastTriggerTime >= this.params.interval) {
      // Apply probability gate
      if (Math.random() * 100 < this.params.gate) {
        this.triggerRepeat();
      }

      this.lastTriggerTime = now;
    }
  }

  /**
   * Trigger a beat repeat sequence
   */
  triggerRepeat() {
    const sliceDuration = this.params.grid;
    const sliceLength = Math.floor(sliceDuration * this.context.sampleRate);

    // Calculate start position in buffer (with offset)
    const offsetSamples = Math.floor((this.params.offset / 100) * sliceLength);
    let startPos = this.writePosition - sliceLength - offsetSamples;

    if (startPos < 0) {
      startPos += this.recordBuffer.length;
    }

    // Create slice buffer
    const sliceBuffer = this.createSliceBuffer(startPos, sliceLength);

    // Apply variation mode
    if (this.params.variation === 'reverse') {
      this.reverseBuffer(sliceBuffer);
    }

    // Create repetitions
    for (let i = 0; i < this.params.repeat; i++) {
      const delay = i * sliceDuration;
      const volume = this.calculateVolume(i);
      const pitch = this.calculatePitch(i);
      const filterFreq = this.calculateFilterFreq(i);

      this.playSlice(sliceBuffer, delay, volume, pitch, filterFreq, sliceDuration);
    }
  }

  /**
   * Create a buffer slice from the record buffer
   */
  createSliceBuffer(startPos, length) {
    const sliceBuffer = this.context.createBuffer(2, length, this.context.sampleRate);
    const sliceL = sliceBuffer.getChannelData(0);
    const sliceR = sliceBuffer.getChannelData(1);
    const bufferL = this.recordBuffer.getChannelData(0);
    const bufferR = this.recordBuffer.getChannelData(1);

    // Copy from circular buffer
    for (let i = 0; i < length; i++) {
      const readPos = (startPos + i) % this.recordBuffer.length;
      sliceL[i] = bufferL[readPos];
      sliceR[i] = bufferR[readPos];
    }

    return sliceBuffer;
  }

  /**
   * Reverse a buffer in place
   */
  reverseBuffer(buffer) {
    for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
      const data = buffer.getChannelData(channel);
      const reversed = Array.from(data).reverse();
      data.set(reversed);
    }
  }

  /**
   * Calculate volume for repeat iteration
   */
  calculateVolume(iteration) {
    const baseVolume = this.params.volume / 100;
    const decayFactor = Math.pow(this.params.decay / 100, iteration);
    return baseVolume * decayFactor;
  }

  /**
   * Calculate pitch for repeat iteration
   */
  calculatePitch(iteration) {
    const totalPitch = this.params.pitch + (this.params.pitchDecay * iteration);
    return Math.pow(2, totalPitch / 12);
  }

  /**
   * Calculate filter frequency for repeat iteration
   */
  calculateFilterFreq(iteration) {
    if (this.params.filterDecay === 0) {
      return this.params.filterFreq;
    }

    const decayAmount = (this.params.filterDecay / 100) * iteration;
    const freq = this.params.filterFreq * Math.pow(0.5, decayAmount);
    return Math.max(20, Math.min(20000, freq));
  }

  /**
   * Play a single slice with effects
   */
  playSlice(buffer, delay, volume, playbackRate, filterFreq, duration) {
    const now = this.context.currentTime;
    const startTime = now + delay;

    // Create buffer source
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.playbackRate.value = playbackRate;

    // Create gain for this slice
    const gain = this.context.createGain();
    gain.gain.value = volume;

    // Create filter for this slice
    const filter = this.context.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = filterFreq;
    filter.Q.value = 1.0;

    // Create envelope for smooth playback
    const envelope = this.context.createGain();
    envelope.gain.setValueAtTime(0, startTime);
    envelope.gain.linearRampToValueAtTime(1, startTime + 0.005); // 5ms attack
    envelope.gain.setValueAtTime(1, startTime + duration - 0.01);
    envelope.gain.linearRampToValueAtTime(0, startTime + duration); // 10ms release

    // Connect: source → filter → gain → envelope → wetGain → output
    source.connect(filter);
    filter.connect(gain);
    gain.connect(envelope);
    envelope.connect(this.wetGain);

    // Start playback
    source.start(startTime);
    source.stop(startTime + duration);

    // Store reference for cleanup
    const sliceData = { source, gain, filter, envelope, stopTime: startTime + duration };
    this.activeSlices.push(sliceData);

    // Cleanup after playback
    source.onended = () => {
      this.cleanupSlice(sliceData);
    };
  }

  /**
   * Cleanup finished slice
   */
  cleanupSlice(sliceData) {
    try {
      sliceData.source.disconnect();
      sliceData.filter.disconnect();
      sliceData.gain.disconnect();
      sliceData.envelope.disconnect();
    } catch (e) {
      // Already disconnected
    }

    const index = this.activeSlices.indexOf(sliceData);
    if (index > -1) {
      this.activeSlices.splice(index, 1);
    }
  }

  /**
   * Update dry/wet mix
   */
  updateMix() {
    const wet = this.params.mix / 100;
    const dry = 1 - wet;

    this.dryGain.gain.value = dry;
    this.wetGain.gain.value = wet;
  }

  // ========== Public Parameter Methods ==========

  setInterval(seconds) {
    this.params.interval = Math.max(0.03125, Math.min(16, seconds)); // 1/32 to 4 bars
  }

  setOffset(percent) {
    this.params.offset = Math.max(0, Math.min(100, percent));
  }

  setGate(percent) {
    this.params.gate = Math.max(0, Math.min(100, percent));
  }

  setVariation(mode) {
    const validModes = ['off', 'trigger', 'loop', 'reverse'];
    if (validModes.includes(mode)) {
      this.params.variation = mode;
    }
  }

  setRepeat(count) {
    this.params.repeat = Math.max(1, Math.min(32, Math.floor(count)));
  }

  setGrid(seconds) {
    this.params.grid = Math.max(0.03125, Math.min(4, seconds)); // 1/32 to 4 bars
  }

  setDecay(percent) {
    this.params.decay = Math.max(0, Math.min(100, percent));
  }

  setPitch(semitones) {
    this.params.pitch = Math.max(-24, Math.min(24, semitones));
  }

  setPitchDecay(semitones) {
    this.params.pitchDecay = Math.max(-24, Math.min(24, semitones));
  }

  setFilterFreq(hz) {
    this.params.filterFreq = Math.max(20, Math.min(20000, hz));
  }

  setFilterDecay(percent) {
    this.params.filterDecay = Math.max(0, Math.min(100, percent));
  }

  setVolume(percent) {
    this.params.volume = Math.max(0, Math.min(100, percent));
  }

  setMix(percent) {
    this.params.mix = Math.max(0, Math.min(100, percent));
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
    // Stop all active slices
    this.activeSlices.forEach(slice => {
      try {
        slice.source.stop();
        this.cleanupSlice(slice);
      } catch (e) {
        // Already stopped
      }
    });

    this.activeSlices = [];

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
    this.output.disconnect();

    this.isRecording = false;
  }
}

export default BeatRepeat;
