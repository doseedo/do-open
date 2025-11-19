/**
 * Grain Delay Effect
 *
 * Granular synthesis-based delay effect for creating evolving textures and ambient soundscapes.
 * Inspired by Ableton Live's Grain Delay device.
 *
 * Features:
 * - Granular buffer playback
 * - Random grain positioning (spray)
 * - Pitch shifting per grain
 * - Density control (grain frequency)
 * - Feedback for evolving textures
 * - Time and pitch spray for organic variation
 *
 * @author Agent 7: Creative Effects
 */

class GrainDelay {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = this.context.createGain();
    this.output = this.context.createGain();
    this.dryGain = this.context.createGain();
    this.wetGain = this.context.createGain();
    this.feedbackGain = this.context.createGain();

    // Delay buffer (5 seconds max)
    this.maxDelayTime = 5;
    this.delayBuffer = this.context.createBuffer(
      2,
      this.maxDelayTime * this.context.sampleRate,
      this.context.sampleRate
    );

    // Parameters with defaults
    this.params = {
      frequency: 10,           // 0.1 to 100 Hz - grain triggering rate
      spray: 0,                // 0 to 100% - random delay time variation
      pitch: 0,                // -24 to +24 semitones - grain pitch shift
      pitchSpray: 0,           // 0 to 100% - random pitch variation
      grainSize: 100,          // 10 to 500 ms - individual grain length
      feedback: 0,             // 0 to 100% - feedback amount
      delayTime: 500,          // 0 to 5000 ms or sync
      dryWet: 50               // 0 to 100% - dry/wet mix
    };

    // State
    this.writePosition = 0;
    this.grainScheduler = null;
    this.activeGrains = [];
    this.processor = null;
    this.isRunning = false;

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
    this.updateFeedback();
    this.startGrainScheduler();
  }

  /**
   * Setup audio routing
   */
  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path (grains will connect here)
    this.wetGain.connect(this.output);

    // Feedback path
    this.wetGain.connect(this.feedbackGain);
    this.feedbackGain.connect(this.input);
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
      const bufferL = this.delayBuffer.getChannelData(0);
      const bufferR = this.delayBuffer.getChannelData(1);

      for (let i = 0; i < inputL.length; i++) {
        bufferL[this.writePosition] = inputL[i];
        bufferR[this.writePosition] = inputR[i];

        this.writePosition++;
        if (this.writePosition >= this.delayBuffer.length) {
          this.writePosition = 0;
        }
      }
    };

    // Connect processor
    this.input.connect(this.processor);
    this.processor.connect(this.context.createGain()); // Connect to dummy node

    this.isRunning = true;
  }

  /**
   * Start grain scheduler
   */
  startGrainScheduler() {
    if (this.grainScheduler) {
      clearInterval(this.grainScheduler);
    }

    const scheduleGrain = () => {
      if (this.isRunning) {
        this.scheduleNextGrain();
      }
    };

    // Schedule grains based on frequency parameter
    const updateScheduler = () => {
      if (this.grainScheduler) {
        clearInterval(this.grainScheduler);
      }

      const interval = 1000 / this.params.frequency; // Convert Hz to ms
      this.grainScheduler = setInterval(scheduleGrain, interval);
    };

    updateScheduler();

    // Store reference to update function
    this._updateScheduler = updateScheduler;
  }

  /**
   * Schedule and play the next grain
   */
  scheduleNextGrain() {
    // Calculate delay time with spray
    const baseDelayMs = this.params.delayTime;
    const sprayRange = (this.params.spray / 100) * baseDelayMs;
    const randomSpray = (Math.random() - 0.5) * 2 * sprayRange;
    const delayMs = Math.max(0, baseDelayMs + randomSpray);
    const delaySamples = Math.floor((delayMs / 1000) * this.context.sampleRate);

    // Calculate pitch with spray
    const basePitch = this.params.pitch;
    const pitchSprayRange = (this.params.pitchSpray / 100) * 24; // Max 24 semitones spray
    const randomPitchSpray = (Math.random() - 0.5) * 2 * pitchSprayRange;
    const totalPitch = basePitch + randomPitchSpray;
    const playbackRate = Math.pow(2, totalPitch / 12);

    // Calculate read position in buffer
    let readPos = this.writePosition - delaySamples;
    if (readPos < 0) {
      readPos += this.delayBuffer.length;
    }

    // Play grain
    const grainDuration = this.params.grainSize / 1000; // Convert ms to seconds
    this.playGrain(readPos, grainDuration, playbackRate);
  }

  /**
   * Play a single grain
   */
  playGrain(readPosition, duration, playbackRate) {
    const now = this.context.currentTime;
    const grainLength = Math.floor(duration * this.context.sampleRate * playbackRate);

    // Create grain buffer
    const grainBuffer = this.extractGrain(readPosition, grainLength, duration);

    if (!grainBuffer) {
      return; // Failed to create grain
    }

    // Create buffer source
    const source = this.context.createBufferSource();
    source.buffer = grainBuffer;
    source.playbackRate.value = playbackRate;

    // Create envelope (Hann window for smooth grain)
    const envelope = this.context.createGain();
    this.applyHannWindow(envelope, now, duration);

    // Connect: source → envelope → wetGain → output
    source.connect(envelope);
    envelope.connect(this.wetGain);

    // Start playback
    source.start(now);
    source.stop(now + duration);

    // Store reference
    const grainData = { source, envelope, stopTime: now + duration };
    this.activeGrains.push(grainData);

    // Cleanup after playback
    source.onended = () => {
      this.cleanupGrain(grainData);
    };
  }

  /**
   * Extract a grain from the delay buffer
   */
  extractGrain(readPosition, length, targetDuration) {
    try {
      const grainBuffer = this.context.createBuffer(
        2,
        Math.floor(targetDuration * this.context.sampleRate),
        this.context.sampleRate
      );

      const grainL = grainBuffer.getChannelData(0);
      const grainR = grainBuffer.getChannelData(1);
      const bufferL = this.delayBuffer.getChannelData(0);
      const bufferR = this.delayBuffer.getChannelData(1);

      // Copy from circular buffer
      for (let i = 0; i < grainL.length; i++) {
        const sourceIndex = Math.floor(i * (length / grainL.length));
        const readPos = (readPosition + sourceIndex) % this.delayBuffer.length;
        grainL[i] = bufferL[readPos];
        grainR[i] = bufferR[readPos];
      }

      return grainBuffer;
    } catch (e) {
      console.error('Failed to create grain buffer:', e);
      return null;
    }
  }

  /**
   * Apply Hann window envelope to grain
   */
  applyHannWindow(gainNode, startTime, duration) {
    const gain = gainNode.gain;

    // Hann window formula: 0.5 * (1 - cos(2π * n / N))
    // Simplified to exponential curves for Web Audio
    gain.setValueAtTime(0, startTime);
    gain.linearRampToValueAtTime(1, startTime + duration * 0.5);
    gain.linearRampToValueAtTime(0, startTime + duration);
  }

  /**
   * Cleanup finished grain
   */
  cleanupGrain(grainData) {
    try {
      grainData.source.disconnect();
      grainData.envelope.disconnect();
    } catch (e) {
      // Already disconnected
    }

    const index = this.activeGrains.indexOf(grainData);
    if (index > -1) {
      this.activeGrains.splice(index, 1);
    }
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

  /**
   * Update feedback amount
   */
  updateFeedback() {
    // Limit feedback to prevent runaway
    const feedback = Math.min(0.95, this.params.feedback / 100);
    this.feedbackGain.gain.value = feedback;
  }

  // ========== Public Parameter Methods ==========

  setFrequency(hz) {
    this.params.frequency = Math.max(0.1, Math.min(100, hz));
    if (this._updateScheduler) {
      this._updateScheduler();
    }
  }

  setSpray(percent) {
    this.params.spray = Math.max(0, Math.min(100, percent));
  }

  setPitch(semitones) {
    this.params.pitch = Math.max(-24, Math.min(24, semitones));
  }

  setPitchSpray(percent) {
    this.params.pitchSpray = Math.max(0, Math.min(100, percent));
  }

  setGrainSize(ms) {
    this.params.grainSize = Math.max(10, Math.min(500, ms));
  }

  setFeedback(percent) {
    this.params.feedback = Math.max(0, Math.min(100, percent));
    this.updateFeedback();
  }

  setDelayTime(ms) {
    this.params.delayTime = Math.max(0, Math.min(5000, ms));
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
    this.isRunning = false;

    // Stop grain scheduler
    if (this.grainScheduler) {
      clearInterval(this.grainScheduler);
      this.grainScheduler = null;
    }

    // Stop all active grains
    this.activeGrains.forEach(grain => {
      try {
        grain.source.stop();
        this.cleanupGrain(grain);
      } catch (e) {
        // Already stopped
      }
    });

    this.activeGrains = [];

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
    this.feedbackGain.disconnect();
    this.output.disconnect();
  }
}

export default GrainDelay;
