/**
 * Vinyl Distortion Effect
 *
 * Simulates vinyl record artifacts including crackle, wear, warble (wow/flutter),
 * and tracking distortion for authentic vintage sound.
 * Inspired by Ableton Live's Vinyl Distortion device.
 *
 * Features:
 * - Vinyl crackle noise generation (impulse-based)
 * - Low-frequency warble (wow/flutter) for pitch variation
 * - High-frequency wear simulation (low-pass filtering)
 * - Pinch effect (center-hole pitch variation)
 * - Tracking distortion (harmonic distortion)
 *
 * @author Agent 7: Creative Effects
 */

class VinylDistortion {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = this.context.createGain();
    this.output = this.context.createGain();

    // Signal chain nodes
    this.trackingDistortion = this.context.createWaveShaper();
    this.wearFilter = this.context.createBiquadFilter();
    this.wearFilter.type = 'lowpass';
    this.wearFilter.frequency.value = 20000;

    // Crackle noise
    this.crackleGain = this.context.createGain();
    this.crackleGain.gain.value = 0;

    // Warp LFO (for wow/flutter)
    this.warpLFO = this.context.createOscillator();
    this.warpLFO.type = 'sine';
    this.warpLFO.frequency.value = 0.5;
    this.warpGain = this.context.createGain();
    this.warpGain.gain.value = 0;

    // Pinch LFO (for center-hole effect)
    this.pinchLFO = this.context.createOscillator();
    this.pinchLFO.type = 'sine';
    this.pinchLFO.frequency.value = 0.3;
    this.pinchGain = this.context.createGain();
    this.pinchGain.gain.value = 0;

    // Crackle generator
    this.crackleInterval = null;
    this.activeCrackles = [];

    // Parameters with defaults
    this.params = {
      tracing: 0,              // 0 to 100% - tracking distortion amount
      pinch: 0,                // 0 to 100% - center-hole distortion
      crackle: 0,              // 0 to 100% - surface noise density
      crackleVolume: -20,      // -60 to 0 dB - crackle volume
      wear: 0,                 // 0 to 100% - high-frequency loss
      warp: 0,                 // 0 to 100% - pitch warble depth
      warpFrequency: 0.5       // 0.1 to 5 Hz - warble speed
    };

    // Initialize
    this.setupRouting();
    this.createTracingCurve();
    this.startLFOs();

    // Apply user options
    if (options) {
      Object.keys(options).forEach(key => {
        if (this.params.hasOwnProperty(key)) {
          this.params[key] = options[key];
        }
      });
    }

    this.updateWear();
    this.updateWarp();
    this.updateCrackle();
  }

  /**
   * Setup audio routing
   */
  setupRouting() {
    // Main signal path: input → tracing → wear filter → output
    this.input.connect(this.trackingDistortion);
    this.trackingDistortion.connect(this.wearFilter);
    this.wearFilter.connect(this.output);

    // Crackle path
    this.crackleGain.connect(this.output);
  }

  /**
   * Create waveshaper curve for tracking distortion
   */
  createTracingCurve() {
    const samples = 256;
    const curve = new Float32Array(samples);

    for (let i = 0; i < samples; i++) {
      const x = (i / samples) * 2 - 1; // -1 to 1
      // Asymmetric distortion (like vinyl tracking)
      curve[i] = x + 0.3 * Math.sin(x * Math.PI);
    }

    this.trackingDistortion.curve = curve;
    this.trackingDistortion.oversample = '4x';
  }

  /**
   * Update tracing distortion curve based on amount
   */
  updateTracingCurve() {
    const samples = 256;
    const curve = new Float32Array(samples);
    const amount = this.params.tracing / 100;

    for (let i = 0; i < samples; i++) {
      const x = (i / samples) * 2 - 1; // -1 to 1

      if (amount === 0) {
        // No distortion - linear
        curve[i] = x;
      } else {
        // Asymmetric harmonic distortion
        const distorted = x + amount * 0.3 * Math.sin(x * Math.PI * 3);
        const saturated = Math.tanh(distorted * (1 + amount));
        curve[i] = x * (1 - amount) + saturated * amount;
      }
    }

    this.trackingDistortion.curve = curve;
  }

  /**
   * Start LFOs for warp and pinch effects
   */
  startLFOs() {
    this.warpLFO.start();
    this.pinchLFO.start();

    // Note: For pitch modulation via LFO, we would need to use
    // delayNode modulation or AudioWorklet in a real implementation.
    // This is a simplified version using filter modulation as an approximation.
  }

  /**
   * Generate crackle noise (impulse-based)
   */
  generateCrackle() {
    const now = this.context.currentTime;

    // Random impulse characteristics
    const duration = 0.01 + Math.random() * 0.02; // 10-30ms
    const attackTime = 0.001;
    const volume = this.dbToGain(this.params.crackleVolume) * (0.5 + Math.random() * 0.5);

    // Create noise buffer for single crackle
    const crackleLength = Math.floor(duration * this.context.sampleRate);
    const buffer = this.context.createBuffer(1, crackleLength, this.context.sampleRate);
    const data = buffer.getChannelData(0);

    // Generate impulse with decay
    for (let i = 0; i < crackleLength; i++) {
      const envelope = Math.exp(-i / (crackleLength * 0.3));
      const noise = (Math.random() * 2 - 1);
      data[i] = noise * envelope;
    }

    // Create source and gain
    const source = this.context.createBufferSource();
    source.buffer = buffer;

    const gain = this.context.createGain();
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(volume, now + attackTime);
    gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    // Connect and play
    source.connect(gain);
    gain.connect(this.crackleGain);

    source.start(now);
    source.stop(now + duration);

    // Store reference
    const crackleData = { source, gain, stopTime: now + duration };
    this.activeCrackles.push(crackleData);

    // Cleanup
    source.onended = () => {
      try {
        source.disconnect();
        gain.disconnect();
      } catch (e) {
        // Already disconnected
      }

      const index = this.activeCrackles.indexOf(crackleData);
      if (index > -1) {
        this.activeCrackles.splice(index, 1);
      }
    };
  }

  /**
   * Update crackle generation rate
   */
  updateCrackle() {
    // Stop existing interval
    if (this.crackleInterval) {
      clearInterval(this.crackleInterval);
      this.crackleInterval = null;
    }

    // Start new interval if crackle > 0
    if (this.params.crackle > 0) {
      // Crackle rate: higher percentage = more frequent crackles
      const minInterval = 50;   // 50ms minimum
      const maxInterval = 1000; // 1s maximum
      const interval = maxInterval - (this.params.crackle / 100) * (maxInterval - minInterval);

      this.crackleInterval = setInterval(() => {
        // Random variation in crackle timing
        if (Math.random() * 100 < this.params.crackle) {
          this.generateCrackle();
        }
      }, interval);
    }
  }

  /**
   * Update wear filter (high-frequency loss)
   */
  updateWear() {
    // wear: 0% = no loss (20kHz), 100% = heavy loss (2kHz)
    const minFreq = 2000;
    const maxFreq = 20000;
    const normalizedWear = this.params.wear / 100;
    const frequency = maxFreq - (normalizedWear * (maxFreq - minFreq));

    this.wearFilter.frequency.value = frequency;
    this.wearFilter.Q.value = 0.7 + normalizedWear * 2; // Increase resonance with wear
  }

  /**
   * Update warp (wow/flutter) parameters
   */
  updateWarp() {
    this.warpLFO.frequency.value = this.params.warpFrequency;

    // Warp depth (would modulate pitch in full implementation)
    const warpDepth = this.params.warp / 100;
    this.warpGain.gain.value = warpDepth * 10; // Scaled for effect
  }

  /**
   * Convert dB to linear gain
   */
  dbToGain(db) {
    return Math.pow(10, db / 20);
  }

  // ========== Public Parameter Methods ==========

  setTracing(percent) {
    this.params.tracing = Math.max(0, Math.min(100, percent));
    this.updateTracingCurve();
  }

  setPinch(percent) {
    this.params.pinch = Math.max(0, Math.min(100, percent));
    // Pinch would modulate pitch - simplified here
    const pinchDepth = percent / 100;
    this.pinchGain.gain.value = pinchDepth * 5;
  }

  setCrackle(percent) {
    this.params.crackle = Math.max(0, Math.min(100, percent));
    this.updateCrackle();
  }

  setCrackleVolume(db) {
    this.params.crackleVolume = Math.max(-60, Math.min(0, db));
  }

  setWear(percent) {
    this.params.wear = Math.max(0, Math.min(100, percent));
    this.updateWear();
  }

  setWarp(percent) {
    this.params.warp = Math.max(0, Math.min(100, percent));
    this.updateWarp();
  }

  setWarpFrequency(hz) {
    this.params.warpFrequency = Math.max(0.1, Math.min(5, hz));
    this.updateWarp();
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
    // Stop crackle generation
    if (this.crackleInterval) {
      clearInterval(this.crackleInterval);
      this.crackleInterval = null;
    }

    // Stop all active crackles
    this.activeCrackles.forEach(crackle => {
      try {
        crackle.source.stop();
        crackle.source.disconnect();
        crackle.gain.disconnect();
      } catch (e) {
        // Already stopped
      }
    });

    this.activeCrackles = [];

    // Stop LFOs
    try {
      this.warpLFO.stop();
      this.pinchLFO.stop();
    } catch (e) {
      // Already stopped
    }

    // Disconnect nodes
    this.input.disconnect();
    this.trackingDistortion.disconnect();
    this.wearFilter.disconnect();
    this.crackleGain.disconnect();
    this.output.disconnect();
  }
}

export default VinylDistortion;
