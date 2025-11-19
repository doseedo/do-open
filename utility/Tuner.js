/**
 * Tuner Plugin
 * Pitch detection and tuning reference
 *
 * Features:
 * - Accurate pitch detection using autocorrelation
 * - Chromatic tuning with all 12 notes
 * - Adjustable reference frequency (A4)
 * - Cents deviation display
 * - Real-time updates
 * - Visual tuning indicator
 */

class Tuner {
  constructor(audioContext, displayElement, options = {}) {
    this.context = audioContext;
    this.display = displayElement;

    // Audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();
    this.analyser = audioContext.createAnalyser();
    this.analyser.fftSize = 4096; // Higher FFT size for better low frequency resolution

    // Buffers for pitch detection
    this.bufferLength = this.analyser.fftSize;
    this.buffer = new Float32Array(this.bufferLength);

    // State
    this.state = {
      referenceFreq: 440, // A4 = 440 Hz
      input: 'auto', // 'auto', 'L', 'R'
      mode: 'chromatic', // 'chromatic', 'guitar', 'bass', etc.
      tolerance: 50, // ±50 cents
      running: false,
      minVolume: 0.01 // Minimum RMS to trigger detection
    };

    // Current detection
    this.currentNote = {
      note: '',
      octave: 0,
      cents: 0,
      frequency: 0,
      confidence: 0
    };

    // Animation frame ID
    this.animationId = null;

    // Note names
    this.noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

    this.setupRouting();
    this.initialize(options);
  }

  setupRouting() {
    // Pass-through with analysis
    this.input.connect(this.analyser);
    this.input.connect(this.output);
  }

  initialize(options) {
    if (options.referenceFreq !== undefined) this.setReferenceFreq(options.referenceFreq);
    if (options.input) this.state.input = options.input;
    if (options.mode) this.state.mode = options.mode;
    if (options.tolerance !== undefined) this.state.tolerance = options.tolerance;
    if (options.minVolume !== undefined) this.state.minVolume = options.minVolume;

    // Auto-start if not explicitly disabled
    if (options.autoStart !== false) {
      this.start();
    }
  }

  /**
   * Set reference frequency (A4)
   */
  setReferenceFreq(freq) {
    this.state.referenceFreq = Math.max(400, Math.min(480, freq));
  }

  /**
   * Start pitch detection loop
   */
  start() {
    if (this.state.running) return;

    this.state.running = true;
    this.detect();
  }

  /**
   * Stop pitch detection loop
   */
  stop() {
    this.state.running = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  /**
   * Detection loop
   */
  detect() {
    if (!this.state.running) return;

    this.animationId = requestAnimationFrame(() => this.detect());

    // Get time domain data
    this.analyser.getFloatTimeDomainData(this.buffer);

    // Detect pitch
    const frequency = this.detectPitch(this.buffer, this.context.sampleRate);

    if (frequency !== -1) {
      const noteInfo = this.frequencyToNote(frequency);
      this.currentNote = noteInfo;
      this.updateDisplay(noteInfo);
    } else {
      // No pitch detected
      this.currentNote = {
        note: '',
        octave: 0,
        cents: 0,
        frequency: 0,
        confidence: 0
      };
      this.updateDisplay(null);
    }
  }

  /**
   * Detect pitch using autocorrelation
   * Returns frequency in Hz, or -1 if no pitch detected
   */
  detectPitch(buffer, sampleRate) {
    const SIZE = buffer.length;
    let sumOfSquares = 0;

    // Calculate RMS to check if signal is loud enough
    for (let i = 0; i < SIZE; i++) {
      const val = buffer[i];
      sumOfSquares += val * val;
    }

    const rms = Math.sqrt(sumOfSquares / SIZE);
    if (rms < this.state.minVolume) {
      return -1; // Signal too quiet
    }

    // Autocorrelation
    let maxCorrelation = 0;
    let maxLag = -1;

    // Find the range of lags to search
    // For musical notes, we're interested in ~80 Hz to ~1200 Hz
    const minLag = Math.floor(sampleRate / 1200); // Highest frequency
    const maxLagLimit = Math.floor(sampleRate / 80); // Lowest frequency

    // Perform autocorrelation
    for (let lag = minLag; lag < Math.min(maxLagLimit, SIZE / 2); lag++) {
      let correlation = 0;

      for (let i = 0; i < SIZE - lag; i++) {
        correlation += buffer[i] * buffer[i + lag];
      }

      // Normalize by the number of samples
      correlation /= (SIZE - lag);

      if (correlation > maxCorrelation) {
        maxCorrelation = correlation;
        maxLag = lag;
      }
    }

    // Check if correlation is strong enough
    if (maxCorrelation < 0.01 || maxLag === -1) {
      return -1;
    }

    // Refine the lag estimate using parabolic interpolation
    const frequency = this.refineFrequency(buffer, maxLag, sampleRate);

    return frequency;
  }

  /**
   * Refine frequency estimate using parabolic interpolation
   */
  refineFrequency(buffer, lag, sampleRate) {
    if (lag < 1 || lag >= buffer.length / 2 - 1) {
      return sampleRate / lag;
    }

    // Get correlation values around the peak
    const SIZE = buffer.length;
    const c0 = this.calculateCorrelation(buffer, SIZE, lag - 1);
    const c1 = this.calculateCorrelation(buffer, SIZE, lag);
    const c2 = this.calculateCorrelation(buffer, SIZE, lag + 1);

    // Parabolic interpolation
    const delta = 0.5 * (c0 - c2) / (c0 - 2 * c1 + c2);
    const refinedLag = lag + delta;

    return sampleRate / refinedLag;
  }

  /**
   * Calculate autocorrelation for a specific lag
   */
  calculateCorrelation(buffer, size, lag) {
    let correlation = 0;
    for (let i = 0; i < size - lag; i++) {
      correlation += buffer[i] * buffer[i + lag];
    }
    return correlation / (size - lag);
  }

  /**
   * Convert frequency to note information
   */
  frequencyToNote(frequency) {
    // Calculate semitones from A4
    const semitonesFromA4 = 12 * Math.log2(frequency / this.state.referenceFreq);

    // Round to nearest semitone
    const roundedSemitones = Math.round(semitonesFromA4);

    // Calculate cents deviation
    const cents = Math.round((semitonesFromA4 - roundedSemitones) * 100);

    // Calculate note index (A = 0)
    const noteIndex = (roundedSemitones + 9 + 1200) % 12; // +9 because C is 9 semitones below A

    // Calculate octave
    const octave = Math.floor((roundedSemitones + 9) / 12) + 4;

    return {
      note: this.noteNames[noteIndex],
      octave: octave,
      cents: cents,
      frequency: frequency.toFixed(2),
      confidence: this.calculateConfidence(cents)
    };
  }

  /**
   * Calculate confidence based on cents deviation
   */
  calculateConfidence(cents) {
    // Confidence decreases as cents deviation increases
    const absCents = Math.abs(cents);
    if (absCents < 5) return 1.0;
    if (absCents < 10) return 0.9;
    if (absCents < 20) return 0.7;
    if (absCents < 30) return 0.5;
    return 0.3;
  }

  /**
   * Update display
   */
  updateDisplay(noteInfo) {
    if (!this.display) return;

    if (!noteInfo) {
      this.display.innerHTML = `
        <div class="tuner-display">
          <div class="tuner-note">--</div>
          <div class="tuner-cents">--</div>
          <div class="tuner-frequency">-- Hz</div>
          <div class="tuner-indicator">
            <div class="tuner-indicator-bar"></div>
          </div>
        </div>
      `;
      return;
    }

    // Determine tuning status
    const absCents = Math.abs(noteInfo.cents);
    let status = 'in-tune';
    if (absCents > 5) status = 'out-of-tune';
    if (absCents > 20) status = 'very-out-of-tune';

    // Indicator position (-50 to +50 cents maps to -100% to +100%)
    const indicatorPos = Math.max(-100, Math.min(100, noteInfo.cents * 2));

    this.display.innerHTML = `
      <div class="tuner-display ${status}">
        <div class="tuner-note">${noteInfo.note}${noteInfo.octave}</div>
        <div class="tuner-cents">${noteInfo.cents > 0 ? '+' : ''}${noteInfo.cents}¢</div>
        <div class="tuner-frequency">${noteInfo.frequency} Hz</div>
        <div class="tuner-indicator">
          <div class="tuner-indicator-center"></div>
          <div class="tuner-indicator-needle" style="left: ${50 + indicatorPos / 2}%"></div>
        </div>
      </div>
    `;

    // Add CSS styles if not already present
    this.ensureStyles();
  }

  /**
   * Ensure CSS styles are present
   */
  ensureStyles() {
    if (document.getElementById('tuner-styles')) return;

    const style = document.createElement('style');
    style.id = 'tuner-styles';
    style.textContent = `
      .tuner-display {
        font-family: monospace;
        text-align: center;
        padding: 20px;
        background: #1a1a1a;
        border-radius: 10px;
      }

      .tuner-note {
        font-size: 48px;
        font-weight: bold;
        color: #00ff00;
        margin-bottom: 10px;
      }

      .tuner-display.out-of-tune .tuner-note {
        color: #ffaa00;
      }

      .tuner-display.very-out-of-tune .tuner-note {
        color: #ff0000;
      }

      .tuner-cents {
        font-size: 24px;
        color: #aaaaaa;
        margin-bottom: 5px;
      }

      .tuner-frequency {
        font-size: 16px;
        color: #666666;
        margin-bottom: 20px;
      }

      .tuner-indicator {
        position: relative;
        height: 40px;
        background: #333333;
        border-radius: 5px;
        overflow: hidden;
      }

      .tuner-indicator-center {
        position: absolute;
        left: 50%;
        top: 0;
        width: 2px;
        height: 100%;
        background: #00ff00;
        transform: translateX(-50%);
      }

      .tuner-indicator-needle {
        position: absolute;
        top: 50%;
        width: 4px;
        height: 80%;
        background: #ffffff;
        transform: translate(-50%, -50%);
        transition: left 0.1s ease;
      }
    `;

    document.head.appendChild(style);
  }

  /**
   * Get current note information
   */
  getCurrentNote() {
    return { ...this.currentNote };
  }

  /**
   * Get current state
   */
  getState() {
    return { ...this.state };
  }

  /**
   * Connect to destination
   */
  connect(destination) {
    this.output.connect(destination);
  }

  /**
   * Disconnect
   */
  disconnect() {
    this.output.disconnect();
  }

  /**
   * Cleanup
   */
  destroy() {
    this.stop();
    this.disconnect();
    this.input.disconnect();
    this.analyser.disconnect();
  }
}

// Export for use in Node.js or as module
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Tuner;
}
