/**
 * Spectrum Analyzer Plugin
 * Real-time FFT-based frequency spectrum display
 *
 * Features:
 * - Real-time FFT visualization
 * - Multiple FFT sizes (512 to 16384)
 * - Temporal smoothing
 * - Linear or logarithmic frequency scale
 * - Peak hold mode
 * - Freeze display
 * - Tilt compensation
 * - Multiple channel display modes
 */

class SpectrumAnalyzer {
  constructor(audioContext, canvasElement, options = {}) {
    this.context = audioContext;
    this.canvas = canvasElement;
    this.canvasContext = canvasElement.getContext('2d');

    // Audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();
    this.analyser = audioContext.createAnalyser();

    // Default FFT size
    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0.8;

    // FFT data buffers
    this.bufferLength = this.analyser.frequencyBinCount;
    this.dataArray = new Uint8Array(this.bufferLength);
    this.floatDataArray = new Float32Array(this.bufferLength);
    this.peakArray = new Uint8Array(this.bufferLength);

    // State
    this.state = {
      fftSize: 2048,
      smoothing: 80, // 0-100%
      scale: 'logarithmic', // 'linear' or 'logarithmic'
      minDb: -100,
      maxDb: 0,
      peakHold: false,
      freeze: false,
      tilt: 0, // -6 to +6 dB/octave
      channels: 'L+R', // 'L', 'R', 'L+R', 'Mid', 'Side'
      running: false
    };

    // Animation frame ID
    this.animationId = null;

    // Colors
    this.colors = {
      background: '#1a1a1a',
      grid: '#333333',
      text: '#aaaaaa',
      spectrum: '#00ff00',
      peak: '#ffff00'
    };

    this.setupRouting();
    this.initialize(options);
  }

  setupRouting() {
    // Pass-through with analysis
    this.input.connect(this.analyser);
    this.input.connect(this.output);
  }

  initialize(options) {
    if (options.fftSize) this.setFFTSize(options.fftSize);
    if (options.smoothing !== undefined) this.setSmoothing(options.smoothing);
    if (options.scale) this.setScale(options.scale);
    if (options.minDb !== undefined) this.state.minDb = options.minDb;
    if (options.maxDb !== undefined) this.state.maxDb = options.maxDb;
    if (options.peakHold !== undefined) this.setPeakHold(options.peakHold);
    if (options.freeze !== undefined) this.setFreeze(options.freeze);
    if (options.tilt !== undefined) this.setTilt(options.tilt);
    if (options.channels) this.setChannels(options.channels);
    if (options.colors) this.colors = { ...this.colors, ...options.colors };

    // Auto-start if not explicitly disabled
    if (options.autoStart !== false) {
      this.start();
    }
  }

  /**
   * Set FFT size (must be power of 2, between 32 and 32768)
   */
  setFFTSize(size) {
    const validSizes = [512, 1024, 2048, 4096, 8192, 16384];
    if (!validSizes.includes(size)) {
      console.warn(`Invalid FFT size: ${size}. Using 2048.`);
      size = 2048;
    }

    this.state.fftSize = size;
    this.analyser.fftSize = size;
    this.bufferLength = this.analyser.frequencyBinCount;
    this.dataArray = new Uint8Array(this.bufferLength);
    this.floatDataArray = new Float32Array(this.bufferLength);
    this.peakArray = new Uint8Array(this.bufferLength);
  }

  /**
   * Set smoothing (0 to 100%)
   */
  setSmoothing(percent) {
    this.state.smoothing = Math.max(0, Math.min(100, percent));
    this.analyser.smoothingTimeConstant = percent / 100;
  }

  /**
   * Set frequency scale ('linear' or 'logarithmic')
   */
  setScale(scale) {
    if (scale === 'linear' || scale === 'logarithmic') {
      this.state.scale = scale;
    }
  }

  /**
   * Set peak hold mode
   */
  setPeakHold(enabled) {
    this.state.peakHold = enabled;
    if (!enabled) {
      this.peakArray.fill(0);
    }
  }

  /**
   * Set freeze display
   */
  setFreeze(enabled) {
    this.state.freeze = enabled;
  }

  /**
   * Set tilt (-6 to +6 dB/octave)
   */
  setTilt(tilt) {
    this.state.tilt = Math.max(-6, Math.min(6, tilt));
  }

  /**
   * Set channel display mode
   */
  setChannels(mode) {
    const validModes = ['L', 'R', 'L+R', 'Mid', 'Side'];
    if (validModes.includes(mode)) {
      this.state.channels = mode;
    }
  }

  /**
   * Start animation loop
   */
  start() {
    if (this.state.running) return;

    this.state.running = true;
    this.animate();
  }

  /**
   * Stop animation loop
   */
  stop() {
    this.state.running = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  /**
   * Animation loop
   */
  animate() {
    if (!this.state.running) return;

    this.animationId = requestAnimationFrame(() => this.animate());

    // Get frequency data
    if (!this.state.freeze) {
      this.analyser.getByteFrequencyData(this.dataArray);

      // Update peak hold
      if (this.state.peakHold) {
        for (let i = 0; i < this.bufferLength; i++) {
          if (this.dataArray[i] > this.peakArray[i]) {
            this.peakArray[i] = this.dataArray[i];
          } else {
            this.peakArray[i] *= 0.995; // Slow decay
          }
        }
      }
    }

    this.draw();
  }

  /**
   * Draw spectrum
   */
  draw() {
    const width = this.canvas.width;
    const height = this.canvas.height;
    const ctx = this.canvasContext;

    // Clear
    ctx.fillStyle = this.colors.background;
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    this.drawGrid(ctx, width, height);

    // Draw frequency bars
    this.drawSpectrum(ctx, width, height);

    // Draw frequency labels
    this.drawFrequencyLabels(ctx, width, height);

    // Draw dB scale
    this.drawDbScale(ctx, width, height);
  }

  /**
   * Draw grid
   */
  drawGrid(ctx, width, height) {
    ctx.strokeStyle = this.colors.grid;
    ctx.lineWidth = 1;

    // Horizontal lines (dB levels)
    const dbSteps = [-80, -60, -40, -20, 0];
    dbSteps.forEach(db => {
      const y = this.dbToY(db, height);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    });

    // Vertical lines (frequencies)
    const frequencies = [100, 1000, 10000];
    frequencies.forEach(freq => {
      const x = this.frequencyToX(freq, width);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    });
  }

  /**
   * Draw spectrum
   */
  drawSpectrum(ctx, width, height) {
    const nyquist = this.context.sampleRate / 2;
    const binWidth = nyquist / this.bufferLength;

    if (this.state.scale === 'logarithmic') {
      this.drawSpectrumLogarithmic(ctx, width, height, binWidth, nyquist);
    } else {
      this.drawSpectrumLinear(ctx, width, height);
    }
  }

  /**
   * Draw spectrum with logarithmic frequency scale
   */
  drawSpectrumLogarithmic(ctx, width, height, binWidth, nyquist) {
    const minFreq = 20;
    const maxFreq = nyquist;

    // Draw filled area
    ctx.fillStyle = this.colors.spectrum;
    ctx.globalAlpha = 0.7;

    ctx.beginPath();
    ctx.moveTo(0, height);

    for (let x = 0; x < width; x++) {
      const freq = this.xToFrequency(x, width, minFreq, maxFreq);
      const bin = Math.floor(freq / binWidth);

      if (bin >= 0 && bin < this.bufferLength) {
        let value = this.dataArray[bin];

        // Apply tilt compensation
        if (this.state.tilt !== 0) {
          const octaves = Math.log2(freq / 1000);
          const tiltDb = octaves * this.state.tilt;
          value = Math.max(0, Math.min(255, value + (tiltDb * 255 / 100)));
        }

        const y = (1 - value / 255) * height;
        ctx.lineTo(x, y);
      }
    }

    ctx.lineTo(width, height);
    ctx.closePath();
    ctx.fill();

    // Draw peak hold
    if (this.state.peakHold) {
      ctx.strokeStyle = this.colors.peak;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 1.0;
      ctx.beginPath();

      for (let x = 0; x < width; x++) {
        const freq = this.xToFrequency(x, width, minFreq, maxFreq);
        const bin = Math.floor(freq / binWidth);

        if (bin >= 0 && bin < this.bufferLength) {
          const value = this.peakArray[bin];
          const y = (1 - value / 255) * height;

          if (x === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
      }

      ctx.stroke();
    }

    ctx.globalAlpha = 1.0;
  }

  /**
   * Draw spectrum with linear frequency scale
   */
  drawSpectrumLinear(ctx, width, height) {
    const barWidth = width / this.bufferLength;

    ctx.fillStyle = this.colors.spectrum;
    ctx.globalAlpha = 0.7;

    for (let i = 0; i < this.bufferLength; i++) {
      const value = this.dataArray[i];
      const barHeight = (value / 255) * height;
      const x = i * barWidth;
      const y = height - barHeight;

      ctx.fillRect(x, y, barWidth, barHeight);

      // Draw peak
      if (this.state.peakHold) {
        const peakValue = this.peakArray[i];
        const peakY = height - (peakValue / 255) * height;
        ctx.fillStyle = this.colors.peak;
        ctx.fillRect(x, peakY - 1, barWidth, 2);
        ctx.fillStyle = this.colors.spectrum;
      }
    }

    ctx.globalAlpha = 1.0;
  }

  /**
   * Draw frequency labels
   */
  drawFrequencyLabels(ctx, width, height) {
    ctx.fillStyle = this.colors.text;
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';

    const frequencies = [20, 100, 1000, 10000, 20000];
    frequencies.forEach(freq => {
      const x = this.frequencyToX(freq, width);
      const label = freq >= 1000 ? `${freq / 1000}k` : `${freq}`;
      ctx.fillText(label, x, height - 5);
    });
  }

  /**
   * Draw dB scale
   */
  drawDbScale(ctx, width, height) {
    ctx.fillStyle = this.colors.text;
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';

    const dbLevels = [-80, -60, -40, -20, 0];
    dbLevels.forEach(db => {
      const y = this.dbToY(db, height);
      ctx.fillText(`${db}dB`, width - 5, y - 3);
    });
  }

  /**
   * Convert frequency to X position (logarithmic)
   */
  frequencyToX(freq, width) {
    const minFreq = 20;
    const maxFreq = this.context.sampleRate / 2;

    if (this.state.scale === 'logarithmic') {
      return width * Math.log(freq / minFreq) / Math.log(maxFreq / minFreq);
    } else {
      return width * freq / maxFreq;
    }
  }

  /**
   * Convert X position to frequency (logarithmic)
   */
  xToFrequency(x, width, minFreq, maxFreq) {
    if (this.state.scale === 'logarithmic') {
      return minFreq * Math.pow(maxFreq / minFreq, x / width);
    } else {
      return (x / width) * maxFreq;
    }
  }

  /**
   * Convert dB to Y position
   */
  dbToY(db, height) {
    const range = this.state.maxDb - this.state.minDb;
    return height * (1 - (db - this.state.minDb) / range);
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
  module.exports = SpectrumAnalyzer;
}
