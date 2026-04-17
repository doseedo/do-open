/**
 * Spectrum Analyzer Plugin
 * Real-time FFT-based frequency spectrum visualization
 *
 * Features:
 * - Real-time FFT analysis (512 to 16384 samples)
 * - Multiple frequency scales (linear, logarithmic)
 * - Temporal smoothing for stable display
 * - Peak hold mode
 * - Configurable dB range
 * - Canvas-based visualization
 * - Event-based spectrum updates
 *
 * @example
 * const audioContext = new AudioContext();
 * const canvas = document.getElementById('spectrum-canvas');
 * const analyzer = new SpectrumAnalyzerPlugin(audioContext, {
 *   canvas: canvas,
 *   fftSize: 2048,
 *   smoothing: 0.8
 * });
 *
 * // Connect audio source
 * source.connect(analyzer.input);
 * analyzer.connect(audioContext.destination);
 *
 * // Listen for spectrum updates
 * analyzer.on('update', (data) => {
 *   console.log('Spectrum data:', data.spectrum);
 * });
 *
 * @author Agent 8 (Analyzer Plugins)
 * @version 1.0.0
 */

class SpectrumAnalyzerPlugin {
  /**
   * Create a Spectrum Analyzer plugin
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Initial configuration
   * @param {HTMLCanvasElement} options.canvas - Canvas element for visualization
   * @param {number} options.fftSize - FFT size (512, 1024, 2048, 4096, 8192, 16384)
   * @param {number} options.smoothing - Temporal smoothing (0 to 1)
   * @param {boolean} options.peakHold - Enable peak hold
   * @param {number} options.updateRate - Updates per second (1 to 120)
   * @param {string} options.scale - Frequency scale ('linear' or 'logarithmic')
   * @param {number} options.minDb - Minimum dB value for display
   * @param {number} options.maxDb - Maximum dB value for display
   */
  constructor(audioContext, options = {}) {
    this.context = audioContext;
    this.canvas = options.canvas || null;
    this.canvasContext = this.canvas ? this.canvas.getContext('2d') : null;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();
    this.workletNode = null;

    // Parameters
    this.parameters = {
      fftSize: options.fftSize || 2048,
      smoothing: options.smoothing !== undefined ? options.smoothing : 0.8,
      peakHold: options.peakHold !== undefined ? options.peakHold : false,
      updateRate: options.updateRate || 30,
      scale: options.scale || 'logarithmic',
      minDb: options.minDb !== undefined ? options.minDb : -100,
      maxDb: options.maxDb !== undefined ? options.maxDb : 0
    };

    // Event listeners
    this.listeners = new Map();

    // Current spectrum data (cached for sync access)
    this.currentSpectrum = {
      spectrum: null,
      peakSpectrum: null,
      fftSize: this.parameters.fftSize,
      sampleRate: audioContext.sampleRate,
      timestamp: 0
    };

    // Colors for visualization
    this.colors = {
      background: '#1a1a1a',
      grid: '#333333',
      text: '#aaaaaa',
      spectrum: '#00ff00',
      peak: '#ffff00'
    };

    // Animation state
    this.animationId = null;
    this.isAnimating = false;

    // Setup state
    this.isReady = false;
    this.setupPromise = this.setupWorklet();
  }

  /**
   * Setup AudioWorklet processor
   * @private
   */
  async setupWorklet() {
    try {
      // Get the base path for worklet files
      const basePath = this.getBasePath();

      // Add worklet modules (FFT lib first, then processor)
      await this.context.audioWorklet.addModule(`${basePath}/../spectral/worklets/fft-lib.js`);
      await this.context.audioWorklet.addModule(`${basePath}/worklets/spectrum-analyzer-processor.js`);

      // Create worklet node
      this.workletNode = new AudioWorkletNode(
        this.context,
        'spectrum-analyzer-processor',
        {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          outputChannelCount: [2]
        }
      );

      // Setup message handler
      this.workletNode.port.onmessage = (e) => {
        this.handleWorkletMessage(e.data);
      };

      // Connect nodes
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      // Apply initial parameters
      this.applyParameters();

      this.isReady = true;

      // Start animation if canvas is available
      if (this.canvas) {
        this.startAnimation();
      }
    } catch (error) {
      console.error('Error setting up Spectrum Analyzer worklet:', error);
      throw error;
    }
  }

  /**
   * Get base path for worklet files
   * @private
   */
  getBasePath() {
    // Try to determine the base path from the script location
    if (typeof document !== 'undefined') {
      const scripts = document.getElementsByTagName('script');
      for (let script of scripts) {
        if (script.src && script.src.includes('SpectrumAnalyzerPlugin.js')) {
          return script.src.substring(0, script.src.lastIndexOf('/'));
        }
      }
    }
    return './analysis';
  }

  /**
   * Handle messages from worklet
   * @private
   */
  handleWorkletMessage(data) {
    if (data.type === 'spectrum-update') {
      // Update current spectrum cache
      this.currentSpectrum = {
        spectrum: data.spectrum,
        peakSpectrum: data.peakSpectrum,
        fftSize: data.fftSize,
        sampleRate: data.sampleRate,
        timestamp: data.timestamp
      };

      // Emit update event
      this.emit('update', this.currentSpectrum);
    }
  }

  /**
   * Apply all parameters to worklet
   * @private
   */
  applyParameters() {
    if (!this.workletNode) return;

    const workletParams = ['fftSize', 'smoothing', 'peakHold', 'updateRate'];

    workletParams.forEach(param => {
      this.workletNode.port.postMessage({
        type: param,
        value: this.parameters[param]
      });
    });
  }

  /**
   * Wait for the processor to be ready
   * @returns {Promise<void>}
   */
  async ready() {
    await this.setupPromise;
  }

  /**
   * Set FFT size
   * @param {number} size - FFT size (512, 1024, 2048, 4096, 8192, 16384)
   */
  setFFTSize(size) {
    const validSizes = [512, 1024, 2048, 4096, 8192, 16384];
    if (!validSizes.includes(size)) {
      console.warn(`Invalid FFT size: ${size}`);
      return;
    }

    this.parameters.fftSize = size;
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'fftSize',
        value: size
      });
    }
  }

  /**
   * Set smoothing amount
   * @param {number} amount - Smoothing (0 to 1)
   */
  setSmoothing(amount) {
    this.parameters.smoothing = Math.max(0, Math.min(1, amount));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'smoothing',
        value: this.parameters.smoothing
      });
    }
  }

  /**
   * Set peak hold mode
   * @param {boolean} enabled - Enable peak hold
   */
  setPeakHold(enabled) {
    this.parameters.peakHold = enabled;
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'peakHold',
        value: enabled
      });
    }
  }

  /**
   * Set update rate
   * @param {number} rate - Updates per second (1 to 120)
   */
  setUpdateRate(rate) {
    this.parameters.updateRate = Math.max(1, Math.min(120, rate));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'updateRate',
        value: this.parameters.updateRate
      });
    }
  }

  /**
   * Set frequency scale
   * @param {string} scale - 'linear' or 'logarithmic'
   */
  setScale(scale) {
    if (scale === 'linear' || scale === 'logarithmic') {
      this.parameters.scale = scale;
    }
  }

  /**
   * Set dB range
   * @param {number} minDb - Minimum dB
   * @param {number} maxDb - Maximum dB
   */
  setDbRange(minDb, maxDb) {
    this.parameters.minDb = minDb;
    this.parameters.maxDb = maxDb;
  }

  /**
   * Reset spectrum
   */
  reset() {
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'reset'
      });
    }
  }

  /**
   * Get current spectrum (synchronous, may be slightly outdated)
   * @returns {Object} Current spectrum data
   */
  getSpectrum() {
    return { ...this.currentSpectrum };
  }

  /**
   * Start animation loop for canvas drawing
   */
  startAnimation() {
    if (this.isAnimating || !this.canvas) return;

    this.isAnimating = true;
    this.animate();
  }

  /**
   * Stop animation loop
   */
  stopAnimation() {
    this.isAnimating = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  /**
   * Animation loop
   * @private
   */
  animate() {
    if (!this.isAnimating) return;

    this.animationId = requestAnimationFrame(() => this.animate());
    this.draw();
  }

  /**
   * Draw spectrum on canvas
   * @private
   */
  draw() {
    if (!this.canvas || !this.canvasContext || !this.currentSpectrum.spectrum) {
      return;
    }

    const width = this.canvas.width;
    const height = this.canvas.height;
    const ctx = this.canvasContext;

    // Clear
    ctx.fillStyle = this.colors.background;
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    this.drawGrid(ctx, width, height);

    // Draw spectrum
    this.drawSpectrum(ctx, width, height);

    // Draw labels
    this.drawLabels(ctx, width, height);
  }

  /**
   * Draw grid
   * @private
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
   * @private
   */
  drawSpectrum(ctx, width, height) {
    const spectrum = this.currentSpectrum.spectrum;
    const peakSpectrum = this.currentSpectrum.peakSpectrum;
    const nyquist = this.currentSpectrum.sampleRate / 2;
    const binWidth = nyquist / spectrum.length;

    ctx.fillStyle = this.colors.spectrum;
    ctx.globalAlpha = 0.7;

    ctx.beginPath();
    ctx.moveTo(0, height);

    // Draw spectrum curve
    for (let x = 0; x < width; x++) {
      const freq = this.xToFrequency(x, width, 20, nyquist);
      const bin = Math.floor(freq / binWidth);

      if (bin >= 0 && bin < spectrum.length) {
        const mag = spectrum[bin];
        const db = mag > 0 ? 20 * Math.log10(mag) : this.parameters.minDb;
        const y = this.dbToY(db, height);
        ctx.lineTo(x, y);
      }
    }

    ctx.lineTo(width, height);
    ctx.closePath();
    ctx.fill();

    // Draw peak hold
    if (this.parameters.peakHold && peakSpectrum) {
      ctx.strokeStyle = this.colors.peak;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 1.0;
      ctx.beginPath();

      let started = false;
      for (let x = 0; x < width; x++) {
        const freq = this.xToFrequency(x, width, 20, nyquist);
        const bin = Math.floor(freq / binWidth);

        if (bin >= 0 && bin < peakSpectrum.length) {
          const mag = peakSpectrum[bin];
          const db = mag > 0 ? 20 * Math.log10(mag) : this.parameters.minDb;
          const y = this.dbToY(db, height);

          if (!started) {
            ctx.moveTo(x, y);
            started = true;
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
   * Draw labels
   * @private
   */
  drawLabels(ctx, width, height) {
    ctx.fillStyle = this.colors.text;
    ctx.font = '10px monospace';

    // Frequency labels
    ctx.textAlign = 'center';
    const frequencies = [20, 100, 1000, 10000, 20000];
    frequencies.forEach(freq => {
      const x = this.frequencyToX(freq, width);
      const label = freq >= 1000 ? `${freq / 1000}k` : `${freq}`;
      ctx.fillText(label, x, height - 5);
    });

    // dB labels
    ctx.textAlign = 'right';
    const dbLevels = [-80, -60, -40, -20, 0];
    dbLevels.forEach(db => {
      const y = this.dbToY(db, height);
      ctx.fillText(`${db}dB`, width - 5, y - 3);
    });
  }

  /**
   * Convert frequency to X position
   * @private
   */
  frequencyToX(freq, width) {
    const minFreq = 20;
    const maxFreq = this.currentSpectrum.sampleRate / 2;

    if (this.parameters.scale === 'logarithmic') {
      return width * Math.log(freq / minFreq) / Math.log(maxFreq / minFreq);
    } else {
      return width * freq / maxFreq;
    }
  }

  /**
   * Convert X position to frequency
   * @private
   */
  xToFrequency(x, width, minFreq, maxFreq) {
    if (this.parameters.scale === 'logarithmic') {
      return minFreq * Math.pow(maxFreq / minFreq, x / width);
    } else {
      return (x / width) * maxFreq;
    }
  }

  /**
   * Convert dB to Y position
   * @private
   */
  dbToY(db, height) {
    const range = this.parameters.maxDb - this.parameters.minDb;
    const normalized = (db - this.parameters.minDb) / range;
    return height * (1 - normalized);
  }

  /**
   * Connect to destination
   * @param {AudioNode|AudioParam} destination - Destination to connect to
   */
  connect(destination) {
    this.output.connect(destination);
  }

  /**
   * Disconnect from all outputs
   */
  disconnect() {
    this.output.disconnect();
  }

  /**
   * Add event listener
   * @param {string} event - Event name ('update')
   * @param {Function} callback - Event callback
   */
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  /**
   * Remove event listener
   * @param {string} event - Event name
   * @param {Function} callback - Event callback to remove
   */
  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index !== -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  /**
   * Emit event to all listeners
   * @private
   */
  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        callback(data);
      });
    }
  }

  /**
   * Cleanup and release resources
   */
  dispose() {
    this.stopAnimation();
    this.disconnect();
    this.input.disconnect();

    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }

    this.listeners.clear();
  }

  /**
   * Get plugin info
   * @returns {Object} Plugin metadata
   */
  getInfo() {
    return {
      name: 'Spectrum Analyzer',
      category: 'analysis',
      description: 'Real-time FFT-based frequency spectrum analysis',
      author: 'Agent 8',
      version: '1.0.0'
    };
  }
}

// Export for use in Node.js or as module
if (typeof module !== 'undefined' && module.exports) {
  module.exports = SpectrumAnalyzerPlugin;
}
