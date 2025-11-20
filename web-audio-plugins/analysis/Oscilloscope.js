/**
 * Oscilloscope Plugin
 * Real-time waveform visualization with triggering
 *
 * Features:
 * - Waveform capture and display
 * - Multiple trigger modes (auto, normal, single)
 * - Configurable trigger level and edge
 * - Dual channel support (stereo)
 * - Configurable buffer size
 * - Canvas-based visualization
 * - Event-based waveform updates
 *
 * @example
 * const audioContext = new AudioContext();
 * const canvas = document.getElementById('scope-canvas');
 * const scope = new OscilloscopePlugin(audioContext, {
 *   canvas: canvas,
 *   bufferSize: 2048,
 *   triggerMode: 'auto'
 * });
 *
 * // Connect audio source
 * source.connect(scope.input);
 * scope.connect(audioContext.destination);
 *
 * // Configure trigger
 * scope.setTriggerLevel(0.1);
 * scope.setTriggerEdge('rising');
 *
 * @author Agent 8 (Analyzer Plugins)
 * @version 1.0.0
 */

class OscilloscopePlugin {
  /**
   * Create an Oscilloscope plugin
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Initial configuration
   * @param {HTMLCanvasElement} options.canvas - Canvas element for visualization
   * @param {number} options.bufferSize - Waveform buffer size (128 to 16384)
   * @param {number} options.updateRate - Updates per second (1 to 120)
   * @param {string} options.triggerMode - 'auto', 'normal', or 'single'
   * @param {number} options.triggerLevel - Trigger level (-1 to 1)
   * @param {string} options.triggerEdge - 'rising' or 'falling'
   * @param {number} options.triggerChannel - 0 (left) or 1 (right)
   * @param {boolean} options.showBothChannels - Show both channels
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
      bufferSize: options.bufferSize || 2048,
      updateRate: options.updateRate || 60,
      triggerMode: options.triggerMode || 'auto',
      triggerLevel: options.triggerLevel !== undefined ? options.triggerLevel : 0.0,
      triggerEdge: options.triggerEdge || 'rising',
      triggerChannel: options.triggerChannel || 0,
      showBothChannels: options.showBothChannels !== undefined ? options.showBothChannels : true
    };

    // Event listeners
    this.listeners = new Map();

    // Current waveform data (cached for sync access)
    this.currentWaveform = {
      waveformL: null,
      waveformR: null,
      bufferSize: this.parameters.bufferSize,
      triggerPosition: 0,
      triggered: false,
      timestamp: 0
    };

    // Colors for visualization
    this.colors = {
      background: '#000000',
      grid: '#333333',
      text: '#aaaaaa',
      waveformL: '#00ff00',
      waveformR: '#ffff00',
      trigger: '#ff0000'
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

      // Add worklet module
      await this.context.audioWorklet.addModule(`${basePath}/worklets/oscilloscope-processor.js`);

      // Create worklet node
      this.workletNode = new AudioWorkletNode(
        this.context,
        'oscilloscope-processor',
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
      console.error('Error setting up Oscilloscope worklet:', error);
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
        if (script.src && script.src.includes('OscilloscopePlugin.js')) {
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
    if (data.type === 'waveform-update') {
      // Update current waveform cache
      this.currentWaveform = {
        waveformL: data.waveformL,
        waveformR: data.waveformR,
        bufferSize: data.bufferSize,
        triggerPosition: data.triggerPosition,
        triggered: data.triggered,
        timestamp: data.timestamp
      };

      // Emit update event
      this.emit('update', this.currentWaveform);
    }
  }

  /**
   * Apply all parameters to worklet
   * @private
   */
  applyParameters() {
    if (!this.workletNode) return;

    const workletParams = ['bufferSize', 'updateRate', 'triggerMode', 'triggerLevel', 'triggerEdge', 'triggerChannel'];

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
   * Set buffer size
   * @param {number} size - Buffer size (128 to 16384)
   */
  setBufferSize(size) {
    this.parameters.bufferSize = Math.max(128, Math.min(16384, size));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'bufferSize',
        value: this.parameters.bufferSize
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
   * Set trigger mode
   * @param {string} mode - 'auto', 'normal', or 'single'
   */
  setTriggerMode(mode) {
    if (['auto', 'normal', 'single'].includes(mode)) {
      this.parameters.triggerMode = mode;
      if (this.workletNode) {
        this.workletNode.port.postMessage({
          type: 'triggerMode',
          value: mode
        });
      }
    }
  }

  /**
   * Set trigger level
   * @param {number} level - Trigger level (-1 to 1)
   */
  setTriggerLevel(level) {
    this.parameters.triggerLevel = Math.max(-1, Math.min(1, level));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'triggerLevel',
        value: this.parameters.triggerLevel
      });
    }
  }

  /**
   * Set trigger edge
   * @param {string} edge - 'rising' or 'falling'
   */
  setTriggerEdge(edge) {
    if (['rising', 'falling'].includes(edge)) {
      this.parameters.triggerEdge = edge;
      if (this.workletNode) {
        this.workletNode.port.postMessage({
          type: 'triggerEdge',
          value: edge
        });
      }
    }
  }

  /**
   * Set trigger channel
   * @param {number} channel - 0 (left) or 1 (right)
   */
  setTriggerChannel(channel) {
    this.parameters.triggerChannel = channel === 0 ? 0 : 1;
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'triggerChannel',
        value: this.parameters.triggerChannel
      });
    }
  }

  /**
   * Reset oscilloscope
   */
  reset() {
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'reset'
      });
    }
  }

  /**
   * Get current waveform (synchronous, may be slightly outdated)
   * @returns {Object} Current waveform data
   */
  getWaveform() {
    return { ...this.currentWaveform };
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
   * Draw waveform on canvas
   * @private
   */
  draw() {
    if (!this.canvas || !this.canvasContext || !this.currentWaveform.waveformL) {
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

    // Draw trigger level
    this.drawTriggerLevel(ctx, width, height);

    // Draw waveforms
    this.drawWaveform(ctx, width, height);

    // Draw status text
    this.drawStatus(ctx, width, height);
  }

  /**
   * Draw grid
   * @private
   */
  drawGrid(ctx, width, height) {
    ctx.strokeStyle = this.colors.grid;
    ctx.lineWidth = 1;

    // Center lines
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.stroke();

    // Grid lines
    const gridSpacing = 50;
    ctx.globalAlpha = 0.3;

    for (let x = gridSpacing; x < width; x += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    for (let y = gridSpacing; y < height; y += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    ctx.globalAlpha = 1.0;
  }

  /**
   * Draw trigger level line
   * @private
   */
  drawTriggerLevel(ctx, width, height) {
    const y = height / 2 - (this.parameters.triggerLevel * height / 2);

    ctx.strokeStyle = this.colors.trigger;
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();

    ctx.setLineDash([]);
  }

  /**
   * Draw waveform
   * @private
   */
  drawWaveform(ctx, width, height) {
    const waveformL = this.currentWaveform.waveformL;
    const waveformR = this.currentWaveform.waveformR;
    const bufferSize = this.currentWaveform.bufferSize;

    if (!waveformL || bufferSize === 0) return;

    const xStep = width / bufferSize;

    // Draw left channel
    ctx.strokeStyle = this.colors.waveformL;
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let i = 0; i < bufferSize; i++) {
      const x = i * xStep;
      const y = height / 2 - (waveformL[i] * height / 2);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.stroke();

    // Draw right channel if enabled and different from left
    if (this.parameters.showBothChannels && waveformR && waveformR !== waveformL) {
      ctx.strokeStyle = this.colors.waveformR;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();

      for (let i = 0; i < bufferSize; i++) {
        const x = i * xStep;
        const y = height / 2 - (waveformR[i] * height / 2);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }
  }

  /**
   * Draw status text
   * @private
   */
  drawStatus(ctx, width, height) {
    ctx.fillStyle = this.colors.text;
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';

    const status = [
      `Mode: ${this.parameters.triggerMode.toUpperCase()}`,
      `Level: ${this.parameters.triggerLevel.toFixed(2)}`,
      `Edge: ${this.parameters.triggerEdge}`,
      `Triggered: ${this.currentWaveform.triggered ? 'YES' : 'NO'}`
    ];

    status.forEach((text, i) => {
      ctx.fillText(text, 10, 20 + i * 15);
    });
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
      name: 'Oscilloscope',
      category: 'analysis',
      description: 'Real-time waveform visualization with triggering',
      author: 'Agent 8',
      version: '1.0.0'
    };
  }
}

// Export for use in Node.js or as module
if (typeof module !== 'undefined' && module.exports) {
  module.exports = OscilloscopePlugin;
}
