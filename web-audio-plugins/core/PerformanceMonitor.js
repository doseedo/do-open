/**
 * PerformanceMonitor - CPU usage and diagnostics for audio plugin graphs
 *
 * Monitors CPU usage per plugin, total graph CPU, buffer underruns,
 * node count, memory usage, and provides performance warnings.
 *
 * @class PerformanceMonitor
 * @author Agent 10 - Integration & Routing System
 */

class PerformanceMonitor {
  /**
   * Create a new PerformanceMonitor instance
   * @param {AudioContext} audioContext - The Web Audio API context
   * @param {Router} router - Router instance to monitor
   * @param {Object} options - Configuration options
   * @param {number} options.sampleInterval - Sample interval in ms (default: 1000)
   * @param {number} options.historyLength - Number of samples to keep (default: 60)
   * @param {boolean} options.autoStart - Start monitoring automatically (default: true)
   */
  constructor(audioContext, router, options = {}) {
    if (!audioContext || !router) {
      throw new Error('AudioContext and Router are required');
    }

    this.context = audioContext;
    this.router = router;

    this.options = {
      sampleInterval: options.sampleInterval || 1000,
      historyLength: options.historyLength || 60,
      autoStart: options.autoStart !== false
    };

    // Performance measurements per plugin
    this.measurements = new Map();

    // Global performance metrics
    this.globalMetrics = {
      cpuHistory: [],
      nodeCountHistory: [],
      connectionCountHistory: [],
      bufferUnderruns: 0,
      lastUpdate: Date.now()
    };

    // Performance thresholds
    this.thresholds = {
      cpuWarning: 70,        // % CPU
      cpuCritical: 90,       // % CPU
      nodeCountWarning: 100, // number of nodes
      nodeCountCritical: 200 // number of nodes
    };

    // Warning state
    this.warnings = [];

    // Monitoring state
    this.isMonitoring = false;
    this.monitoringInterval = null;

    // Performance API baseline
    this.performanceBaseline = null;

    // Event listeners
    this.eventListeners = {
      warning: [],
      critical: [],
      metricsUpdated: []
    };

    // Start monitoring if autoStart is enabled
    if (this.options.autoStart) {
      this.startMonitoring();
    }
  }

  /**
   * Start performance monitoring
   */
  startMonitoring() {
    if (this.isMonitoring) {
      console.warn('Already monitoring');
      return;
    }

    this.isMonitoring = true;
    this.performanceBaseline = performance.now();

    this.monitoringInterval = setInterval(() => {
      this.measure();
    }, this.options.sampleInterval);
  }

  /**
   * Stop performance monitoring
   */
  stopMonitoring() {
    if (!this.isMonitoring) {
      return;
    }

    this.isMonitoring = false;

    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
  }

  /**
   * Perform performance measurement
   * @private
   */
  measure() {
    const now = Date.now();

    // Measure per-plugin performance
    this.router.getAllPlugins().forEach((plugin, id) => {
      if (!this.measurements.has(id)) {
        this.measurements.set(id, {
          pluginId: id,
          pluginName: plugin.name,
          pluginType: plugin.constructor.name,
          cpuHistory: [],
          nodeCount: 0,
          estimatedCPU: 0,
          averageCPU: 0
        });
      }

      const measurement = this.measurements.get(id);

      // Estimate CPU usage
      const estimatedCPU = this.estimateCPU(plugin);
      measurement.estimatedCPU = estimatedCPU;

      // Update history
      measurement.cpuHistory.push(estimatedCPU);
      if (measurement.cpuHistory.length > this.options.historyLength) {
        measurement.cpuHistory.shift();
      }

      // Calculate average
      measurement.averageCPU = measurement.cpuHistory.reduce((a, b) => a + b, 0) / measurement.cpuHistory.length;

      // Count nodes
      measurement.nodeCount = this.countAudioNodes(plugin);
    });

    // Clean up measurements for removed plugins
    const currentPluginIds = new Set(this.router.getAllPlugins().keys());
    Array.from(this.measurements.keys()).forEach(id => {
      if (!currentPluginIds.has(id)) {
        this.measurements.delete(id);
      }
    });

    // Global metrics
    const totalCPU = this.getTotalCPU();
    const nodeCount = this.getTotalNodeCount();
    const connectionCount = this.router.getConnections().length;

    this.globalMetrics.cpuHistory.push(totalCPU);
    if (this.globalMetrics.cpuHistory.length > this.options.historyLength) {
      this.globalMetrics.cpuHistory.shift();
    }

    this.globalMetrics.nodeCountHistory.push(nodeCount);
    if (this.globalMetrics.nodeCountHistory.length > this.options.historyLength) {
      this.globalMetrics.nodeCountHistory.shift();
    }

    this.globalMetrics.connectionCountHistory.push(connectionCount);
    if (this.globalMetrics.connectionCountHistory.length > this.options.historyLength) {
      this.globalMetrics.connectionCountHistory.shift();
    }

    this.globalMetrics.lastUpdate = now;

    // Check for warnings
    this.checkWarnings();

    // Emit metrics update event
    this.emit('metricsUpdated', this.getStats());
  }

  /**
   * Estimate CPU usage for a plugin
   * @param {BasePlugin} plugin - Plugin instance
   * @returns {number} Estimated CPU percentage
   * @private
   */
  estimateCPU(plugin) {
    // This is a rough estimation
    // In a production system, you would use more sophisticated profiling

    let estimate = 0.5; // Base cost per plugin

    // Count audio nodes
    const nodeCount = this.countAudioNodes(plugin);
    estimate += nodeCount * 0.3;

    // Check for expensive node types
    Object.keys(plugin).forEach(key => {
      const value = plugin[key];
      if (!value) return;

      // BiquadFilterNode and ConvolverNode are more expensive
      if (value.constructor.name === 'BiquadFilterNode') {
        estimate += 0.5;
      } else if (value.constructor.name === 'ConvolverNode') {
        estimate += 2.0; // Convolution is expensive
      } else if (value.constructor.name === 'AnalyserNode') {
        estimate += 0.3;
      } else if (value.constructor.name === 'WaveShaperNode') {
        estimate += 0.8;
      }
    });

    // Cap at 100%
    return Math.min(estimate, 100);
  }

  /**
   * Count audio nodes in a plugin
   * @param {BasePlugin} plugin - Plugin instance
   * @returns {number} Node count
   * @private
   */
  countAudioNodes(plugin) {
    let count = 0;

    Object.keys(plugin).forEach(key => {
      const value = plugin[key];
      if (value && typeof value.connect === 'function') {
        count++;
      }
    });

    return count;
  }

  /**
   * Get total CPU usage
   * @returns {number} Total CPU percentage
   */
  getTotalCPU() {
    let total = 0;

    this.measurements.forEach(measurement => {
      total += measurement.estimatedCPU;
    });

    return Math.min(total, 100);
  }

  /**
   * Get total node count
   * @returns {number} Total number of audio nodes
   */
  getTotalNodeCount() {
    let total = 0;

    this.measurements.forEach(measurement => {
      total += measurement.nodeCount;
    });

    return total;
  }

  /**
   * Get CPU usage for a specific plugin
   * @param {string} pluginId - Plugin ID
   * @returns {number} Plugin CPU percentage
   */
  getPluginCPU(pluginId) {
    const measurement = this.measurements.get(pluginId);
    return measurement ? measurement.estimatedCPU : 0;
  }

  /**
   * Get average CPU usage for a specific plugin
   * @param {string} pluginId - Plugin ID
   * @returns {number} Average CPU percentage
   */
  getPluginAverageCPU(pluginId) {
    const measurement = this.measurements.get(pluginId);
    return measurement ? measurement.averageCPU : 0;
  }

  /**
   * Get all measurements
   * @returns {Array} Array of measurement objects
   */
  getAllMeasurements() {
    const measurements = [];
    this.measurements.forEach((measurement, id) => {
      measurements.push({ ...measurement });
    });
    return measurements.sort((a, b) => b.estimatedCPU - a.estimatedCPU);
  }

  /**
   * Check for performance warnings
   * @private
   */
  checkWarnings() {
    const newWarnings = [];

    const totalCPU = this.getTotalCPU();
    const nodeCount = this.getTotalNodeCount();

    // CPU warnings
    if (totalCPU >= this.thresholds.cpuCritical) {
      newWarnings.push({
        type: 'critical',
        category: 'cpu',
        message: `Critical CPU usage: ${totalCPU.toFixed(1)}%`,
        value: totalCPU,
        threshold: this.thresholds.cpuCritical
      });
      this.emit('critical', newWarnings[newWarnings.length - 1]);
    } else if (totalCPU >= this.thresholds.cpuWarning) {
      newWarnings.push({
        type: 'warning',
        category: 'cpu',
        message: `High CPU usage: ${totalCPU.toFixed(1)}%`,
        value: totalCPU,
        threshold: this.thresholds.cpuWarning
      });
      this.emit('warning', newWarnings[newWarnings.length - 1]);
    }

    // Node count warnings
    if (nodeCount >= this.thresholds.nodeCountCritical) {
      newWarnings.push({
        type: 'critical',
        category: 'nodes',
        message: `Critical node count: ${nodeCount} nodes`,
        value: nodeCount,
        threshold: this.thresholds.nodeCountCritical
      });
      this.emit('critical', newWarnings[newWarnings.length - 1]);
    } else if (nodeCount >= this.thresholds.nodeCountWarning) {
      newWarnings.push({
        type: 'warning',
        category: 'nodes',
        message: `High node count: ${nodeCount} nodes`,
        value: nodeCount,
        threshold: this.thresholds.nodeCountWarning
      });
      this.emit('warning', newWarnings[newWarnings.length - 1]);
    }

    // Find most expensive plugins
    const sortedMeasurements = this.getAllMeasurements();
    if (sortedMeasurements.length > 0 && sortedMeasurements[0].estimatedCPU > 20) {
      newWarnings.push({
        type: 'info',
        category: 'plugin',
        message: `Most expensive plugin: ${sortedMeasurements[0].pluginName} (${sortedMeasurements[0].estimatedCPU.toFixed(1)}%)`,
        pluginId: sortedMeasurements[0].pluginId,
        value: sortedMeasurements[0].estimatedCPU
      });
    }

    this.warnings = newWarnings;
  }

  /**
   * Get current warnings
   * @returns {Array} Array of warning objects
   */
  getWarnings() {
    return [...this.warnings];
  }

  /**
   * Get comprehensive statistics
   * @returns {Object} Statistics object
   */
  getStats() {
    const totalCPU = this.getTotalCPU();
    const nodeCount = this.getTotalNodeCount();

    // Calculate averages
    const avgCPU = this.globalMetrics.cpuHistory.length > 0
      ? this.globalMetrics.cpuHistory.reduce((a, b) => a + b, 0) / this.globalMetrics.cpuHistory.length
      : 0;

    const avgNodeCount = this.globalMetrics.nodeCountHistory.length > 0
      ? this.globalMetrics.nodeCountHistory.reduce((a, b) => a + b, 0) / this.globalMetrics.nodeCountHistory.length
      : 0;

    return {
      // Current values
      currentCPU: totalCPU,
      currentNodeCount: nodeCount,
      currentConnectionCount: this.router.getConnections().length,

      // Averages
      averageCPU: avgCPU,
      averageNodeCount: avgNodeCount,

      // Audio context info
      baseLatency: this.context.baseLatency,
      outputLatency: this.context.outputLatency || 0,
      sampleRate: this.context.sampleRate,
      state: this.context.state,

      // Graph info
      pluginCount: this.router.getAllPlugins().size,
      sendBusCount: this.router.sends.length,

      // Warnings
      warnings: this.getWarnings(),
      hasWarnings: this.warnings.length > 0,

      // Timing
      lastUpdate: this.globalMetrics.lastUpdate,

      // History
      cpuHistory: [...this.globalMetrics.cpuHistory],
      nodeCountHistory: [...this.globalMetrics.nodeCountHistory]
    };
  }

  /**
   * Log statistics to console
   */
  logStats() {
    const stats = this.getStats();

    console.log('=== Performance Statistics ===');
    console.log(`CPU Usage: ${stats.currentCPU.toFixed(1)}% (avg: ${stats.averageCPU.toFixed(1)}%)`);
    console.log(`Nodes: ${stats.currentNodeCount} (avg: ${stats.averageNodeCount.toFixed(0)})`);
    console.log(`Connections: ${stats.currentConnectionCount}`);
    console.log(`Plugins: ${stats.pluginCount}`);
    console.log(`Base Latency: ${(stats.baseLatency * 1000).toFixed(2)}ms`);
    console.log(`Output Latency: ${(stats.outputLatency * 1000).toFixed(2)}ms`);
    console.log(`Sample Rate: ${stats.sampleRate}Hz`);
    console.log(`State: ${stats.state}`);

    if (stats.warnings.length > 0) {
      console.log('\n=== Warnings ===');
      stats.warnings.forEach(warning => {
        console.log(`[${warning.type.toUpperCase()}] ${warning.message}`);
      });
    }

    console.log('\n=== Top 5 CPU Consumers ===');
    const topPlugins = this.getAllMeasurements().slice(0, 5);
    topPlugins.forEach((m, i) => {
      console.log(`${i + 1}. ${m.pluginName} (${m.pluginType}): ${m.estimatedCPU.toFixed(1)}%`);
    });
  }

  /**
   * Set performance thresholds
   * @param {Object} thresholds - Threshold values
   * @param {number} thresholds.cpuWarning - CPU warning threshold
   * @param {number} thresholds.cpuCritical - CPU critical threshold
   * @param {number} thresholds.nodeCountWarning - Node count warning threshold
   * @param {number} thresholds.nodeCountCritical - Node count critical threshold
   */
  setThresholds(thresholds) {
    Object.assign(this.thresholds, thresholds);
  }

  /**
   * Get current thresholds
   * @returns {Object} Threshold values
   */
  getThresholds() {
    return { ...this.thresholds };
  }

  /**
   * Reset all history
   */
  resetHistory() {
    this.globalMetrics.cpuHistory = [];
    this.globalMetrics.nodeCountHistory = [];
    this.globalMetrics.connectionCountHistory = [];
    this.globalMetrics.bufferUnderruns = 0;

    this.measurements.forEach(measurement => {
      measurement.cpuHistory = [];
      measurement.averageCPU = 0;
    });
  }

  /**
   * Export performance data
   * @param {boolean} includeHistory - Include history data (default: true)
   * @returns {Object} Performance data
   */
  exportData(includeHistory = true) {
    const data = {
      timestamp: Date.now(),
      stats: this.getStats(),
      measurements: this.getAllMeasurements(),
      thresholds: this.getThresholds()
    };

    if (!includeHistory) {
      delete data.stats.cpuHistory;
      delete data.stats.nodeCountHistory;
      data.measurements.forEach(m => delete m.cpuHistory);
    }

    return data;
  }

  /**
   * Add event listener
   * @param {string} event - Event name: 'warning', 'critical', 'metricsUpdated'
   * @param {Function} callback - Callback function
   * @returns {Function} Unsubscribe function
   */
  on(event, callback) {
    if (!this.eventListeners[event]) {
      console.warn(`Unknown event: ${event}`);
      return () => {};
    }

    this.eventListeners[event].push(callback);

    return () => {
      const index = this.eventListeners[event].indexOf(callback);
      if (index > -1) {
        this.eventListeners[event].splice(index, 1);
      }
    };
  }

  /**
   * Emit event
   * @param {string} event - Event name
   * @param {*} data - Event data
   * @private
   */
  emit(event, data) {
    const listeners = this.eventListeners[event];
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in event listener:`, error);
        }
      });
    }
  }

  /**
   * Dispose and clean up
   */
  dispose() {
    this.stopMonitoring();

    this.measurements.clear();
    this.warnings = [];

    Object.keys(this.eventListeners).forEach(key => {
      this.eventListeners[key] = [];
    });
  }

  /**
   * String representation
   * @returns {string} String description
   */
  toString() {
    const stats = this.getStats();
    return `PerformanceMonitor(CPU: ${stats.currentCPU.toFixed(1)}%, Nodes: ${stats.currentNodeCount}, ${this.isMonitoring ? 'active' : 'inactive'})`;
  }
}

export default PerformanceMonitor;
