/**
 * Router - Master signal flow graph for complex plugin chains
 *
 * Handles arbitrary plugin connections, parallel and serial routing,
 * send/return buses, circular dependency detection, and gain staging.
 *
 * @class Router
 * @author Agent 10 - Integration & Routing System
 */

class Router {
  /**
   * Create a new Router instance
   * @param {AudioContext} audioContext - The Web Audio API context
   * @param {Object} options - Configuration options
   * @param {number} options.sendCount - Number of send/return buses (default: 4)
   * @param {boolean} options.autoConnect - Auto-connect new plugins to master (default: false)
   */
  constructor(audioContext, options = {}) {
    if (!audioContext) {
      throw new Error('AudioContext is required');
    }

    this.context = audioContext;
    this.options = {
      sendCount: options.sendCount || 4,
      autoConnect: options.autoConnect || false
    };

    // Plugin registry
    this.plugins = new Map();

    // Connection graph
    this.connections = [];

    // Send/return buses
    this.sends = [];
    this.returns = [];
    this.sendEffects = new Map(); // sendIndex -> plugin

    // Master input/output
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();
    this.masterGain = audioContext.createGain();

    // Master gain defaults to unity
    this.masterGain.gain.value = 1;
    this.masterGain.connect(this.output);

    // Create send/return buses
    this.createSendReturns(this.options.sendCount);

    // Event listeners
    this.eventListeners = {
      pluginAdded: [],
      pluginRemoved: [],
      connectionMade: [],
      connectionRemoved: []
    };
  }

  /**
   * Create send and return buses
   * @param {number} count - Number of send/return pairs to create
   * @private
   */
  createSendReturns(count) {
    for (let i = 0; i < count; i++) {
      // Send bus (pre-fader sends)
      const send = this.context.createGain();
      send.gain.value = 0; // Start at 0, controlled per-plugin

      // Return bus
      const returnBus = this.context.createGain();
      returnBus.gain.value = 1;

      this.sends.push(send);
      this.returns.push(returnBus);

      // Returns feed into master output
      returnBus.connect(this.masterGain);
    }
  }

  /**
   * Add a plugin to the router
   * @param {BasePlugin} plugin - Plugin instance to add
   * @param {string} id - Optional custom ID (uses plugin.id if not provided)
   * @returns {string|null} Plugin ID or null on failure
   */
  addPlugin(plugin, id = null) {
    if (!plugin) {
      console.error('Cannot add null plugin');
      return null;
    }

    const pluginId = id || plugin.id;

    if (this.plugins.has(pluginId)) {
      console.warn(`Plugin with ID ${pluginId} already exists`);
      return null;
    }

    this.plugins.set(pluginId, plugin);

    // Auto-connect to master if enabled
    if (this.options.autoConnect) {
      plugin.connect(this.masterGain);
    }

    // Emit event
    this.emit('pluginAdded', { id: pluginId, plugin });

    return pluginId;
  }

  /**
   * Remove a plugin from the router
   * @param {string} pluginId - ID of plugin to remove
   * @param {boolean} dispose - Whether to dispose the plugin (default: true)
   * @returns {boolean} Success status
   */
  removePlugin(pluginId, dispose = true) {
    const plugin = this.plugins.get(pluginId);
    if (!plugin) {
      console.warn(`Plugin ${pluginId} not found`);
      return false;
    }

    // Remove all connections involving this plugin
    this.connections = this.connections.filter(conn => {
      const shouldRemove = conn.source !== pluginId && conn.destination !== pluginId;
      if (!shouldRemove) {
        this.emit('connectionRemoved', conn);
      }
      return shouldRemove;
    });

    // Disconnect the plugin
    plugin.disconnect();

    // Dispose if requested
    if (dispose) {
      plugin.dispose();
    }

    this.plugins.delete(pluginId);

    // Emit event
    this.emit('pluginRemoved', { id: pluginId, plugin });

    return true;
  }

  /**
   * Get a plugin by ID
   * @param {string} pluginId - Plugin ID
   * @returns {BasePlugin|null} Plugin instance or null
   */
  getPlugin(pluginId) {
    return this.plugins.get(pluginId) || null;
  }

  /**
   * Get all plugins
   * @returns {Map} Map of plugin ID -> plugin instance
   */
  getAllPlugins() {
    return new Map(this.plugins);
  }

  /**
   * Connect two plugins
   * @param {string} sourceId - Source plugin ID
   * @param {string} destinationId - Destination plugin ID or 'master'
   * @param {Object} options - Connection options
   * @param {number} options.gain - Gain to apply to connection (default: 1)
   * @returns {boolean} Success status
   */
  connect(sourceId, destinationId, options = {}) {
    const source = this.plugins.get(sourceId);

    if (!source) {
      console.error(`Source plugin ${sourceId} not found`);
      return false;
    }

    let destination;
    if (destinationId === 'master') {
      destination = this.masterGain;
    } else {
      destination = this.plugins.get(destinationId);
      if (!destination) {
        console.error(`Destination plugin ${destinationId} not found`);
        return false;
      }
    }

    // Check for circular dependencies (only if not connecting to master)
    if (destinationId !== 'master' && this.wouldCreateCycle(sourceId, destinationId)) {
      console.error(`Connection from ${sourceId} to ${destinationId} would create a circular dependency`);
      return false;
    }

    // Create connection with optional gain staging
    if (options.gain !== undefined && options.gain !== 1) {
      const gainNode = this.context.createGain();
      gainNode.gain.value = options.gain;

      source.connect(gainNode);
      if (destinationId === 'master') {
        gainNode.connect(destination);
      } else {
        gainNode.connect(destination);
      }
    } else {
      source.connect(destination);
    }

    // Record connection
    const connection = {
      source: sourceId,
      destination: destinationId,
      gain: options.gain || 1,
      timestamp: Date.now()
    };

    this.connections.push(connection);

    // Emit event
    this.emit('connectionMade', connection);

    return true;
  }

  /**
   * Disconnect two plugins
   * @param {string} sourceId - Source plugin ID
   * @param {string} destinationId - Destination plugin ID or null for all
   * @returns {boolean} Success status
   */
  disconnect(sourceId, destinationId = null) {
    const source = this.plugins.get(sourceId);
    if (!source) {
      console.warn(`Source plugin ${sourceId} not found`);
      return false;
    }

    if (destinationId) {
      let destination;
      if (destinationId === 'master') {
        destination = this.masterGain;
      } else {
        destination = this.plugins.get(destinationId);
      }

      if (destination) {
        source.disconnect(destination);

        // Remove from connection list
        this.connections = this.connections.filter(conn => {
          const shouldRemove = conn.source === sourceId && conn.destination === destinationId;
          if (shouldRemove) {
            this.emit('connectionRemoved', conn);
          }
          return !shouldRemove;
        });
      }
    } else {
      // Disconnect all
      source.disconnect();

      // Remove all connections from this source
      this.connections = this.connections.filter(conn => {
        const shouldRemove = conn.source === sourceId;
        if (shouldRemove) {
          this.emit('connectionRemoved', conn);
        }
        return !shouldRemove;
      });
    }

    return true;
  }

  /**
   * Check if a connection would create a cycle
   * @param {string} sourceId - Source plugin ID
   * @param {string} destinationId - Destination plugin ID
   * @returns {boolean} True if cycle would be created
   * @private
   */
  wouldCreateCycle(sourceId, destinationId) {
    // Depth-first search to detect cycles
    const visited = new Set();
    const stack = [destinationId];

    while (stack.length > 0) {
      const current = stack.pop();

      if (current === sourceId) {
        return true; // Cycle detected
      }

      if (visited.has(current)) {
        continue;
      }

      visited.add(current);

      // Find all outgoing connections from current
      const outgoing = this.connections
        .filter(conn => conn.source === current)
        .map(conn => conn.destination)
        .filter(dest => dest !== 'master'); // Ignore master connections

      stack.push(...outgoing);
    }

    return false;
  }

  /**
   * Connect a plugin to a send bus
   * @param {string} pluginId - Plugin ID
   * @param {number} sendIndex - Send bus index (0-based)
   * @param {number} amount - Send amount (0-1, default: 1)
   * @returns {boolean} Success status
   */
  connectToSend(pluginId, sendIndex, amount = 1.0) {
    const plugin = this.plugins.get(pluginId);
    const send = this.sends[sendIndex];

    if (!plugin) {
      console.error(`Plugin ${pluginId} not found`);
      return false;
    }

    if (!send) {
      console.error(`Send bus ${sendIndex} does not exist`);
      return false;
    }

    // Create a gain node for this send amount
    const sendGain = this.context.createGain();
    sendGain.gain.value = amount;

    plugin.output.connect(sendGain);
    sendGain.connect(send);

    return true;
  }

  /**
   * Set send amount for a plugin
   * @param {string} pluginId - Plugin ID
   * @param {number} sendIndex - Send bus index
   * @param {number} amount - New send amount (0-1)
   * @param {number} rampTime - Ramp time in seconds (default: 0)
   */
  setSendAmount(pluginId, sendIndex, amount, rampTime = 0) {
    // This would require tracking send gain nodes per plugin
    // For now, reconnect with new amount
    // In a production system, we'd store references to send gain nodes
    console.warn('Dynamic send amount not fully implemented - use connectToSend');
  }

  /**
   * Add an effect to a send bus
   * @param {number} sendIndex - Send bus index
   * @param {BasePlugin} effect - Effect plugin
   * @returns {boolean} Success status
   */
  addSendEffect(sendIndex, effect) {
    const send = this.sends[sendIndex];
    const returnBus = this.returns[sendIndex];

    if (!send || !returnBus) {
      console.error(`Send/return ${sendIndex} does not exist`);
      return false;
    }

    // Connect: send -> effect -> return
    send.connect(effect);
    effect.connect(returnBus);

    this.sendEffects.set(sendIndex, effect);

    return true;
  }

  /**
   * Get send bus
   * @param {number} index - Send bus index
   * @returns {GainNode|null} Send bus or null
   */
  getSendBus(index) {
    return this.sends[index] || null;
  }

  /**
   * Get return bus
   * @param {number} index - Return bus index
   * @returns {GainNode|null} Return bus or null
   */
  getReturnBus(index) {
    return this.returns[index] || null;
  }

  /**
   * Get topological processing order of plugins
   * @returns {string[]} Array of plugin IDs in processing order
   */
  getProcessingOrder() {
    return this.topologicalSort();
  }

  /**
   * Perform topological sort of plugin graph using Kahn's algorithm
   * @returns {string[]} Sorted plugin IDs
   * @private
   */
  topologicalSort() {
    const inDegree = new Map();
    const queue = [];
    const result = [];

    // Initialize in-degrees
    this.plugins.forEach((plugin, id) => {
      inDegree.set(id, 0);
    });

    // Calculate in-degrees (ignore master connections)
    this.connections.forEach(conn => {
      if (conn.destination !== 'master') {
        inDegree.set(conn.destination, (inDegree.get(conn.destination) || 0) + 1);
      }
    });

    // Find nodes with no incoming edges
    inDegree.forEach((degree, id) => {
      if (degree === 0) {
        queue.push(id);
      }
    });

    // Kahn's algorithm
    while (queue.length > 0) {
      const current = queue.shift();
      result.push(current);

      // Find outgoing edges
      this.connections
        .filter(conn => conn.source === current && conn.destination !== 'master')
        .forEach(conn => {
          const newDegree = inDegree.get(conn.destination) - 1;
          inDegree.set(conn.destination, newDegree);

          if (newDegree === 0) {
            queue.push(conn.destination);
          }
        });
    }

    return result;
  }

  /**
   * Get all connections
   * @returns {Array} Array of connection objects
   */
  getConnections() {
    return [...this.connections];
  }

  /**
   * Save entire routing chain as preset
   * @param {string} name - Chain name
   * @returns {Object} Chain preset object
   */
  saveChain(name = 'Untitled Chain') {
    const chain = {
      name: name,
      version: '1.0.0',
      plugins: [],
      connections: this.connections.map(conn => ({ ...conn })),
      masterGain: this.masterGain.gain.value,
      timestamp: Date.now()
    };

    this.plugins.forEach((plugin, id) => {
      chain.plugins.push({
        id: id,
        preset: plugin.savePreset()
      });
    });

    return chain;
  }

  /**
   * Load a routing chain preset
   * @param {Object} chain - Chain preset object
   * @param {Function} pluginFactory - Factory function to create plugins: (type, context) => plugin
   * @param {boolean} clearExisting - Clear existing plugins first (default: true)
   * @returns {boolean} Success status
   */
  loadChain(chain, pluginFactory, clearExisting = true) {
    if (!chain || !pluginFactory) {
      console.error('Chain and pluginFactory are required');
      return false;
    }

    try {
      // Clear existing
      if (clearExisting) {
        this.clear();
      }

      // Create plugins
      chain.plugins.forEach(pluginData => {
        const plugin = pluginFactory(pluginData.preset.type, this.context);
        if (plugin) {
          plugin.loadPreset(pluginData.preset);
          this.addPlugin(plugin, pluginData.id);
        }
      });

      // Restore connections
      chain.connections.forEach(conn => {
        this.connect(conn.source, conn.destination, {
          gain: conn.gain
        });
      });

      // Restore master gain
      if (chain.masterGain !== undefined) {
        this.masterGain.gain.value = chain.masterGain;
      }

      return true;
    } catch (error) {
      console.error('Error loading chain:', error);
      return false;
    }
  }

  /**
   * Clear all plugins and connections
   */
  clear() {
    // Remove all plugins
    const pluginIds = Array.from(this.plugins.keys());
    pluginIds.forEach(id => this.removePlugin(id));

    this.connections = [];
  }

  /**
   * Set master gain
   * @param {number} gain - Master gain value
   * @param {number} rampTime - Ramp time in seconds (default: 0)
   */
  setMasterGain(gain, rampTime = 0) {
    const now = this.context.currentTime;

    if (rampTime === 0) {
      this.masterGain.gain.value = gain;
    } else {
      this.masterGain.gain.setValueAtTime(this.masterGain.gain.value, now);
      this.masterGain.gain.linearRampToValueAtTime(gain, now + rampTime);
    }
  }

  /**
   * Get master gain
   * @returns {number} Current master gain value
   */
  getMasterGain() {
    return this.masterGain.gain.value;
  }

  /**
   * Get graph statistics
   * @returns {Object} Graph statistics
   */
  getStats() {
    return {
      pluginCount: this.plugins.size,
      connectionCount: this.connections.length,
      sendBusCount: this.sends.length,
      processingOrder: this.getProcessingOrder(),
      masterGain: this.getMasterGain()
    };
  }

  /**
   * Add event listener
   * @param {string} event - Event name
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
   * Emit event to all listeners
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
   * Dispose and clean up all resources
   */
  dispose() {
    // Clear all plugins
    this.clear();

    // Disconnect buses
    this.sends.forEach(send => send.disconnect());
    this.returns.forEach(returnBus => returnBus.disconnect());

    // Disconnect master
    this.input.disconnect();
    this.output.disconnect();
    this.masterGain.disconnect();

    // Clear event listeners
    Object.keys(this.eventListeners).forEach(key => {
      this.eventListeners[key] = [];
    });
  }

  /**
   * Export graph as DOT format for visualization
   * @returns {string} DOT format string
   */
  toDOT() {
    let dot = 'digraph PluginGraph {\n';
    dot += '  rankdir=LR;\n';
    dot += '  node [shape=box];\n\n';

    // Add nodes
    this.plugins.forEach((plugin, id) => {
      dot += `  "${id}" [label="${plugin.name}\\n(${plugin.constructor.name})"];\n`;
    });

    dot += '  "master" [shape=ellipse, label="Master Out"];\n\n';

    // Add edges
    this.connections.forEach(conn => {
      const label = conn.gain !== 1 ? `[label="${conn.gain.toFixed(2)}"]` : '';
      dot += `  "${conn.source}" -> "${conn.destination}" ${label};\n`;
    });

    dot += '}';
    return dot;
  }

  /**
   * String representation
   * @returns {string} String description
   */
  toString() {
    return `Router(${this.plugins.size} plugins, ${this.connections.length} connections)`;
  }
}

export default Router;
