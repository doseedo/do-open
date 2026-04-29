/**
 * ModulationMatrix - Visual Modulation Routing System
 *
 * Features:
 * - Visual modulation routing (sources → destinations)
 * - Multiple sources to multiple destinations
 * - Modulation depth control per routing
 * - Modulation of modulation (meta-modulation)
 * - Routing presets
 * - Visual feedback and monitoring
 * - Performance mode with streamlined routing
 *
 * Inspired by: Bitwig Grid, Reaktor, VCV Rack, Vital
 *
 * @author Agent 17 (Modulation Matrix & Advanced LFOs)
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';
import PluginFactory from '../core/PluginFactory.js';

export class ModulationMatrix extends BasePlugin {
  /**
   * Routing visualization modes
   */
  static VISUALIZATION_MODES = {
    MATRIX: 'matrix',        // Grid view (sources × destinations)
    GRAPH: 'graph',          // Node graph view
    LIST: 'list'             // List view
  };

  constructor(audioContext, options = {}) {
    super(audioContext, {
      ...options,
      category: 'modulation',
      description: 'Visual modulation routing matrix'
    });

    // Registered modulation sources
    this._sources = new Map(); // id -> { source, label, color }

    // Registered modulation destinations
    this._destinations = new Map(); // id -> { param, label, plugin, paramName }

    // Active routings: routingId -> { sourceId, destId, depth, routing }
    this._routings = new Map();

    // Routing counter for unique IDs
    this._routingIdCounter = 0;

    // Visualization mode
    this._visualizationMode = ModulationMatrix.VISUALIZATION_MODES.MATRIX;

    // Performance monitoring
    this._activeRoutingCount = 0;
    this._maxRoutings = 256; // Performance limit

    // Meta-modulation support (modulating modulation depth)
    this._metaModulations = new Map(); // routingId -> { sourceId, depth }
  }

  /**
   * Register a modulation source
   * @param {string} id - Unique source identifier
   * @param {ModulationSource} source - Modulation source instance
   * @param {Object} options - Source options
   * @param {string} options.label - Display label
   * @param {string} options.color - Color for visualization
   * @returns {boolean} True if registered successfully
   */
  registerSource(id, source, options = {}) {
    if (this._sources.has(id)) {
      console.warn(`Source ${id} is already registered`);
      return false;
    }

    this._sources.set(id, {
      source,
      label: options.label || id,
      color: options.color || this._generateColor(this._sources.size)
    });

    return true;
  }

  /**
   * Unregister a modulation source
   * @param {string} id - Source identifier
   */
  unregisterSource(id) {
    // Remove all routings using this source
    const routingsToRemove = [];

    this._routings.forEach((routing, routingId) => {
      if (routing.sourceId === id) {
        routingsToRemove.push(routingId);
      }
    });

    routingsToRemove.forEach(routingId => this.removeRouting(routingId));

    // Remove source
    this._sources.delete(id);
  }

  /**
   * Register a modulation destination
   * @param {string} id - Unique destination identifier
   * @param {AudioParam} param - Target audio parameter
   * @param {Object} options - Destination options
   * @param {string} options.label - Display label
   * @param {BasePlugin} options.plugin - Plugin that owns this parameter
   * @param {string} options.paramName - Parameter name
   * @returns {boolean} True if registered successfully
   */
  registerDestination(id, param, options = {}) {
    if (!(param instanceof AudioParam)) {
      console.warn('Destination must be an AudioParam');
      return false;
    }

    if (this._destinations.has(id)) {
      console.warn(`Destination ${id} is already registered`);
      return false;
    }

    this._destinations.set(id, {
      param,
      label: options.label || id,
      plugin: options.plugin || null,
      paramName: options.paramName || 'unknown'
    });

    return true;
  }

  /**
   * Unregister a modulation destination
   * @param {string} id - Destination identifier
   */
  unregisterDestination(id) {
    // Remove all routings to this destination
    const routingsToRemove = [];

    this._routings.forEach((routing, routingId) => {
      if (routing.destId === id) {
        routingsToRemove.push(routingId);
      }
    });

    routingsToRemove.forEach(routingId => this.removeRouting(routingId));

    // Remove destination
    this._destinations.delete(id);
  }

  /**
   * Create a routing from source to destination
   * @param {string} sourceId - Source identifier
   * @param {string} destId - Destination identifier
   * @param {number} depth - Modulation depth (0-1)
   * @returns {string|null} Routing ID or null if failed
   */
  createRouting(sourceId, destId, depth = 1.0) {
    // Check if we've hit the routing limit
    if (this._routings.size >= this._maxRoutings) {
      console.warn('Maximum routing limit reached');
      return null;
    }

    // Validate source and destination
    const sourceEntry = this._sources.get(sourceId);
    const destEntry = this._destinations.get(destId);

    if (!sourceEntry) {
      console.warn(`Source ${sourceId} not found`);
      return null;
    }

    if (!destEntry) {
      console.warn(`Destination ${destId} not found`);
      return null;
    }

    // Check if routing already exists
    const existing = this._findRouting(sourceId, destId);
    if (existing) {
      console.warn('Routing already exists');
      return existing;
    }

    // Create routing
    const routingId = `routing_${this._routingIdCounter++}`;
    const routing = sourceEntry.source.routeTo(destEntry.param, depth);

    this._routings.set(routingId, {
      sourceId,
      destId,
      depth,
      routing,
      enabled: true,
      created: Date.now()
    });

    this._activeRoutingCount++;

    return routingId;
  }

  /**
   * Find existing routing between source and destination
   * @private
   * @param {string} sourceId - Source ID
   * @param {string} destId - Destination ID
   * @returns {string|null} Routing ID or null
   */
  _findRouting(sourceId, destId) {
    for (const [routingId, routing] of this._routings.entries()) {
      if (routing.sourceId === sourceId && routing.destId === destId) {
        return routingId;
      }
    }
    return null;
  }

  /**
   * Remove a routing
   * @param {string} routingId - Routing identifier
   * @returns {boolean} True if removed successfully
   */
  removeRouting(routingId) {
    const routing = this._routings.get(routingId);

    if (!routing) {
      return false;
    }

    // Disconnect the routing
    routing.routing.disconnect();

    // Remove meta-modulations for this routing
    this._metaModulations.delete(routingId);

    // Remove from map
    this._routings.delete(routingId);
    this._activeRoutingCount--;

    return true;
  }

  /**
   * Set routing depth
   * @param {string} routingId - Routing identifier
   * @param {number} depth - New depth (0-1)
   */
  setRoutingDepth(routingId, depth) {
    const routing = this._routings.get(routingId);

    if (!routing) {
      console.warn(`Routing ${routingId} not found`);
      return;
    }

    const clampedDepth = Math.max(0, Math.min(1, depth));
    routing.depth = clampedDepth;
    routing.routing.depthGain.gain.value = clampedDepth;
  }

  /**
   * Enable/disable a routing
   * @param {string} routingId - Routing identifier
   * @param {boolean} enabled - Enabled state
   */
  setRoutingEnabled(routingId, enabled) {
    const routing = this._routings.get(routingId);

    if (!routing) {
      return;
    }

    routing.enabled = enabled;

    // Set depth to 0 if disabled, restore if enabled
    if (enabled) {
      routing.routing.depthGain.gain.value = routing.depth;
    } else {
      routing.routing.depthGain.gain.value = 0;
    }
  }

  /**
   * Create meta-modulation (modulate modulation depth)
   * @param {string} routingId - Routing to modulate
   * @param {string} sourceId - Modulation source for the depth
   * @param {number} depth - Meta-modulation depth
   * @returns {boolean} True if created successfully
   */
  createMetaModulation(routingId, sourceId, depth = 1.0) {
    const routing = this._routings.get(routingId);
    const source = this._sources.get(sourceId);

    if (!routing || !source) {
      console.warn('Invalid routing or source for meta-modulation');
      return false;
    }

    // Route source to the routing's depth gain
    const metaRouting = source.source.routeTo(routing.routing.depthGain.gain, depth);

    this._metaModulations.set(routingId, {
      sourceId,
      depth,
      routing: metaRouting
    });

    return true;
  }

  /**
   * Remove meta-modulation
   * @param {string} routingId - Routing identifier
   */
  removeMetaModulation(routingId) {
    const meta = this._metaModulations.get(routingId);

    if (meta) {
      meta.routing.disconnect();
      this._metaModulations.delete(routingId);
    }
  }

  /**
   * Clear all routings
   */
  clearAllRoutings() {
    this._routings.forEach((routing, routingId) => {
      this.removeRouting(routingId);
    });

    this._routings.clear();
    this._metaModulations.clear();
    this._activeRoutingCount = 0;
  }

  /**
   * Get all sources
   * @returns {Array<Object>} Source information
   */
  getSources() {
    const sources = [];

    this._sources.forEach((entry, id) => {
      sources.push({
        id,
        label: entry.label,
        color: entry.color,
        running: entry.source.isRunning(),
        type: entry.source.constructor.name
      });
    });

    return sources;
  }

  /**
   * Get all destinations
   * @returns {Array<Object>} Destination information
   */
  getDestinations() {
    const destinations = [];

    this._destinations.forEach((entry, id) => {
      destinations.push({
        id,
        label: entry.label,
        paramName: entry.paramName,
        plugin: entry.plugin ? entry.plugin.name : 'Unknown',
        currentValue: entry.param.value
      });
    });

    return destinations;
  }

  /**
   * Get all routings
   * @returns {Array<Object>} Routing information
   */
  getRoutings() {
    const routings = [];

    this._routings.forEach((routing, id) => {
      const sourceEntry = this._sources.get(routing.sourceId);
      const destEntry = this._destinations.get(routing.destId);

      routings.push({
        id,
        sourceId: routing.sourceId,
        sourceLabel: sourceEntry?.label || 'Unknown',
        destId: routing.destId,
        destLabel: destEntry?.label || 'Unknown',
        depth: routing.depth,
        enabled: routing.enabled,
        hasMetaModulation: this._metaModulations.has(id)
      });
    });

    return routings;
  }

  /**
   * Get routing matrix (for matrix visualization)
   * @returns {Object} Matrix data
   */
  getMatrix() {
    const sources = Array.from(this._sources.keys());
    const destinations = Array.from(this._destinations.keys());
    const matrix = {};

    sources.forEach(sourceId => {
      matrix[sourceId] = {};

      destinations.forEach(destId => {
        const routingId = this._findRouting(sourceId, destId);
        matrix[sourceId][destId] = routingId ? this._routings.get(routingId).depth : 0;
      });
    });

    return {
      sources,
      destinations,
      matrix
    };
  }

  /**
   * Set visualization mode
   * @param {string} mode - Visualization mode
   */
  setVisualizationMode(mode) {
    if (!Object.values(ModulationMatrix.VISUALIZATION_MODES).includes(mode)) {
      console.warn(`Invalid visualization mode: ${mode}`);
      return;
    }

    this._visualizationMode = mode;
  }

  /**
   * Save routing preset
   * @param {string} name - Preset name
   * @returns {Object} Preset data
   */
  saveRoutingPreset(name) {
    const routings = [];

    this._routings.forEach((routing, id) => {
      routings.push({
        sourceId: routing.sourceId,
        destId: routing.destId,
        depth: routing.depth,
        enabled: routing.enabled
      });
    });

    const metaModulations = [];

    this._metaModulations.forEach((meta, routingId) => {
      metaModulations.push({
        routingId,
        sourceId: meta.sourceId,
        depth: meta.depth
      });
    });

    return {
      name,
      version: '1.0.0',
      created: Date.now(),
      routings,
      metaModulations
    };
  }

  /**
   * Load routing preset
   * @param {Object} preset - Preset data
   */
  loadRoutingPreset(preset) {
    if (!preset || !preset.routings) {
      console.warn('Invalid preset data');
      return;
    }

    // Clear existing routings
    this.clearAllRoutings();

    // Create routings
    preset.routings.forEach(r => {
      const routingId = this.createRouting(r.sourceId, r.destId, r.depth);

      if (routingId && !r.enabled) {
        this.setRoutingEnabled(routingId, false);
      }
    });

    // Create meta-modulations
    if (preset.metaModulations) {
      preset.metaModulations.forEach(m => {
        this.createMetaModulation(m.routingId, m.sourceId, m.depth);
      });
    }
  }

  /**
   * Generate random routing (for creative exploration)
   * @param {number} count - Number of routings to create
   * @param {Object} options - Randomization options
   */
  generateRandomRoutings(count, options = {}) {
    const sourceIds = Array.from(this._sources.keys());
    const destIds = Array.from(this._destinations.keys());

    if (sourceIds.length === 0 || destIds.length === 0) {
      console.warn('No sources or destinations available');
      return;
    }

    const minDepth = options.minDepth || 0.1;
    const maxDepth = options.maxDepth || 1.0;

    for (let i = 0; i < count; i++) {
      const sourceId = sourceIds[Math.floor(Math.random() * sourceIds.length)];
      const destId = destIds[Math.floor(Math.random() * destIds.length)];
      const depth = minDepth + Math.random() * (maxDepth - minDepth);

      this.createRouting(sourceId, destId, depth);
    }
  }

  /**
   * Generate color for visualization
   * @private
   * @param {number} index - Color index
   * @returns {string} CSS color
   */
  _generateColor(index) {
    const hue = (index * 137.5) % 360; // Golden angle for good distribution
    return `hsl(${hue}, 70%, 60%)`;
  }

  /**
   * Get modulation matrix info
   * @returns {Object} Matrix metadata
   */
  getInfo() {
    return {
      ...super.getInfo(),
      sourceCount: this._sources.size,
      destinationCount: this._destinations.size,
      routingCount: this._routings.size,
      activeRoutings: this._activeRoutingCount,
      metaModulationCount: this._metaModulations.size,
      visualizationMode: this._visualizationMode,
      maxRoutings: this._maxRoutings
    };
  }

  /**
   * Cleanup
   */
  dispose() {
    this.clearAllRoutings();
    this._sources.clear();
    this._destinations.clear();
    super.dispose();
  }
}

// Register with PluginFactory
PluginFactory.register('ModulationMatrix', ModulationMatrix, {
  category: 'modulation',
  description: 'Visual modulation routing matrix with meta-modulation support',
  tags: ['modulation', 'matrix', 'routing', 'meta-modulation'],
  version: '1.0.0',
  author: 'Agent 17'
});

export default ModulationMatrix;
