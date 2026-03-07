/**
 * PluginParamController - UI-side param controller for modulation routing
 * Bridges ModMatrix connections to WebAudioDSPEngine
 */

export class PluginParamControllerUI {
  constructor() {
    this._connections = [];
    this._baseValues = {};
    this._listeners = [];
  }

  // Get/set connections
  getConnections() { return this._connections; }
  
  setModConnections(connections) {
    this._connections = connections;
    this._notifyListeners();
  }

  // Base values (knob positions before modulation)
  setBaseValue(key, value) {
    this._baseValues[key] = value;
    this._notifyListeners();
  }

  getBaseValue(key) {
    return this._baseValues[key] ?? 0.5;
  }

  // Calculate modulated value (base + sum of mod depths)
  getModulatedValue(key) {
    let value = this.getBaseValue(key);
    // Add modulation from connections targeting this param
    for (const conn of this._connections) {
      if (conn.dest === key) {
        value += conn.depth * 0.5; // depth affects ±0.5 range
      }
    }
    return Math.max(0, Math.min(1, value));
  }

  // Listeners
  addListener(fn) { this._listeners.push(fn); }
  removeListener(fn) { this._listeners = this._listeners.filter(l => l !== fn); }
  _notifyListeners() { this._listeners.forEach(fn => fn()); }
}

export default PluginParamControllerUI;
