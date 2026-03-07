/**
 * Modulation Utilities - Connect MSEG/LFO to DSP params
 */

/**
 * Get all modulatable parameters from a DSP config
 * @param {Object} dspConfig - The DSP configuration object
 * @returns {Array} List of modulatable param targets
 */
export function getModulatableTargets(dspConfig) {
  const targets = [];
  
  if (!dspConfig?.parameters) return targets;
  
  Object.entries(dspConfig.parameters).forEach(([key, param]) => {
    if (param.modulatable !== false) { // Default to modulatable
      targets.push({
        id: key,
        name: param.name || key,
        min: param.min ?? 0,
        max: param.max ?? 1,
        unit: param.unit || '',
        currentValue: param.value ?? param.default ?? 0.5,
      });
    }
  });
  
  return targets;
}

/**
 * Create a modulation connection
 * @param {string} sourceId - MSEG/LFO id
 * @param {string} targetId - Parameter id
 * @param {number} depth - Modulation depth (-1 to 1)
 * @returns {Object} Modulation connection object
 */
export function createModConnection(sourceId, targetId, depth = 0.5) {
  return {
    id: `mod_${sourceId}_${targetId}_${Date.now()}`,
    source: sourceId,
    target: targetId,
    depth: Math.max(-1, Math.min(1, depth)),
    bipolar: depth < 0,
    active: true,
  };
}

/**
 * Apply modulation to a parameter value
 * @param {number} baseValue - Original parameter value (0-1 normalized)
 * @param {number} modValue - Modulation source value (0-1)
 * @param {number} depth - Modulation depth (-1 to 1)
 * @param {Object} param - Parameter config with min/max
 * @returns {number} Modulated value
 */
export function applyModulation(baseValue, modValue, depth, param) {
  const range = (param.max ?? 1) - (param.min ?? 0);
  const modAmount = (modValue - 0.5) * 2 * depth * range;
  const result = baseValue + modAmount;
  return Math.max(param.min ?? 0, Math.min(param.max ?? 1, result));
}

/**
 * Get modulation sources available in the current config
 * @param {Object} dspConfig - DSP configuration
 * @returns {Array} List of modulation sources
 */
export function getModulationSources(dspConfig) {
  const sources = [
    { id: 'env1', name: 'Envelope 1', type: 'envelope' },
    { id: 'env2', name: 'Envelope 2', type: 'envelope' },
    { id: 'lfo1', name: 'LFO 1', type: 'lfo' },
    { id: 'lfo2', name: 'LFO 2', type: 'lfo' },
    { id: 'mseg1', name: 'MSEG 1', type: 'mseg' },
    { id: 'velocity', name: 'Velocity', type: 'midi' },
    { id: 'modwheel', name: 'Mod Wheel', type: 'midi' },
    { id: 'aftertouch', name: 'Aftertouch', type: 'midi' },
  ];
  
  // Add any custom sources from config
  if (dspConfig?.modulationSources) {
    sources.push(...dspConfig.modulationSources);
  }
  
  return sources;
}

/**
 * Serialize modulation matrix for DSP export
 * @param {Array} connections - Array of mod connections
 * @returns {Object} Serialized mod matrix for JUCE codegen
 */
export function serializeModMatrix(connections) {
  return {
    connections: connections.map(c => ({
      src: c.source,
      dst: c.target,
      amt: c.depth,
    })),
    count: connections.length,
  };
}

export default {
  getModulatableTargets,
  createModConnection,
  applyModulation,
  getModulationSources,
  serializeModMatrix,
};
