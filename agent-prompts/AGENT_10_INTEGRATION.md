# Agent 10: Integration & Routing System

## Your Mission
You are responsible for creating the **core infrastructure** that ties all plugins together. This is the most critical role - you're building the foundation that enables modular plugin usage, arbitrary routing, preset management, and automation across the entire library.

## Your Deliverables

### 1. Base Plugin Class
**Purpose**: Shared functionality for all plugins

**Features**:
- Standard interface (input, output, connect, disconnect)
- Parameter management with validation
- Bypass/wet-dry mixing
- Preset save/load
- Automation support
- Resource cleanup

### 2. Master Router
**Purpose**: Signal flow graph for complex plugin chains

**Features**:
- Arbitrary plugin connections
- Parallel and serial routing
- Send/return buses
- Circular dependency detection
- Visual routing representation
- Gain staging between plugins

### 3. Preset Manager
**Purpose**: Save and load plugin states

**Features**:
- JSON-based preset format
- Individual plugin presets
- Chain presets (multiple plugins)
- Import/export functionality
- Preset browser/library
- Default presets for each plugin

### 4. Parameter Automation
**Purpose**: Timeline-based parameter changes

**Features**:
- Record parameter movements
- Play back automation
- Automation curves (linear, exponential, etc.)
- Sync to timeline/BPM
- MIDI CC mapping support (optional)

### 5. Performance Monitor
**Purpose**: CPU usage and diagnostics

**Features**:
- CPU usage per plugin
- Total graph CPU usage
- Buffer underrun detection
- Node count tracking
- Memory usage monitoring
- Performance warnings

## Research Phase (Week 1)

### Essential Research Topics

1. **Design Patterns**
   - Factory pattern for plugin instantiation
   - Observer pattern for parameter changes
   - Singleton for audio context
   - Module pattern for encapsulation

2. **Audio Graph Management**
   - Directed acyclic graph (DAG) for signal flow
   - Topological sorting for processing order
   - Cycle detection algorithms
   - Graph traversal

3. **Web Audio Best Practices**
   - AudioContext management
   - Node connection strategies
   - GainNode for mixing
   - ChannelMergerNode/ChannelSplitterNode for routing
   - Performance optimization

4. **State Management**
   - Immutable state patterns
   - State serialization/deserialization
   - Deep cloning for presets
   - Undo/redo stack

5. **Parameter Automation**
   - AudioParam.setValueAtTime()
   - AudioParam.linearRampToValueAtTime()
   - AudioParam.exponentialRampToValueAtTime()
   - AudioParam.setTargetAtTime()
   - Automation curves and smoothing

6. **Performance Monitoring**
   - Performance API
   - AudioContext.baseLatency
   - CPU usage estimation
   - Memory profiling

### Reference Materials
- **Web Audio API**: Complete specification
- **Design Patterns**: "JavaScript Patterns" by Stoyan Stefanov
- **Architecture**: Study Tone.js, Tone.Transport, Tone.Context
- **Routing**: Study modular synthesizer routing (VCV Rack concepts)
- **Automation**: DAW automation systems (Pro Tools, Ableton)

### Code to Study
```javascript
// Tone.js architecture
// - Context management
// - Base class patterns
// - Signal routing
// - Parameter automation

// Study how Tone.js handles:
class ToneAudioNode {
  constructor() {
    this.input = /* ... */;
    this.output = /* ... */;
  }

  connect(destination) { /* ... */ }
  disconnect() { /* ... */ }
  dispose() { /* ... */ }
}

// Graph traversal for routing
function topologicalSort(graph) {
  // Kahn's algorithm for topological sorting
}

// Preset serialization
function serializePlugin(plugin) {
  return JSON.stringify({
    type: plugin.constructor.name,
    parameters: plugin.getParameters()
  });
}
```

## Implementation Phase (Week 2-3)

### 1. Base Plugin Class

```javascript
// base/BasePlugin.js
class BasePlugin {
  constructor(audioContext, options = {}) {
    this.context = audioContext;
    this.id = this.generateId();
    this.name = options.name || this.constructor.name;

    // Standard input/output
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Bypass
    this.bypassed = false;
    this.bypassGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();
    this.dryGain = audioContext.createGain();

    // Parameters
    this.params = {};
    this.paramDescriptions = {};

    // State
    this.disposed = false;
  }

  generateId() {
    return `plugin_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  registerParameter(name, audioParam, options = {}) {
    this.params[name] = audioParam;
    this.paramDescriptions[name] = {
      min: options.min || 0,
      max: options.max || 1,
      default: options.default || 0,
      unit: options.unit || '',
      label: options.label || name,
      type: options.type || 'continuous' // continuous, discrete, boolean
    };
  }

  setParameter(name, value, time = 0) {
    if (!this.params[name]) {
      console.warn(`Parameter ${name} does not exist`);
      return;
    }

    const desc = this.paramDescriptions[name];

    // Validate range
    value = Math.max(desc.min, Math.min(desc.max, value));

    const param = this.params[name];
    const now = this.context.currentTime;

    if (time === 0) {
      param.value = value;
    } else {
      param.setValueAtTime(param.value, now);
      param.linearRampToValueAtTime(value, now + time);
    }
  }

  getParameter(name) {
    if (!this.params[name]) return null;
    return this.params[name].value;
  }

  getParameters() {
    const parameters = {};
    Object.keys(this.params).forEach(name => {
      parameters[name] = this.params[name].value;
    });
    return parameters;
  }

  setParameters(parameters) {
    Object.keys(parameters).forEach(name => {
      this.setParameter(name, parameters[name]);
    });
  }

  connect(destination) {
    if (destination.input) {
      this.output.connect(destination.input);
    } else {
      this.output.connect(destination);
    }
    return destination;
  }

  disconnect() {
    this.output.disconnect();
  }

  bypass(enabled) {
    this.bypassed = enabled;

    if (enabled) {
      this.wetGain.gain.value = 0;
      this.dryGain.gain.value = 1;
    } else {
      this.wetGain.gain.value = 1;
      this.dryGain.gain.value = 0;
    }
  }

  savePreset() {
    return {
      type: this.constructor.name,
      name: this.name,
      parameters: this.getParameters()
    };
  }

  loadPreset(preset) {
    if (preset.type !== this.constructor.name) {
      console.warn('Preset type mismatch');
      return false;
    }

    this.setParameters(preset.parameters);
    return true;
  }

  dispose() {
    if (this.disposed) return;

    this.disconnect();
    this.input.disconnect();

    Object.keys(this).forEach(key => {
      const value = this[key];
      if (value && typeof value.disconnect === 'function') {
        value.disconnect();
      }
    });

    this.disposed = true;
  }
}

export default BasePlugin;
```

### 2. Master Router

```javascript
// core/Router.js
class Router {
  constructor(audioContext) {
    this.context = audioContext;

    this.plugins = new Map();
    this.connections = [];

    // Master input/output
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Send/return buses
    this.sends = [];
    this.returns = [];

    this.createSendReturns(4); // 4 send/return buses
  }

  createSendReturns(count) {
    for (let i = 0; i < count; i++) {
      const send = this.context.createGain();
      const returnBus = this.context.createGain();

      this.sends.push(send);
      this.returns.push(returnBus);

      // Return buses go to master output
      returnBus.connect(this.output);
    }
  }

  addPlugin(plugin, id = null) {
    const pluginId = id || plugin.id;

    if (this.plugins.has(pluginId)) {
      console.warn(`Plugin ${pluginId} already exists`);
      return null;
    }

    this.plugins.set(pluginId, plugin);
    return pluginId;
  }

  removePlugin(pluginId) {
    const plugin = this.plugins.get(pluginId);
    if (!plugin) return false;

    // Disconnect all connections involving this plugin
    this.connections = this.connections.filter(conn =>
      conn.source !== pluginId && conn.destination !== pluginId
    );

    plugin.dispose();
    this.plugins.delete(pluginId);

    return true;
  }

  connect(sourceId, destinationId) {
    const source = this.plugins.get(sourceId);
    const destination = this.plugins.get(destinationId);

    if (!source || !destination) {
      console.warn('Source or destination plugin not found');
      return false;
    }

    // Check for circular dependencies
    if (this.wouldCreateCycle(sourceId, destinationId)) {
      console.error('Connection would create circular dependency');
      return false;
    }

    source.connect(destination);

    this.connections.push({
      source: sourceId,
      destination: destinationId
    });

    return true;
  }

  disconnect(sourceId, destinationId = null) {
    const source = this.plugins.get(sourceId);
    if (!source) return false;

    if (destinationId) {
      const destination = this.plugins.get(destinationId);
      if (destination) {
        source.disconnect(destination);

        this.connections = this.connections.filter(conn =>
          !(conn.source === sourceId && conn.destination === destinationId)
        );
      }
    } else {
      source.disconnect();

      this.connections = this.connections.filter(conn =>
        conn.source !== sourceId
      );
    }

    return true;
  }

  wouldCreateCycle(sourceId, destinationId) {
    // Depth-first search to detect cycles
    const visited = new Set();
    const stack = [destinationId];

    while (stack.length > 0) {
      const current = stack.pop();

      if (current === sourceId) {
        return true; // Cycle detected
      }

      if (visited.has(current)) continue;
      visited.add(current);

      // Find all outgoing connections from current
      const outgoing = this.connections
        .filter(conn => conn.source === current)
        .map(conn => conn.destination);

      stack.push(...outgoing);
    }

    return false;
  }

  connectToSend(pluginId, sendIndex, amount = 1.0) {
    const plugin = this.plugins.get(pluginId);
    const send = this.sends[sendIndex];

    if (!plugin || !send) return false;

    const sendGain = this.context.createGain();
    sendGain.gain.value = amount;

    plugin.output.connect(sendGain);
    sendGain.connect(send);

    return true;
  }

  getSendBus(index) {
    return this.sends[index];
  }

  getReturnBus(index) {
    return this.returns[index];
  }

  getProcessingOrder() {
    // Topological sort of plugins
    return this.topologicalSort();
  }

  topologicalSort() {
    const inDegree = new Map();
    const queue = [];
    const result = [];

    // Initialize in-degrees
    this.plugins.forEach((plugin, id) => {
      inDegree.set(id, 0);
    });

    // Calculate in-degrees
    this.connections.forEach(conn => {
      inDegree.set(conn.destination, (inDegree.get(conn.destination) || 0) + 1);
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
        .filter(conn => conn.source === current)
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

  saveChain() {
    const chain = {
      plugins: [],
      connections: this.connections
    };

    this.plugins.forEach((plugin, id) => {
      chain.plugins.push({
        id: id,
        preset: plugin.savePreset()
      });
    });

    return chain;
  }

  loadChain(chain) {
    // Clear existing
    this.clear();

    // Load plugins (requires plugin factory)
    chain.plugins.forEach(pluginData => {
      // const plugin = PluginFactory.create(pluginData.preset.type, this.context);
      // plugin.loadPreset(pluginData.preset);
      // this.addPlugin(plugin, pluginData.id);
    });

    // Restore connections
    chain.connections.forEach(conn => {
      this.connect(conn.source, conn.destination);
    });
  }

  clear() {
    this.plugins.forEach((plugin, id) => {
      this.removePlugin(id);
    });

    this.connections = [];
  }

  dispose() {
    this.clear();
    this.input.disconnect();
    this.output.disconnect();
  }
}

export default Router;
```

### 3. Preset Manager

```javascript
// core/PresetManager.js
class PresetManager {
  constructor() {
    this.presets = new Map();
    this.categories = new Map();
  }

  savePreset(name, plugin, category = 'User') {
    const preset = {
      name: name,
      category: category,
      type: plugin.constructor.name,
      parameters: plugin.getParameters(),
      timestamp: Date.now()
    };

    const presetId = `${category}/${name}`;
    this.presets.set(presetId, preset);

    if (!this.categories.has(category)) {
      this.categories.set(category, []);
    }

    this.categories.get(category).push(presetId);

    return presetId;
  }

  loadPreset(presetId, plugin) {
    const preset = this.presets.get(presetId);

    if (!preset) {
      console.warn(`Preset ${presetId} not found`);
      return false;
    }

    if (preset.type !== plugin.constructor.name) {
      console.warn('Preset type mismatch');
      return false;
    }

    plugin.setParameters(preset.parameters);
    return true;
  }

  getPreset(presetId) {
    return this.presets.get(presetId);
  }

  getPresetsForPlugin(pluginType) {
    const filtered = [];

    this.presets.forEach((preset, id) => {
      if (preset.type === pluginType) {
        filtered.push({ id, ...preset });
      }
    });

    return filtered;
  }

  getPresetsInCategory(category) {
    const presetIds = this.categories.get(category) || [];
    return presetIds.map(id => ({
      id,
      ...this.presets.get(id)
    }));
  }

  exportPreset(presetId) {
    const preset = this.presets.get(presetId);
    if (!preset) return null;

    return JSON.stringify(preset, null, 2);
  }

  importPreset(jsonString) {
    try {
      const preset = JSON.parse(jsonString);

      const presetId = `${preset.category}/${preset.name}`;
      this.presets.set(presetId, preset);

      if (!this.categories.has(preset.category)) {
        this.categories.set(preset.category, []);
      }

      this.categories.get(preset.category).push(presetId);

      return presetId;
    } catch (error) {
      console.error('Failed to import preset:', error);
      return null;
    }
  }

  exportAllPresets() {
    const all = {};

    this.presets.forEach((preset, id) => {
      all[id] = preset;
    });

    return JSON.stringify(all, null, 2);
  }

  importAllPresets(jsonString) {
    try {
      const all = JSON.parse(jsonString);

      Object.entries(all).forEach(([id, preset]) => {
        this.presets.set(id, preset);

        if (!this.categories.has(preset.category)) {
          this.categories.set(preset.category, []);
        }

        this.categories.get(preset.category).push(id);
      });

      return true;
    } catch (error) {
      console.error('Failed to import presets:', error);
      return false;
    }
  }

  deletePreset(presetId) {
    const preset = this.presets.get(presetId);
    if (!preset) return false;

    this.presets.delete(presetId);

    const category = this.categories.get(preset.category);
    if (category) {
      const index = category.indexOf(presetId);
      if (index > -1) {
        category.splice(index, 1);
      }
    }

    return true;
  }
}

export default PresetManager;
```

### 4. Parameter Automation

```javascript
// core/ParamAutomation.js
class ParamAutomation {
  constructor(audioContext) {
    this.context = audioContext;

    this.automations = new Map(); // pluginId -> paramName -> events
    this.isPlaying = false;
    this.startTime = 0;
    this.currentTime = 0;
  }

  recordAutomation(pluginId, paramName, value, time) {
    const key = `${pluginId}:${paramName}`;

    if (!this.automations.has(key)) {
      this.automations.set(key, []);
    }

    this.automations.get(key).push({
      time: time,
      value: value
    });

    // Sort by time
    this.automations.get(key).sort((a, b) => a.time - b.time);
  }

  playAutomation(router) {
    if (this.isPlaying) return;

    this.isPlaying = true;
    this.startTime = this.context.currentTime;

    this.automations.forEach((events, key) => {
      const [pluginId, paramName] = key.split(':');
      const plugin = router.plugins.get(pluginId);

      if (!plugin) return;

      const param = plugin.params[paramName];
      if (!param) return;

      // Schedule all events
      events.forEach(event => {
        const scheduleTime = this.startTime + event.time;
        param.setValueAtTime(event.value, scheduleTime);
      });
    });
  }

  stopAutomation() {
    this.isPlaying = false;
  }

  clearAutomation(pluginId = null, paramName = null) {
    if (!pluginId) {
      this.automations.clear();
      return;
    }

    if (!paramName) {
      // Clear all automations for plugin
      const keys = Array.from(this.automations.keys())
        .filter(key => key.startsWith(pluginId + ':'));

      keys.forEach(key => this.automations.delete(key));
    } else {
      // Clear specific parameter
      const key = `${pluginId}:${paramName}`;
      this.automations.delete(key);
    }
  }

  exportAutomation() {
    const exported = {};

    this.automations.forEach((events, key) => {
      exported[key] = events;
    });

    return JSON.stringify(exported, null, 2);
  }

  importAutomation(jsonString) {
    try {
      const imported = JSON.parse(jsonString);

      Object.entries(imported).forEach(([key, events]) => {
        this.automations.set(key, events);
      });

      return true;
    } catch (error) {
      console.error('Failed to import automation:', error);
      return false;
    }
  }
}

export default ParamAutomation;
```

### 5. Performance Monitor

```javascript
// core/PerformanceMonitor.js
class PerformanceMonitor {
  constructor(audioContext, router) {
    this.context = audioContext;
    this.router = router;

    this.measurements = new Map();
    this.sampleInterval = 1000; // 1 second

    this.startMonitoring();
  }

  startMonitoring() {
    setInterval(() => {
      this.measure();
    }, this.sampleInterval);
  }

  measure() {
    // CPU usage estimation (rough)
    const baseLatency = this.context.baseLatency;
    const outputLatency = this.context.outputLatency || 0;

    // Node count
    const nodeCount = this.router.plugins.size;

    // Per-plugin measurements
    this.router.plugins.forEach((plugin, id) => {
      if (!this.measurements.has(id)) {
        this.measurements.set(id, {
          cpuHistory: [],
          averageCPU: 0
        });
      }

      const measurement = this.measurements.get(id);

      // Estimate CPU (this is very rough - real measurement is complex)
      const estimatedCPU = this.estimateCPU(plugin);

      measurement.cpuHistory.push(estimatedCPU);

      if (measurement.cpuHistory.length > 10) {
        measurement.cpuHistory.shift();
      }

      measurement.averageCPU = measurement.cpuHistory.reduce((a, b) => a + b, 0) / measurement.cpuHistory.length;
    });
  }

  estimateCPU(plugin) {
    // Very rough estimation
    // In reality, you'd need Performance API and more sophisticated measurement

    let estimate = 0.5; // Base cost

    // Count audio nodes
    let nodeCount = 0;
    Object.keys(plugin).forEach(key => {
      if (plugin[key] && typeof plugin[key].connect === 'function') {
        nodeCount++;
      }
    });

    estimate += nodeCount * 0.1;

    return Math.min(estimate, 100);
  }

  getTotalCPU() {
    let total = 0;

    this.measurements.forEach(measurement => {
      total += measurement.averageCPU;
    });

    return total;
  }

  getPluginCPU(pluginId) {
    const measurement = this.measurements.get(pluginId);
    return measurement ? measurement.averageCPU : 0;
  }

  getStats() {
    return {
      totalCPU: this.getTotalCPU(),
      nodeCount: this.router.plugins.size,
      connectionCount: this.router.connections.length,
      baseLatency: this.context.baseLatency,
      outputLatency: this.context.outputLatency || 0,
      sampleRate: this.context.sampleRate
    };
  }

  logStats() {
    const stats = this.getStats();

    console.log('=== Performance Stats ===');
    console.log(`Total CPU: ${stats.totalCPU.toFixed(2)}%`);
    console.log(`Nodes: ${stats.nodeCount}`);
    console.log(`Connections: ${stats.connectionCount}`);
    console.log(`Base Latency: ${(stats.baseLatency * 1000).toFixed(2)}ms`);
    console.log(`Output Latency: ${(stats.outputLatency * 1000).toFixed(2)}ms`);
    console.log(`Sample Rate: ${stats.sampleRate}Hz`);
  }
}

export default PerformanceMonitor;
```

## Testing & Integration (Week 4)

### Testing Checklist

- [ ] Base Plugin: All methods work correctly
- [ ] Base Plugin: Parameter validation works
- [ ] Base Plugin: Preset save/load works
- [ ] Router: Plugins can be added/removed
- [ ] Router: Connections work (serial and parallel)
- [ ] Router: Cycle detection prevents circular routing
- [ ] Router: Topological sort is correct
- [ ] Router: Send/return buses work
- [ ] Preset Manager: Presets save/load correctly
- [ ] Preset Manager: Import/export works
- [ ] Automation: Parameters can be automated
- [ ] Automation: Playback is accurate
- [ ] Performance Monitor: CPU estimates are reasonable
- [ ] Integration: Works with all 9 other agents' plugins

### Integration Testing

Create comprehensive examples that use plugins from all categories:

1. **Full Mixing Console**:
   - Input → Channel EQ → Compressor → Send (Reverb) → EQ Eight → Limiter → Output
   - Multiple tracks in parallel
   - Send/return effects

2. **Creative Sound Design Chain**:
   - Input → Grain Delay → Spectral Time → Filter → Distortion → Echo → Output

3. **Mastering Chain**:
   - Input → EQ Eight → Multiband Dynamics → Glue Compressor → Limiter → Spectrum Analyzer → Output

## Deliverables

### Code Files
```
/core/
├── BasePlugin.js
├── Router.js
├── PresetManager.js
├── ParamAutomation.js
├── PerformanceMonitor.js
├── PluginFactory.js (create plugins by name)
└── README.md

/examples/
├── master-routing-example.html
├── full-mixing-console-example.html
├── preset-management-example.html
├── automation-demo-example.html
└── performance-monitoring-example.html
```

### Documentation

Create `/core/README.md`:
- Architecture overview
- How to create new plugins
- Routing system usage
- Preset management guide
- Automation guide
- Performance optimization tips

## Success Criteria

✅ Base plugin class provides solid foundation
✅ Router handles complex signal flow correctly
✅ Cycle detection prevents feedback loops
✅ Presets save and load reliably
✅ Automation system works accurately
✅ Performance monitoring provides useful insights
✅ All 9 agents' plugins integrate seamlessly
✅ Comprehensive examples demonstrate system
✅ Documentation is clear and complete

## Final Integration

Once all 10 agents complete their work:

1. **Create Main Entry Point**:
   ```javascript
   // index.js
   import * as Dynamics from './dynamics/index.js';
   import * as EQ from './eq/index.js';
   // ... all categories

   export {
     Dynamics,
     EQ,
     // ... etc
     Router,
     PresetManager,
     ParamAutomation
   };
   ```

2. **Build System**:
   - Set up bundler (Webpack, Rollup, or Vite)
   - Create UMD, ESM, and CJS builds
   - Minified production build

3. **Complete Documentation**:
   - API reference
   - Getting started guide
   - Advanced routing examples
   - Performance guide

4. **Testing Suite**:
   - Unit tests for each plugin
   - Integration tests for routing
   - Performance benchmarks

You are the glue that holds this entire project together. Your infrastructure will determine how well everything works! 🏗️
