import { getCategoryForType, getNodeSchema } from './dspNodeDefinitions';

// ── Port definitions per node type ────────────────────────────────────────

const SYNTHESIS_SOURCES = ['oscillator', 'noise', 'wavetable', 'sample_player'];
const MOD_SOURCES = ['lfo', 'envelope_follower', 'envelope_adsr'];
const ANALYSIS_SINKS = ['peak_meter', 'rms_meter'];
const MATH_NODES = ['math_add', 'math_multiply', 'math_constant', 'math_scale', 'math_crossfade', 'math_abs', 'math_rectifier', 'math_slew'];
const ROUTING_NODES = ['splitter', 'merger', 'feedback_delay'];
const SPECTRAL_NODES = ['pitch_shift', 'spectral_filter', 'spectral_freeze'];

export function getNodePorts(nodeType) {
  const schema = getNodeSchema(nodeType);
  const ports = {
    inputs: [{ id: 'audio_in', type: 'audio', label: 'In' }],
    outputs: [{ id: 'audio_out', type: 'audio', label: 'Out' }],
    paramInputs: [],
  };

  // Synthesis sources: no audio input
  if (SYNTHESIS_SOURCES.includes(nodeType)) {
    ports.inputs = [];
  }

  // LFO: no audio input, mod output instead of audio
  if (nodeType === 'lfo') {
    ports.inputs = [];
    ports.outputs = [{ id: 'mod_out', type: 'modulation', label: 'Mod' }];
  }

  // Envelope follower / ADSR: audio in + mod output
  if (nodeType === 'envelope_follower' || nodeType === 'envelope_adsr') {
    ports.outputs = [
      { id: 'audio_out', type: 'audio', label: 'Out' },
      { id: 'mod_out', type: 'modulation', label: 'Mod' },
    ];
  }

  // Analysis sinks: no audio output
  if (ANALYSIS_SINKS.includes(nodeType)) {
    ports.outputs = [];
  }

  // Mix node: two audio inputs
  if (nodeType === 'mix') {
    ports.inputs = [
      { id: 'audio_in_dry', type: 'audio', label: 'Dry' },
      { id: 'audio_in_wet', type: 'audio', label: 'Wet' },
    ];
  }

  // Splitter: 1 audio in, multiple audio outs
  if (nodeType === 'splitter') {
    ports.outputs = [
      { id: 'audio_out_1', type: 'audio', label: 'Out 1' },
      { id: 'audio_out_2', type: 'audio', label: 'Out 2' },
      { id: 'audio_out_3', type: 'audio', label: 'Out 3' },
    ];
  }

  // Merger: multiple audio ins, 1 audio out
  if (nodeType === 'merger') {
    ports.inputs = [
      { id: 'audio_in_1', type: 'audio', label: 'In 1' },
      { id: 'audio_in_2', type: 'audio', label: 'In 2' },
      { id: 'audio_in_3', type: 'audio', label: 'In 3' },
    ];
  }

  // Crossfade: two audio inputs
  if (nodeType === 'math_crossfade') {
    ports.inputs = [
      { id: 'audio_in_a', type: 'audio', label: 'A' },
      { id: 'audio_in_b', type: 'audio', label: 'B' },
    ];
  }

  // Math constant: no audio input, produces audio output
  if (nodeType === 'math_constant') {
    ports.inputs = [];
  }

  // Sidechain: no normal audio input (receives from sidechain bus)
  if (nodeType === 'sidechain') {
    ports.inputs = [];
    ports.outputs = [{ id: 'audio_out', type: 'audio', label: 'SC Out' }];
  }

  // MIDI CC map: no audio in/out, only mod output
  if (nodeType === 'midi_cc') {
    ports.inputs = [];
    ports.outputs = [{ id: 'mod_out', type: 'modulation', label: 'CC' }];
  }

  // Parameter modulation inputs: every numeric (non-enum) param
  if (schema?.params) {
    for (const [key, def] of Object.entries(schema.params)) {
      if (def.options) continue;
      if (key === 'target' || key === 'cc_number') continue; // skip non-modulatable
      ports.paramInputs.push({ id: `param_${key}`, type: 'modulation', label: key, paramKey: key });
    }
  }

  return ports;
}

// ── Connection validation ─────────────────────────────────────────────────

export function isValidConnection(connection, nodes) {
  const { sourceHandle, targetHandle } = connection;
  if (!sourceHandle || !targetHandle) return false;
  // Prevent self-connection
  if (connection.source === connection.target) return false;
  // audio_out* → audio_in*
  if (sourceHandle.startsWith('audio_out') && targetHandle.startsWith('audio_in')) return true;
  // mod_out → param_*
  if (sourceHandle === 'mod_out' && targetHandle.startsWith('param_')) return true;
  return false;
}

export function determineEdgeType(connection) {
  if (connection.sourceHandle === 'mod_out') return 'modulation';
  return 'audio';
}

// ── Chain → Graph migration ───────────────────────────────────────────────

const SPACING_X = 250;
const START_X = 100;
const CENTER_Y = 250;

export function chainToGraph(dspConfig) {
  if (dspConfig?.graph?.nodePositions && dspConfig?.graph?.connections) {
    return dspConfig.graph;
  }

  const chain = dspConfig?.dspChain || [];
  const nodePositions = {};
  const connections = [];

  nodePositions['__input__'] = { x: START_X, y: CENTER_Y };

  chain.forEach((node, idx) => {
    nodePositions[node.id] = { x: START_X + SPACING_X * (idx + 1), y: CENTER_Y };

    const prevId = idx === 0 ? '__input__' : chain[idx - 1].id;
    const prevType = idx === 0 ? null : chain[idx - 1].type;
    const sourceHandle = (prevType && MOD_SOURCES.includes(prevType) && !SYNTHESIS_SOURCES.includes(prevType))
      ? 'audio_out' : 'audio_out';

    // Skip connection if source is pure mod source (LFO)
    if (prevType === 'lfo') {
      // LFO doesn't connect via audio chain, position it above
      nodePositions[chain[idx - 1].id] = {
        x: START_X + SPACING_X * idx,
        y: CENTER_Y - 180,
      };
    } else {
      connections.push({
        id: `e_${prevId}_${node.id}`,
        source: prevId,
        sourceHandle: 'audio_out',
        target: node.id,
        targetHandle: 'audio_in',
        type: 'audio',
      });
    }
  });

  // Connect last non-LFO node to output
  const audioChain = chain.filter(n => n.type !== 'lfo');
  const lastId = audioChain.length > 0 ? audioChain[audioChain.length - 1].id : '__input__';
  nodePositions['__output__'] = { x: START_X + SPACING_X * (chain.length + 1), y: CENTER_Y };
  connections.push({
    id: `e_${lastId}___output__`,
    source: lastId,
    sourceHandle: 'audio_out',
    target: '__output__',
    targetHandle: 'audio_in',
    type: 'audio',
  });

  // Create modulation connections from @param bindings + LFO target fields
  chain.forEach(node => {
    if (node.type === 'lfo' && node.params?.target) {
      // LFO with target param — find which node uses @target or has that param
      const targetParam = node.params.target;
      for (const other of chain) {
        if (other.id === node.id) continue;
        if (other.params && targetParam in other.params) {
          connections.push({
            id: `e_mod_${node.id}_${other.id}_${targetParam}`,
            source: node.id,
            sourceHandle: 'mod_out',
            target: other.id,
            targetHandle: `param_${targetParam}`,
            type: 'modulation',
          });
          break;
        }
      }
    }
  });

  return { nodePositions, connections };
}

// ── Graph → Chain compilation ─────────────────────────────────────────────

export function graphToChain(dspConfig, graphNodes, graphEdges) {
  const chain = dspConfig?.dspChain || [];
  const audioEdges = graphEdges.filter(e => e.type === 'audio');

  // Build adjacency list
  const adj = {};
  const inDegree = {};
  const nodeIds = new Set();

  for (const edge of audioEdges) {
    if (!adj[edge.source]) adj[edge.source] = [];
    adj[edge.source].push(edge.target);
    nodeIds.add(edge.source);
    nodeIds.add(edge.target);
    inDegree[edge.target] = (inDegree[edge.target] || 0) + 1;
    if (!(edge.source in inDegree)) inDegree[edge.source] = inDegree[edge.source] || 0;
  }

  // Topological sort (Kahn's algorithm)
  const queue = [];
  for (const id of nodeIds) {
    if ((inDegree[id] || 0) === 0) queue.push(id);
  }

  const sorted = [];
  while (queue.length > 0) {
    const curr = queue.shift();
    sorted.push(curr);
    for (const next of (adj[curr] || [])) {
      inDegree[next]--;
      if (inDegree[next] === 0) queue.push(next);
    }
  }

  // Filter to only actual DSP nodes (not __input__ / __output__)
  const orderedIds = sorted.filter(id => id !== '__input__' && id !== '__output__');

  // Build new dspChain preserving node data
  const nodeMap = {};
  for (const node of chain) {
    nodeMap[node.id] = node;
  }
  // Also include nodes from graphNodes that might not be in chain yet
  for (const gNode of graphNodes) {
    if (gNode.id !== '__input__' && gNode.id !== '__output__' && gNode.data?.nodeData) {
      if (!nodeMap[gNode.id]) {
        nodeMap[gNode.id] = gNode.data.nodeData;
      }
    }
  }

  const newChain = orderedIds
    .map(id => nodeMap[id])
    .filter(Boolean);

  // Handle modulation edges — store as metadata on target nodes
  const modEdges = graphEdges.filter(e => e.type === 'modulation');
  for (const edge of modEdges) {
    const targetNode = newChain.find(n => n.id === edge.target);
    if (targetNode && edge.targetHandle?.startsWith('param_')) {
      const paramKey = edge.targetHandle.replace('param_', '');
      if (!targetNode.modulations) targetNode.modulations = [];
      const existing = targetNode.modulations.find(
        m => m.source_node === edge.source && m.target_param === paramKey
      );
      if (!existing) {
        targetNode.modulations.push({
          target_param: paramKey,
          source_node: edge.source,
          depth: 0.5,
        });
      }
    }
  }

  return {
    dspChain: newChain,
    routing: {
      input: dspConfig?.routing?.input || 'stereo',
      chain: newChain.map(n => n.id),
      output: dspConfig?.routing?.output || 'stereo',
    },
  };
}

// ── Auto-layout (organized chain view) ────────────────────────────────────

export function autoLayoutNodes(dspConfig) {
  const chain = dspConfig?.dspChain || [];
  const positions = {};

  positions['__input__'] = { x: START_X, y: CENTER_Y };

  let audioIdx = 0;
  chain.forEach((node) => {
    if (MOD_SOURCES.includes(node.type) && node.type === 'lfo') {
      // Position LFOs above the chain
      positions[node.id] = { x: START_X + SPACING_X * (audioIdx + 1), y: CENTER_Y - 180 };
    } else {
      audioIdx++;
      positions[node.id] = { x: START_X + SPACING_X * audioIdx, y: CENTER_Y };
    }
  });

  positions['__output__'] = { x: START_X + SPACING_X * (audioIdx + 1), y: CENTER_Y };

  return positions;
}

// ── Convert dspConfig to React Flow nodes/edges ───────────────────────────

export function convertToReactFlow(dspConfig) {
  const graph = dspConfig?.graph || chainToGraph(dspConfig);
  const chain = dspConfig?.dspChain || [];
  const parameters = dspConfig?.parameters || [];
  const routing = dspConfig?.routing || { input: 'stereo', output: 'stereo' };

  const rfNodes = [];
  const rfEdges = [];

  // Input node
  const inputPos = graph.nodePositions['__input__'] || { x: START_X, y: CENTER_Y };
  rfNodes.push({
    id: '__input__',
    type: 'ioNode',
    position: inputPos,
    data: { label: routing.input, direction: 'input' },
    deletable: false,
  });

  // Output node
  const outputPos = graph.nodePositions['__output__'] || { x: START_X + SPACING_X * (chain.length + 1), y: CENTER_Y };
  rfNodes.push({
    id: '__output__',
    type: 'ioNode',
    position: outputPos,
    data: { label: routing.output, direction: 'output' },
    deletable: false,
  });

  // DSP nodes
  for (const node of chain) {
    const pos = graph.nodePositions[node.id] || { x: 300, y: 250 };
    const category = getCategoryForType(node.type);
    const ports = getNodePorts(node.type);

    rfNodes.push({
      id: node.id,
      type: 'dspNode',
      position: pos,
      data: {
        nodeData: node,
        category,
        ports,
        parameters,
      },
    });
  }

  // Edges
  for (const conn of (graph.connections || [])) {
    rfEdges.push({
      id: conn.id,
      source: conn.source,
      sourceHandle: conn.sourceHandle,
      target: conn.target,
      targetHandle: conn.targetHandle,
      type: conn.type || 'audio',
    });
  }

  return { rfNodes, rfEdges };
}
