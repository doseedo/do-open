import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import {
  ReactFlow,
  Background,
  MiniMap,
  Controls,
  useNodesState,
  useEdgesState,
  addEdge,
  useReactFlow,
  ReactFlowProvider,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import DSPNodeComponent from './DSPNodeComponent';
import DSPIONode from './DSPIONode';
import { DSPAudioEdge, DSPModulationEdge } from './DSPEdgeComponent';
import {
  NODE_CATEGORIES,
  CATEGORY_COLORS,
  getNodeSchema,
  getCategoryForType,
  nextNodeId,
} from './dspNodeDefinitions';
import {
  convertToReactFlow,
  chainToGraph,
  graphToChain,
  isValidConnection,
  determineEdgeType,
  getNodePorts,
  autoLayoutNodes,
} from './dspGraphUtils';
import styles from './DSPGraphEditor.module.css';

const nodeTypes = {
  dspNode: DSPNodeComponent,
  ioNode: DSPIONode,
};

const edgeTypes = {
  audio: DSPAudioEdge,
  modulation: DSPModulationEdge,
};

// ── Add Node Menu ─────────────────────────────────────────────────────────

const AddNodeMenu = ({ position, onSelect, onClose }) => {
  const [filter, setFilter] = useState('');
  const menuRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    inputRef.current?.focus();
    const handleClick = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) onClose();
    };
    const handleKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('mousedown', handleClick);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handleClick);
      document.removeEventListener('keydown', handleKey);
    };
  }, [onClose]);

  const q = filter.toLowerCase();

  return (
    <div
      ref={menuRef}
      className={styles.addMenuOverlay}
      style={{ left: position.x, top: position.y }}
    >
      <input
        ref={inputRef}
        className={styles.addMenuSearch}
        placeholder="Search nodes..."
        value={filter}
        onChange={e => setFilter(e.target.value)}
      />
      <div className={styles.addMenuList}>
        {Object.entries(NODE_CATEGORIES).map(([cat, nodes]) => {
          const filtered = nodes.filter(n =>
            !q || n.label.toLowerCase().includes(q) || n.type.includes(q) || cat.toLowerCase().includes(q)
          );
          if (filtered.length === 0) return null;
          return (
            <div key={cat}>
              <div className={styles.addMenuCat} style={{ color: CATEGORY_COLORS[cat] }}>{cat}</div>
              {filtered.map(n => (
                <button
                  key={n.type}
                  className={styles.addMenuItem}
                  onClick={() => onSelect(n.type)}
                >
                  <span className={styles.addMenuDot} style={{ background: CATEGORY_COLORS[cat] }} />
                  {n.label}
                  <span className={styles.addMenuType}>{n.type}</span>
                </button>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ── Node Context Menu ─────────────────────────────────────────────────────

const NodeContextMenu = ({ position, nodeId, onClose, onDelete, onDuplicate, onDisconnect }) => {
  const menuRef = useRef(null);

  useEffect(() => {
    const handleClick = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) onClose();
    };
    const handleKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('mousedown', handleClick);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handleClick);
      document.removeEventListener('keydown', handleKey);
    };
  }, [onClose]);

  return (
    <div
      ref={menuRef}
      className={styles.nodeContextMenu}
      style={{ left: position.x, top: position.y }}
    >
      <button className={styles.contextMenuItem} onClick={() => { onDuplicate(nodeId); onClose(); }}>
        <i className="fa-solid fa-copy" /> Duplicate
      </button>
      <button className={styles.contextMenuItem} onClick={() => { onDisconnect(nodeId); onClose(); }}>
        <i className="fa-solid fa-link-slash" /> Disconnect All
      </button>
      <div className={styles.contextMenuDivider} />
      <button className={`${styles.contextMenuItem} ${styles.contextMenuItemDanger}`} onClick={() => { onDelete(nodeId); onClose(); }}>
        <i className="fa-solid fa-trash" /> Delete Node
      </button>
    </div>
  );
};

// ── Main Graph Editor (inner, needs ReactFlowProvider) ────────────────────

const DSPGraphEditorInner = ({ dspConfig, onUpdateDsp, onSelectNode }) => {
  const reactFlowInstance = useReactFlow();
  const [addMenu, setAddMenu] = useState(null); // { x, y, canvasX, canvasY }
  const [contextMenu, setContextMenu] = useState(null); // { x, y, nodeId }
  const containerRef = useRef(null);

  // Convert dspConfig to React Flow nodes/edges
  const { rfNodes: initialNodes, rfEdges: initialEdges } = useMemo(
    () => convertToReactFlow(dspConfig),
    // Only recompute when dspConfig reference changes from parent
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [dspConfig]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Sync React Flow state when dspConfig changes from parent (chat, chain view, etc.)
  const lastDspRef = useRef(dspConfig);
  useEffect(() => {
    if (dspConfig !== lastDspRef.current) {
      lastDspRef.current = dspConfig;
      const { rfNodes, rfEdges } = convertToReactFlow(dspConfig);
      setNodes(rfNodes);
      setEdges(rfEdges);
    }
  }, [dspConfig, setNodes, setEdges]);

  // ── Sync graph state back to dspConfig ──────────────────────────────

  const syncToParent = useCallback((currentNodes, currentEdges) => {
    if (!dspConfig) return;

    // Build nodePositions
    const nodePositions = {};
    for (const node of currentNodes) {
      nodePositions[node.id] = { x: node.position.x, y: node.position.y };
    }

    // Build connections
    const connections = currentEdges.map(e => ({
      id: e.id,
      source: e.source,
      sourceHandle: e.sourceHandle,
      target: e.target,
      targetHandle: e.targetHandle,
      type: e.type || 'audio',
    }));

    // Derive chain from graph
    const { dspChain, routing } = graphToChain(dspConfig, currentNodes, currentEdges);

    // Raw graph for parallel routing support
    const dspGraph = {
      nodes: currentNodes.map(n => ({
        id: n.id, type: n.data?.type || n.type,
        params: n.data?.params, position: n.position,
      })),
      edges: currentEdges.map(e => ({
        id: e.id, source: e.source, target: e.target,
        sourceHandle: e.sourceHandle, targetHandle: e.targetHandle,
        type: e.type || 'audio',
      })),
    };

    const updated = {
      ...dspConfig,
      dspChain,
      routing,
      dspGraph,
      graph: { nodePositions, connections },
    };

    lastDspRef.current = updated;
    onUpdateDsp(updated);
  }, [dspConfig, onUpdateDsp]);

  // ── Event handlers ──────────────────────────────────────────────────

  const onNodeDragStop = useCallback((event, node) => {
    // Use the latest nodes/edges from state
    setNodes(currentNodes => {
      setEdges(currentEdges => {
        syncToParent(currentNodes, currentEdges);
        return currentEdges;
      });
      return currentNodes;
    });
  }, [syncToParent, setNodes, setEdges]);

  const onConnect = useCallback((connection) => {
    if (!isValidConnection(connection, nodes)) return;
    const edgeType = determineEdgeType(connection);
    const newEdge = {
      ...connection,
      id: `e_${connection.source}_${connection.sourceHandle}_${connection.target}_${connection.targetHandle}`,
      type: edgeType,
    };
    setEdges(prev => {
      const next = addEdge(newEdge, prev);
      // Sync after edge added
      setTimeout(() => syncToParent(nodes, next), 0);
      return next;
    });
  }, [nodes, setEdges, syncToParent]);

  const onEdgesDelete = useCallback((deletedEdges) => {
    setTimeout(() => {
      setNodes(currentNodes => {
        setEdges(currentEdges => {
          syncToParent(currentNodes, currentEdges);
          return currentEdges;
        });
        return currentNodes;
      });
    }, 0);
  }, [syncToParent, setNodes, setEdges]);

  const handleValidateConnection = useCallback((connection) => {
    return isValidConnection(connection, nodes);
  }, [nodes]);

  // ── Add node ────────────────────────────────────────────────────────

  const addNode = useCallback((type) => {
    if (!dspConfig || !addMenu) return;
    const schema = getNodeSchema(type);
    const id = nextNodeId(type);
    const params = {};
    if (schema) {
      for (const [key, def] of Object.entries(schema.params)) {
        params[key] = def.default;
      }
    }

    const newNodeData = { type, id, params };
    const category = getCategoryForType(type);
    const ports = getNodePorts(type);

    const newRfNode = {
      id,
      type: 'dspNode',
      position: { x: addMenu.canvasX, y: addMenu.canvasY },
      data: {
        nodeData: newNodeData,
        category,
        ports,
        parameters: dspConfig.parameters || [],
      },
    };

    setNodes(prev => {
      const next = [...prev, newRfNode];
      // Add to dspChain
      const newChain = [...(dspConfig.dspChain || []), newNodeData];
      const nodePositions = {};
      for (const n of next) {
        nodePositions[n.id] = { x: n.position.x, y: n.position.y };
      }
      setEdges(currentEdges => {
        const connections = currentEdges.map(e => ({
          id: e.id,
          source: e.source,
          sourceHandle: e.sourceHandle,
          target: e.target,
          targetHandle: e.targetHandle,
          type: e.type || 'audio',
        }));
        const updated = {
          ...dspConfig,
          dspChain: newChain,
          routing: { ...dspConfig.routing, chain: newChain.map(n => n.id) },
          graph: { nodePositions, connections },
        };
        lastDspRef.current = updated;
        onUpdateDsp(updated);
        return currentEdges;
      });
      return next;
    });

    setAddMenu(null);
    onSelectNode(id);
  }, [dspConfig, addMenu, onUpdateDsp, onSelectNode, setNodes, setEdges]);

  // ── Delete node ─────────────────────────────────────────────────────

  const deleteNode = useCallback((nodeId) => {
    if (nodeId === '__input__' || nodeId === '__output__') return;

    setNodes(prev => {
      const next = prev.filter(n => n.id !== nodeId);
      setEdges(prevEdges => {
        const nextEdges = prevEdges.filter(e => e.source !== nodeId && e.target !== nodeId);
        syncToParent(next, nextEdges);
        return nextEdges;
      });
      return next;
    });

    if (onSelectNode) onSelectNode(null);
  }, [setNodes, setEdges, syncToParent, onSelectNode]);

  // ── Duplicate node ──────────────────────────────────────────────────

  const duplicateNode = useCallback((nodeId) => {
    const sourceNode = nodes.find(n => n.id === nodeId);
    if (!sourceNode || !sourceNode.data?.nodeData) return;

    const origData = sourceNode.data.nodeData;
    const newId = nextNodeId(origData.type);
    const newNodeData = { ...origData, id: newId, params: { ...origData.params } };
    const category = getCategoryForType(origData.type);
    const ports = getNodePorts(origData.type);

    const newRfNode = {
      id: newId,
      type: 'dspNode',
      position: { x: sourceNode.position.x + 30, y: sourceNode.position.y + 30 },
      data: {
        nodeData: newNodeData,
        category,
        ports,
        parameters: dspConfig?.parameters || [],
      },
    };

    setNodes(prev => {
      const next = [...prev, newRfNode];
      setEdges(currentEdges => {
        syncToParent(next, currentEdges);
        return currentEdges;
      });
      return next;
    });

    onSelectNode(newId);
  }, [nodes, dspConfig, setNodes, setEdges, syncToParent, onSelectNode]);

  // ── Disconnect all edges from a node ────────────────────────────────

  const disconnectNode = useCallback((nodeId) => {
    setEdges(prev => {
      const next = prev.filter(e => e.source !== nodeId && e.target !== nodeId);
      setNodes(currentNodes => {
        syncToParent(currentNodes, next);
        return currentNodes;
      });
      return next;
    });
  }, [setEdges, setNodes, syncToParent]);

  // ── Context menus ───────────────────────────────────────────────────

  const onNodeContextMenu = useCallback((event, node) => {
    event.preventDefault();
    if (node.id === '__input__' || node.id === '__output__') return;
    setContextMenu({
      x: event.clientX,
      y: event.clientY,
      nodeId: node.id,
    });
    setAddMenu(null);
  }, []);

  const onPaneContextMenu = useCallback((event) => {
    event.preventDefault();
    const bounds = containerRef.current?.getBoundingClientRect();
    if (!bounds) return;

    // Convert screen coords to canvas coords
    const canvasPos = reactFlowInstance.screenToFlowPosition({
      x: event.clientX,
      y: event.clientY,
    });

    setAddMenu({
      x: event.clientX - bounds.left,
      y: event.clientY - bounds.top,
      canvasX: canvasPos.x,
      canvasY: canvasPos.y,
    });
    setContextMenu(null);
  }, [reactFlowInstance]);

  const onPaneClick = useCallback(() => {
    setAddMenu(null);
    setContextMenu(null);
    onSelectNode(null);
  }, [onSelectNode]);

  const onNodeClick = useCallback((event, node) => {
    setAddMenu(null);
    setContextMenu(null);
    if (node.id !== '__input__' && node.id !== '__output__') {
      onSelectNode(node.id);
    }
  }, [onSelectNode]);

  // Handle Delete/Backspace on selected nodes
  const onNodesDelete = useCallback((deletedNodes) => {
    const toDelete = deletedNodes.filter(n => n.id !== '__input__' && n.id !== '__output__');
    if (toDelete.length === 0) return;
    setTimeout(() => {
      setNodes(currentNodes => {
        setEdges(currentEdges => {
          syncToParent(currentNodes, currentEdges);
          return currentEdges;
        });
        return currentNodes;
      });
    }, 0);
    onSelectNode(null);
  }, [syncToParent, setNodes, setEdges, onSelectNode]);

  // Auto-layout (organize)
  const handleAutoLayout = useCallback(() => {
    if (!dspConfig) return;
    const positions = autoLayoutNodes(dspConfig);
    setNodes(prev => prev.map(n => ({
      ...n,
      position: positions[n.id] || n.position,
    })));
    setTimeout(() => {
      setNodes(currentNodes => {
        setEdges(currentEdges => {
          syncToParent(currentNodes, currentEdges);
          return currentEdges;
        });
        return currentNodes;
      });
      reactFlowInstance.fitView({ padding: 0.2, duration: 300 });
    }, 50);
  }, [dspConfig, setNodes, setEdges, syncToParent, reactFlowInstance]);

  // ── Keyboard shortcuts ──────────────────────────────────────────────
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleKeyDown = (e) => {
      // Only handle when the graph container or its children are focused
      if (!container.contains(document.activeElement) && document.activeElement !== container) return;
      // Don't intercept if user is typing in an input/textarea
      const tag = document.activeElement?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      // Ctrl+A / Cmd+A: select all nodes
      if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
        e.preventDefault();
        const allNodes = reactFlowInstance.getNodes();
        const selected = allNodes.map(n => ({ ...n, selected: true }));
        setNodes(selected);
      }

      // Delete / Backspace: delete selected nodes
      if (e.key === 'Delete' || e.key === 'Backspace') {
        const selectedNodes = reactFlowInstance.getNodes().filter(n => n.selected);
        const toDelete = selectedNodes.filter(n => n.id !== '__input__' && n.id !== '__output__');
        if (toDelete.length > 0) {
          for (const node of toDelete) {
            deleteNode(node.id);
          }
        }
      }
    };

    container.addEventListener('keydown', handleKeyDown);
    return () => container.removeEventListener('keydown', handleKeyDown);
  }, [reactFlowInstance, setNodes, deleteNode]);

  // Check if audio chain is connected from input to output
  const chainConnected = useMemo(() => {
    const audioEdges = edges.filter(e => e.type === 'audio');
    // Simple check: is __input__ reachable to __output__?
    const visited = new Set();
    const queue = ['__input__'];
    while (queue.length > 0) {
      const curr = queue.shift();
      if (curr === '__output__') return true;
      if (visited.has(curr)) continue;
      visited.add(curr);
      for (const edge of audioEdges) {
        if (edge.source === curr) queue.push(edge.target);
      }
    }
    return false;
  }, [edges]);

  const minimapNodeColor = useCallback((node) => {
    if (node.type === 'ioNode') return 'rgba(255,255,255,0.2)';
    return CATEGORY_COLORS[node.data?.category] || '#90a4ae';
  }, []);

  return (
    <div ref={containerRef} className={styles.graphContainer}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onEdgesDelete={onEdgesDelete}
        onNodesDelete={onNodesDelete}
        onNodeDragStop={onNodeDragStop}
        onNodeClick={onNodeClick}
        onNodeContextMenu={onNodeContextMenu}
        onPaneContextMenu={onPaneContextMenu}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        isValidConnection={handleValidateConnection}
        fitView
        snapToGrid
        snapGrid={[20, 20]}
        minZoom={0.2}
        maxZoom={3}
        deleteKeyCode={['Backspace', 'Delete']}
        multiSelectionKeyCode="Shift"
        selectionOnDrag
        panOnScroll
        zoomOnPinch
        zoomOnScroll
        attributionPosition="bottom-left"
      >
        <Background variant="dots" gap={20} size={1} color="rgba(255,255,255,0.04)" />
        <MiniMap
          nodeColor={minimapNodeColor}
          maskColor="rgba(0,0,0,0.6)"
          style={{ background: '#111122', borderRadius: 8, border: '1px solid rgba(255,255,255,0.08)' }}
          pannable
          zoomable
        />
        <Controls
          showInteractive={false}
          style={{ background: 'transparent', border: 'none' }}
        />
      </ReactFlow>

      {/* Add node menu */}
      {addMenu && (
        <AddNodeMenu
          position={{ x: addMenu.x, y: addMenu.y }}
          onSelect={addNode}
          onClose={() => setAddMenu(null)}
        />
      )}

      {/* Node context menu */}
      {contextMenu && (
        <NodeContextMenu
          position={{ x: contextMenu.x - (containerRef.current?.getBoundingClientRect()?.left || 0), y: contextMenu.y - (containerRef.current?.getBoundingClientRect()?.top || 0) }}
          nodeId={contextMenu.nodeId}
          onClose={() => setContextMenu(null)}
          onDelete={deleteNode}
          onDuplicate={duplicateNode}
          onDisconnect={disconnectNode}
        />
      )}

      {/* Chain not connected warning */}
      {!chainConnected && nodes.length > 2 && (
        <div className={styles.chainWarning}>
          <i className="fa-solid fa-triangle-exclamation" />
          Audio chain not connected — connect nodes from Input to Output
        </div>
      )}
    </div>
  );
};

// ── Wrapped with ReactFlowProvider ────────────────────────────────────────

const DSPGraphEditor = (props) => (
  <ReactFlowProvider>
    <DSPGraphEditorInner {...props} />
  </ReactFlowProvider>
);

export default DSPGraphEditor;
