import React, { useState, useCallback, useMemo, useRef } from 'react';
import styles from './DSPEditor.module.css';
import DSPGraphEditor from './DSPGraphEditor';
import {
  NODE_CATEGORIES,
  CATEGORY_COLORS,
  getCategoryForType,
  getNodeSchema,
  nextNodeId,
} from './dspNodeDefinitions';

// ── Main DSP Editor Component ──────────────────────────────────────────────

const DSPEditor = ({ dspConfig, onUpdateDsp }) => {
  const [viewMode, setViewMode] = useState('graph'); // 'graph' | 'chain'
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedParam, setSelectedParam] = useState(null);
  const [showAddMenu, setShowAddMenu] = useState(null); // null or insert index
  const [addMenuFilter, setAddMenuFilter] = useState('');
  const [dragNode, setDragNode] = useState(null);
  const [dragOver, setDragOver] = useState(null);
  const chainRef = useRef(null);

  const chain = dspConfig?.dspChain || [];
  const parameters = dspConfig?.parameters || [];
  const routing = dspConfig?.routing || { input: 'stereo', chain: [], output: 'stereo' };

  // Build param→node binding map
  const paramBindings = useMemo(() => {
    const bindings = {}; // paramId → { nodeId, nodeParam }
    for (const node of chain) {
      if (!node.params) continue;
      for (const [key, val] of Object.entries(node.params)) {
        if (typeof val === 'string' && val.startsWith('@')) {
          const paramId = val.slice(1);
          if (!bindings[paramId]) bindings[paramId] = [];
          bindings[paramId].push({ nodeId: node.id, nodeParam: key });
        }
      }
    }
    return bindings;
  }, [chain]);

  // ── Node operations ────────────────────────────────────────────────

  const updateNode = useCallback((nodeId, updates) => {
    if (!dspConfig) return;
    const newChain = chain.map(n =>
      n.id === nodeId ? { ...n, ...updates } : n
    );
    onUpdateDsp({
      ...dspConfig,
      dspChain: newChain,
      routing: { ...routing, chain: newChain.map(n => n.id) },
    });
  }, [dspConfig, chain, routing, onUpdateDsp]);

  const updateNodeParam = useCallback((nodeId, paramKey, value) => {
    const node = chain.find(n => n.id === nodeId);
    if (!node) return;
    updateNode(nodeId, {
      params: { ...node.params, [paramKey]: value },
    });
  }, [chain, updateNode]);

  const removeNode = useCallback((nodeId) => {
    if (!dspConfig) return;
    const newChain = chain.filter(n => n.id !== nodeId);
    onUpdateDsp({
      ...dspConfig,
      dspChain: newChain,
      routing: { ...routing, chain: newChain.map(n => n.id) },
    });
    if (selectedNode === nodeId) setSelectedNode(null);
  }, [dspConfig, chain, routing, selectedNode, onUpdateDsp]);

  const addNode = useCallback((type, insertIndex) => {
    if (!dspConfig) return;
    const schema = getNodeSchema(type);
    const id = nextNodeId(type);
    const params = {};
    if (schema) {
      for (const [key, def] of Object.entries(schema.params)) {
        params[key] = def.default;
      }
    }
    const newNode = { type, id, params };
    const newChain = [...chain];
    newChain.splice(insertIndex, 0, newNode);
    onUpdateDsp({
      ...dspConfig,
      dspChain: newChain,
      routing: { ...routing, chain: newChain.map(n => n.id) },
    });
    setShowAddMenu(null);
    setAddMenuFilter('');
    setSelectedNode(id);
  }, [dspConfig, chain, routing, onUpdateDsp]);

  const moveNode = useCallback((fromIndex, toIndex) => {
    if (fromIndex === toIndex || !dspConfig) return;
    const newChain = [...chain];
    const [moved] = newChain.splice(fromIndex, 1);
    newChain.splice(toIndex, 0, moved);
    onUpdateDsp({
      ...dspConfig,
      dspChain: newChain,
      routing: { ...routing, chain: newChain.map(n => n.id) },
    });
  }, [dspConfig, chain, routing, onUpdateDsp]);

  // ── Parameter operations ───────────────────────────────────────────

  const addParameter = useCallback((param) => {
    if (!dspConfig) return;
    onUpdateDsp({
      ...dspConfig,
      parameters: [...parameters, param],
    });
  }, [dspConfig, parameters, onUpdateDsp]);

  const updateParameter = useCallback((paramId, updates) => {
    if (!dspConfig) return;
    onUpdateDsp({
      ...dspConfig,
      parameters: parameters.map(p => p.id === paramId ? { ...p, ...updates } : p),
    });
  }, [dspConfig, parameters, onUpdateDsp]);

  const removeParameter = useCallback((paramId) => {
    if (!dspConfig) return;
    // Also unbind from any nodes
    const newChain = chain.map(node => {
      if (!node.params) return node;
      const newParams = { ...node.params };
      for (const [key, val] of Object.entries(newParams)) {
        if (val === `@${paramId}`) {
          const schema = getNodeSchema(node.type);
          newParams[key] = schema?.params?.[key]?.default ?? 0;
        }
      }
      return { ...node, params: newParams };
    });
    onUpdateDsp({
      ...dspConfig,
      parameters: parameters.filter(p => p.id !== paramId),
      dspChain: newChain,
    });
    if (selectedParam === paramId) setSelectedParam(null);
  }, [dspConfig, parameters, chain, selectedParam, onUpdateDsp]);

  // ── Drag and drop (chain view) ──────────────────────────────────────

  const handleDragStart = useCallback((e, index) => {
    setDragNode(index);
    e.dataTransfer.effectAllowed = 'move';
  }, []);

  const handleDragOver = useCallback((e, index) => {
    e.preventDefault();
    setDragOver(index);
  }, []);

  const handleDrop = useCallback((e, index) => {
    e.preventDefault();
    if (dragNode !== null) {
      moveNode(dragNode, index);
    }
    setDragNode(null);
    setDragOver(null);
  }, [dragNode, moveNode]);

  // ── Graph view node selection ───────────────────────────────────────

  const handleGraphSelectNode = useCallback((nodeId) => {
    setSelectedNode(nodeId);
    setSelectedParam(null);
  }, []);

  // ── No DSP config ──────────────────────────────────────────────────

  if (!dspConfig) {
    return (
      <div className={styles.editor}>
        <div className={styles.emptyState}>
          <i className="fa-solid fa-microchip" />
          <h3>No DSP Chain</h3>
          <p>Use the Backend Coder chat to generate a DSP chain, or add nodes manually.</p>
          <button
            className={styles.createBtn}
            onClick={() => {
              onUpdateDsp({
                pluginType: 'effect',
                name: 'My Plugin',
                parameters: [],
                dspChain: [],
                routing: { input: 'stereo', chain: [], output: 'stereo' },
              });
            }}
          >
            <i className="fa-solid fa-plus" /> Create Empty Chain
          </button>
        </div>
      </div>
    );
  }

  const selNode = chain.find(n => n.id === selectedNode);
  const selParam = parameters.find(p => p.id === selectedParam);

  return (
    <div className={styles.editor}>
      {/* ── Toolbar ── */}
      <div className={styles.toolbar}>
        <div className={styles.toolbarLeft}>
          {/* View toggle */}
          <div className={styles.viewToggle}>
            <button
              className={`${styles.viewToggleBtn} ${viewMode === 'graph' ? styles.viewToggleBtnActive : ''}`}
              onClick={() => setViewMode('graph')}
              title="Free-form graph view"
            >
              <i className="fa-solid fa-diagram-project" /> Graph
            </button>
            <button
              className={`${styles.viewToggleBtn} ${viewMode === 'chain' ? styles.viewToggleBtnActive : ''}`}
              onClick={() => setViewMode('chain')}
              title="Linear chain view"
            >
              <i className="fa-solid fa-bars" /> Chain
            </button>
          </div>
          <span className={styles.toolbarMeta}>
            {chain.length} node{chain.length !== 1 ? 's' : ''} &middot; {parameters.length} param{parameters.length !== 1 ? 's' : ''}
          </span>
        </div>
        <div className={styles.toolbarRight}>
          <select
            className={styles.ioSelect}
            value={routing.input}
            onChange={e => onUpdateDsp({ ...dspConfig, routing: { ...routing, input: e.target.value } })}
          >
            <option value="stereo">Stereo In</option>
            <option value="mono">Mono In</option>
          </select>
          <span className={styles.arrow}>&rarr;</span>
          <select
            className={styles.ioSelect}
            value={routing.output}
            onChange={e => onUpdateDsp({ ...dspConfig, routing: { ...routing, output: e.target.value } })}
          >
            <option value="stereo">Stereo Out</option>
            <option value="mono">Mono Out</option>
          </select>
        </div>
      </div>

      <div className={styles.mainArea}>
        {viewMode === 'graph' ? (
          /* ── Graph View ── */
          <div className={styles.graphArea}>
            <DSPGraphEditor
              dspConfig={dspConfig}
              onUpdateDsp={onUpdateDsp}
              onSelectNode={handleGraphSelectNode}
            />
          </div>
        ) : (
          /* ── Chain View (existing) ── */
          <div className={styles.chainArea} ref={chainRef}>
            {/* Input node */}
            <div className={styles.ioNode}>
              <div className={styles.ioIcon}><i className="fa-solid fa-right-to-bracket" /></div>
              <span>{routing.input}</span>
            </div>

            {chain.map((node, idx) => {
              const cat = getCategoryForType(node.type);
              const color = CATEGORY_COLORS[cat] || '#90a4ae';
              const isSelected = selectedNode === node.id;
              const isDragTarget = dragOver === idx && dragNode !== idx;
              const boundParams = Object.entries(node.params || {}).filter(
                ([, v]) => typeof v === 'string' && v.startsWith('@')
              );

              return (
                <React.Fragment key={node.id}>
                  {/* Wire + insert button */}
                  <div
                    className={`${styles.wire} ${isDragTarget ? styles.wireDragOver : ''}`}
                    onDragOver={e => handleDragOver(e, idx)}
                    onDrop={e => handleDrop(e, idx)}
                  >
                    <div className={styles.wireLine} />
                    <button
                      className={styles.insertBtn}
                      onClick={() => setShowAddMenu(showAddMenu === idx ? null : idx)}
                      title="Insert node"
                    >
                      <i className="fa-solid fa-plus" />
                    </button>
                    {showAddMenu === idx && (
                      <AddNodeMenu
                        filter={addMenuFilter}
                        onFilterChange={setAddMenuFilter}
                        onSelect={(type) => addNode(type, idx)}
                        onClose={() => { setShowAddMenu(null); setAddMenuFilter(''); }}
                      />
                    )}
                  </div>

                  {/* Node box */}
                  <div
                    className={`${styles.node} ${isSelected ? styles.nodeSelected : ''}`}
                    style={{ '--node-color': color }}
                    onClick={() => { setSelectedNode(node.id); setSelectedParam(null); }}
                    draggable
                    onDragStart={e => handleDragStart(e, idx)}
                    onDragOver={e => handleDragOver(e, idx)}
                    onDrop={e => handleDrop(e, idx)}
                  >
                    {/* Header */}
                    <div className={styles.nodeHeader} style={{ background: color }}>
                      <span className={styles.nodeType}>{node.type}</span>
                      <button
                        className={styles.nodeRemove}
                        onClick={e => { e.stopPropagation(); removeNode(node.id); }}
                        title="Remove node"
                      >
                        <i className="fa-solid fa-xmark" />
                      </button>
                    </div>

                    {/* ID */}
                    <div className={styles.nodeId}>{node.id}</div>

                    {/* Inline params */}
                    <div className={styles.nodeParams}>
                      {Object.entries(node.params || {}).map(([key, val]) => {
                        const isBound = typeof val === 'string' && val.startsWith('@');
                        const paramDef = isBound ? parameters.find(p => p.id === val.slice(1)) : null;
                        return (
                          <div key={key} className={styles.nodeParam}>
                            <span className={styles.nodeParamKey}>{key}</span>
                            {isBound ? (
                              <span
                                className={styles.nodeParamBound}
                                title={paramDef ? `${paramDef.name} (${paramDef.min}-${paramDef.max} ${paramDef.unit || ''})` : val}
                              >
                                <i className="fa-solid fa-link" /> {val.slice(1)}
                              </span>
                            ) : (
                              <span className={styles.nodeParamVal}>
                                {typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(2)) : String(val)}
                              </span>
                            )}
                          </div>
                        );
                      })}
                      {Object.keys(node.params || {}).length === 0 && (
                        <div className={styles.nodeParamEmpty}>no params</div>
                      )}
                    </div>

                    {/* Binding indicators */}
                    {boundParams.length > 0 && (
                      <div className={styles.nodeBindings}>
                        {boundParams.map(([key, val]) => (
                          <div key={key} className={styles.bindingDot} style={{ background: color }} title={`${key} → ${val}`} />
                        ))}
                      </div>
                    )}
                  </div>
                </React.Fragment>
              );
            })}

            {/* Final wire + add */}
            <div className={styles.wire}>
              <div className={styles.wireLine} />
              <button
                className={styles.insertBtn}
                onClick={() => setShowAddMenu(showAddMenu === chain.length ? null : chain.length)}
                title="Add node at end"
              >
                <i className="fa-solid fa-plus" />
              </button>
              {showAddMenu === chain.length && (
                <AddNodeMenu
                  filter={addMenuFilter}
                  onFilterChange={setAddMenuFilter}
                  onSelect={(type) => addNode(type, chain.length)}
                  onClose={() => { setShowAddMenu(null); setAddMenuFilter(''); }}
                />
              )}
            </div>

            {/* Output node */}
            <div className={styles.ioNode}>
              <div className={styles.ioIcon}><i className="fa-solid fa-right-from-bracket" /></div>
              <span>{routing.output}</span>
            </div>
          </div>
        )}

        {/* ── Detail Panel (right side) ── */}
        <div className={styles.detailPanel}>
          {/* Parameters section */}
          <div className={styles.paramSection}>
            <div className={styles.paramHeader}>
              <span><i className="fa-solid fa-sliders" /> Parameters</span>
              <button
                className={styles.addParamBtn}
                onClick={() => {
                  const id = `param_${parameters.length + 1}`;
                  addParameter({ id, name: 'New Param', min: 0, max: 1, default: 0.5, skew: 1, unit: '' });
                  setSelectedParam(id);
                  setSelectedNode(null);
                }}
                title="Add parameter"
              >
                <i className="fa-solid fa-plus" />
              </button>
            </div>
            <div className={styles.paramList}>
              {parameters.map(p => {
                const binds = paramBindings[p.id] || [];
                return (
                  <div
                    key={p.id}
                    className={`${styles.paramItem} ${selectedParam === p.id ? styles.paramItemSelected : ''}`}
                    onClick={() => { setSelectedParam(p.id); setSelectedNode(null); }}
                  >
                    <div className={styles.paramName}>{p.name || p.id}</div>
                    <div className={styles.paramMeta}>
                      {p.min}–{p.max} {p.unit || ''}
                    </div>
                    {binds.length > 0 && (
                      <div className={styles.paramBindCount}>
                        <i className="fa-solid fa-link" /> {binds.length}
                      </div>
                    )}
                  </div>
                );
              })}
              {parameters.length === 0 && (
                <div className={styles.emptyParams}>No parameters defined</div>
              )}
            </div>
          </div>

          {/* Presets section */}
          <PresetsPanel
            presets={dspConfig?.presets || []}
            parameters={parameters}
            onUpdatePresets={(presets) => onUpdateDsp({ ...dspConfig, presets })}
          />

          {/* Selected node detail */}
          {selNode && (
            <NodeDetail
              node={selNode}
              parameters={parameters}
              onUpdateParam={updateNodeParam}
              onUpdateId={(newId) => updateNode(selNode.id, { id: newId })}
            />
          )}

          {/* Selected parameter detail */}
          {selParam && (
            <ParamDetail
              param={selParam}
              bindings={paramBindings[selParam.id] || []}
              chain={chain}
              onUpdate={(updates) => updateParameter(selParam.id, updates)}
              onRemove={() => removeParameter(selParam.id)}
            />
          )}
        </div>
      </div>
    </div>
  );
};

// ── Presets Panel ─────────────────────────────────────────────────────────

const PresetsPanel = ({ presets, parameters, onUpdatePresets }) => {
  const [editingIdx, setEditingIdx] = useState(null);
  const [editName, setEditName] = useState('');

  const addPreset = () => {
    const values = Object.fromEntries(
      parameters.map(p => [p.id, p.default != null ? p.default : 0.5])
    );
    const name = `Preset ${presets.length + 1}`;
    onUpdatePresets([...presets, { name, values }]);
  };

  const removePreset = (idx) => {
    onUpdatePresets(presets.filter((_, i) => i !== idx));
    if (editingIdx === idx) setEditingIdx(null);
  };

  const updatePresetName = (idx, name) => {
    const updated = [...presets];
    updated[idx] = { ...updated[idx], name };
    onUpdatePresets(updated);
  };

  const updatePresetValue = (presetIdx, paramId, value) => {
    const updated = [...presets];
    updated[presetIdx] = {
      ...updated[presetIdx],
      values: { ...updated[presetIdx].values, [paramId]: value },
    };
    onUpdatePresets(updated);
  };

  return (
    <div className={styles.paramSection} style={{ marginTop: 8 }}>
      <div className={styles.paramHeader}>
        <span><i className="fa-solid fa-layer-group" /> Presets</span>
        <button className={styles.addParamBtn} onClick={addPreset} title="Add preset">
          <i className="fa-solid fa-plus" />
        </button>
      </div>
      <div className={styles.paramList}>
        {presets.map((preset, idx) => (
          <div key={idx} className={styles.paramItem} style={{ cursor: 'pointer' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
              {editingIdx === idx ? (
                <input
                  className={styles.addMenuSearch}
                  value={editName}
                  onChange={e => setEditName(e.target.value)}
                  onBlur={() => { updatePresetName(idx, editName); setEditingIdx(null); }}
                  onKeyDown={e => { if (e.key === 'Enter') { updatePresetName(idx, editName); setEditingIdx(null); } }}
                  autoFocus
                  style={{ flex: 1, fontSize: 12, padding: '2px 6px', margin: 0 }}
                />
              ) : (
                <div
                  className={styles.paramName}
                  onDoubleClick={() => { setEditingIdx(idx); setEditName(preset.name); }}
                  style={{ flex: 1 }}
                >
                  {preset.name}
                </div>
              )}
              <button
                onClick={(e) => { e.stopPropagation(); removePreset(idx); }}
                style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.3)', cursor: 'pointer', fontSize: 11, padding: '0 4px' }}
                title="Delete preset"
              >
                <i className="fa-solid fa-trash" />
              </button>
            </div>
            {/* Param values — always visible for quick editing */}
            <div style={{ width: '100%', marginTop: 4 }}>
              {parameters.map(p => {
                const val = preset.values?.[p.id] ?? p.default ?? 0.5;
                return (
                  <div key={p.id} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '1px 0', fontSize: 11 }}>
                    <span style={{ color: 'rgba(255,255,255,0.5)', width: 60, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {p.name || p.id}
                    </span>
                    <input
                      type="range"
                      min={p.min ?? 0}
                      max={p.max ?? 1}
                      step={0.01}
                      value={val}
                      onChange={e => updatePresetValue(idx, p.id, parseFloat(e.target.value))}
                      style={{ flex: 1, height: 12, accentColor: '#ba9cff' }}
                    />
                    <span style={{ color: 'rgba(255,255,255,0.4)', width: 32, textAlign: 'right', fontSize: 10 }}>
                      {typeof val === 'number' ? val.toFixed(2) : val}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
        {presets.length === 0 && (
          <div className={styles.emptyParams}>No presets — add one to include factory presets in your plugin</div>
        )}
      </div>
    </div>
  );
};

// ── Add Node Menu (dropdown, used by chain view) ──────────────────────────

const AddNodeMenu = ({ filter, onFilterChange, onSelect, onClose }) => {
  const menuRef = useRef(null);
  const inputRef = useRef(null);

  React.useEffect(() => {
    inputRef.current?.focus();
    const handleClick = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) onClose();
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [onClose]);

  const q = filter.toLowerCase();

  return (
    <div className={styles.addMenu} ref={menuRef}>
      <input
        ref={inputRef}
        className={styles.addMenuSearch}
        placeholder="Search nodes..."
        value={filter}
        onChange={e => onFilterChange(e.target.value)}
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

// ── Node Detail Panel ──────────────────────────────────────────────────────

const NodeDetail = ({ node, parameters, onUpdateParam, onUpdateId }) => {
  const cat = getCategoryForType(node.type);
  const color = CATEGORY_COLORS[cat];
  const schema = getNodeSchema(node.type);
  const [editingId, setEditingId] = useState(false);
  const [idVal, setIdVal] = useState(node.id);

  return (
    <div className={styles.detailSection}>
      <div className={styles.detailHeader} style={{ borderColor: color }}>
        <span className={styles.detailType} style={{ color }}>{node.type}</span>
        {editingId ? (
          <input
            className={styles.detailIdInput}
            value={idVal}
            onChange={e => setIdVal(e.target.value)}
            onBlur={() => { onUpdateId(idVal); setEditingId(false); }}
            onKeyDown={e => { if (e.key === 'Enter') { onUpdateId(idVal); setEditingId(false); } }}
            autoFocus
          />
        ) : (
          <span className={styles.detailId} onClick={() => { setEditingId(true); setIdVal(node.id); }}>
            {node.id} <i className="fa-solid fa-pen" style={{ fontSize: 9, opacity: 0.4 }} />
          </span>
        )}
      </div>

      <div className={styles.detailParams}>
        {Object.entries(node.params || {}).map(([key, val]) => {
          const isBound = typeof val === 'string' && val.startsWith('@');
          const schemaDef = schema?.params?.[key];

          return (
            <div key={key} className={styles.detailParamRow}>
              <label className={styles.detailParamLabel}>{key}</label>
              {schemaDef?.options ? (
                // Dropdown for enum params
                <select
                  className={styles.detailParamSelect}
                  value={isBound ? '' : val}
                  onChange={e => onUpdateParam(node.id, key, e.target.value)}
                >
                  {schemaDef.options.map(o => <option key={o} value={o}>{o}</option>)}
                </select>
              ) : (
                <div className={styles.detailParamInputGroup}>
                  <input
                    className={styles.detailParamInput}
                    value={isBound ? val : (typeof val === 'number' ? val : String(val))}
                    onChange={e => {
                      const v = e.target.value;
                      if (v.startsWith('@')) {
                        onUpdateParam(node.id, key, v);
                      } else {
                        const num = parseFloat(v);
                        onUpdateParam(node.id, key, isNaN(num) ? v : num);
                      }
                    }}
                  />
                  {schemaDef?.unit && <span className={styles.detailParamUnit}>{schemaDef.unit}</span>}
                </div>
              )}
              {/* Quick-bind dropdown */}
              {!schemaDef?.options && (
                <select
                  className={styles.bindSelect}
                  value={isBound ? val : ''}
                  onChange={e => {
                    if (e.target.value) onUpdateParam(node.id, key, e.target.value);
                  }}
                  title="Bind to parameter"
                >
                  <option value="">bind...</option>
                  {parameters.map(p => (
                    <option key={p.id} value={`@${p.id}`}>{p.name || p.id}</option>
                  ))}
                </select>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ── Parameter Detail Panel ─────────────────────────────────────────────────

const ParamDetail = ({ param, bindings, chain, onUpdate, onRemove }) => {
  return (
    <div className={styles.detailSection}>
      <div className={styles.detailHeader} style={{ borderColor: '#667eea' }}>
        <span className={styles.detailType} style={{ color: '#667eea' }}>Parameter</span>
        <button className={styles.removeBtn} onClick={onRemove} title="Remove parameter">
          <i className="fa-solid fa-trash" />
        </button>
      </div>

      <div className={styles.detailParams}>
        <div className={styles.detailParamRow}>
          <label className={styles.detailParamLabel}>id</label>
          <input className={styles.detailParamInput} value={param.id} onChange={e => onUpdate({ id: e.target.value })} />
        </div>
        <div className={styles.detailParamRow}>
          <label className={styles.detailParamLabel}>name</label>
          <input className={styles.detailParamInput} value={param.name} onChange={e => onUpdate({ name: e.target.value })} />
        </div>
        <div className={styles.detailParamRow}>
          <label className={styles.detailParamLabel}>min</label>
          <input className={styles.detailParamInput} type="number" value={param.min} onChange={e => onUpdate({ min: parseFloat(e.target.value) || 0 })} />
        </div>
        <div className={styles.detailParamRow}>
          <label className={styles.detailParamLabel}>max</label>
          <input className={styles.detailParamInput} type="number" value={param.max} onChange={e => onUpdate({ max: parseFloat(e.target.value) || 1 })} />
        </div>
        <div className={styles.detailParamRow}>
          <label className={styles.detailParamLabel}>default</label>
          <input className={styles.detailParamInput} type="number" value={param.default} onChange={e => onUpdate({ default: parseFloat(e.target.value) || 0 })} />
        </div>
        <div className={styles.detailParamRow}>
          <label className={styles.detailParamLabel}>skew</label>
          <input className={styles.detailParamInput} type="number" step="0.1" value={param.skew} onChange={e => onUpdate({ skew: parseFloat(e.target.value) || 1 })} />
        </div>
        <div className={styles.detailParamRow}>
          <label className={styles.detailParamLabel}>unit</label>
          <input className={styles.detailParamInput} value={param.unit || ''} onChange={e => onUpdate({ unit: e.target.value })} />
        </div>
      </div>

      {/* Bindings */}
      {bindings.length > 0 && (
        <div className={styles.bindingsSection}>
          <div className={styles.bindingsTitle}>Used by</div>
          {bindings.map((b, i) => {
            const node = chain.find(n => n.id === b.nodeId);
            const cat = node ? getCategoryForType(node.type) : 'Utility';
            return (
              <div key={i} className={styles.bindingItem}>
                <span className={styles.bindingDot} style={{ background: CATEGORY_COLORS[cat] }} />
                <span>{b.nodeId}</span>
                <span className={styles.bindingArrow}>&rarr;</span>
                <span className={styles.bindingTarget}>{b.nodeParam}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default DSPEditor;
