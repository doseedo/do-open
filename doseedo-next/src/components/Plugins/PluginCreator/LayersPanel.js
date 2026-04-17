import React, { useState, useRef, useEffect, useCallback } from 'react';
import styles from './PluginCreator.module.css';

const TYPE_ICONS = {
  knob: 'fa-solid fa-circle-dot',
  slider: 'fa-solid fa-sliders',
  button: 'fa-solid fa-square',
  label: 'fa-solid fa-font',
  led: 'fa-solid fa-circle',
  dropdown: 'fa-solid fa-caret-down',
  image: 'fa-solid fa-image',
  panel: 'fa-solid fa-vector-square',
  meter: 'fa-solid fa-signal',
  waveform: 'fa-solid fa-wave-square',
  'xy-pad': 'fa-solid fa-up-down-left-right',
  'click-knob': 'fa-solid fa-circle-dot',
  'spectrum-analyzer': 'fa-solid fa-chart-bar',
  'oscilloscope': 'fa-solid fa-wave-square',
  'mseg-editor': 'fa-solid fa-bezier-curve',
  'adsr': 'fa-solid fa-chart-line',
  'mod-matrix': 'fa-solid fa-table-cells',
  cable: 'fa-solid fa-plug',
  'tab-group': 'fa-solid fa-folder',
};

const LayersPanel = ({ components, selectedIds, onSelect, onUpdateComponent, onClose }) => {
  const [renamingId, setRenamingId] = useState(null);
  const [renameValue, setRenameValue] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [dragId, setDragId] = useState(null);
  const [dragOverId, setDragOverId] = useState(null);
  const renameRef = useRef(null);
  const listRef = useRef(null);
  const itemRefs = useRef({});

  // Sort by zIndex descending (top layer first)
  const sorted = [...components].sort((a, b) => (b.zIndex || 0) - (a.zIndex || 0));

  // Filter by search term (match label or type)
  const filtered = searchTerm.trim()
    ? sorted.filter(comp => {
        const term = searchTerm.toLowerCase();
        const label = (comp.label || '').toLowerCase();
        const type = (comp.type || '').toLowerCase();
        return label.includes(term) || type.includes(term);
      })
    : sorted;

  const moveUp = (comp) => {
    onUpdateComponent(comp.id, { zIndex: (comp.zIndex || 1) + 1 });
  };
  const moveDown = (comp) => {
    onUpdateComponent(comp.id, { zIndex: Math.max(0, (comp.zIndex || 1) - 1) });
  };

  const startRename = (comp) => {
    setRenamingId(comp.id);
    setRenameValue(comp.label || comp.type);
  };

  const confirmRename = (comp) => {
    if (renameValue.trim()) {
      onUpdateComponent(comp.id, { label: renameValue.trim() });
    }
    setRenamingId(null);
  };

  const cancelRename = () => {
    setRenamingId(null);
  };

  // Auto-focus rename input
  useEffect(() => {
    if (renamingId && renameRef.current) {
      renameRef.current.focus();
      renameRef.current.select();
    }
  }, [renamingId]);

  // --- Drag-to-reorder handlers ---
  const handleDragStart = useCallback((e, comp) => {
    setDragId(comp.id);
    e.dataTransfer.effectAllowed = 'move';
    // Required for Firefox
    e.dataTransfer.setData('text/plain', comp.id);
  }, []);

  const handleDragOver = useCallback((e, comp) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    if (comp.id !== dragId) {
      setDragOverId(comp.id);
    }
  }, [dragId]);

  const handleDragLeave = useCallback(() => {
    setDragOverId(null);
  }, []);

  const handleDrop = useCallback((e, targetComp) => {
    e.preventDefault();
    setDragOverId(null);

    if (!dragId || dragId === targetComp.id) {
      setDragId(null);
      return;
    }

    // Find positions in the sorted (visual) list
    const dragIndex = sorted.findIndex(c => c.id === dragId);
    const targetIndex = sorted.findIndex(c => c.id === targetComp.id);

    if (dragIndex === -1 || targetIndex === -1) {
      setDragId(null);
      return;
    }

    // Build the new visual order by moving the dragged item to the target position
    const reordered = [...sorted];
    const [removed] = reordered.splice(dragIndex, 1);
    reordered.splice(targetIndex, 0, removed);

    // Reassign zIndex values: top item gets highest zIndex
    const maxZ = reordered.length;
    reordered.forEach((comp, i) => {
      const newZ = maxZ - i;
      if ((comp.zIndex || 0) !== newZ) {
        onUpdateComponent(comp.id, { zIndex: newZ });
      }
    });

    setDragId(null);
  }, [dragId, sorted, onUpdateComponent]);

  const handleDragEnd = useCallback(() => {
    setDragId(null);
    setDragOverId(null);
  }, []);

  // --- Keyboard navigation ---
  const handleKeyDown = useCallback((e, comp, index) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      const nextIndex = index + 1;
      if (nextIndex < filtered.length) {
        const nextComp = filtered[nextIndex];
        if (itemRefs.current[nextComp.id]) {
          itemRefs.current[nextComp.id].focus();
        }
      }
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      const prevIndex = index - 1;
      if (prevIndex >= 0) {
        const prevComp = filtered[prevIndex];
        if (itemRefs.current[prevComp.id]) {
          itemRefs.current[prevComp.id].focus();
        }
      }
    } else if (e.key === 'Enter') {
      e.preventDefault();
      onSelect(comp.id);
    }
  }, [filtered, onSelect]);

  return (
    <div className={styles.layersPanel}>
      <div className={styles.layersHeader}>
        <span>Layers</span>
        <button className={styles.layersCloseBtn} onClick={onClose}>
          <i className="fa-solid fa-xmark" />
        </button>
      </div>
      {/* Search/filter input */}
      <div style={{ padding: '6px 6px 0' }}>
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Filter layers..."
          style={{
            width: '100%',
            padding: '5px 8px',
            background: 'rgba(255, 255, 255, 0.06)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: 6,
            color: '#fff',
            fontSize: 11,
            fontFamily: 'inherit',
            outline: 'none',
            boxSizing: 'border-box',
          }}
          onFocus={(e) => { e.target.style.borderColor = 'rgba(186, 156, 255, 0.4)'; }}
          onBlur={(e) => { e.target.style.borderColor = 'rgba(255, 255, 255, 0.1)'; }}
        />
      </div>
      <div
        className={styles.layersList}
        ref={listRef}
        role="listbox"
        aria-label="Layers"
      >
        {filtered.map((comp, index) => (
          <div
            key={comp.id}
            ref={(el) => { itemRefs.current[comp.id] = el; }}
            className={`${styles.layerItem} ${selectedIds.includes(comp.id) ? styles.layerItemSelected : ''}`}
            onClick={() => onSelect(comp.id)}
            onKeyDown={(e) => handleKeyDown(e, comp, index)}
            style={{
              opacity: comp.hidden ? 0.4 : 1,
              borderTop: dragOverId === comp.id ? '2px solid rgba(186, 156, 255, 0.7)' : '2px solid transparent',
              transition: 'background 0.15s, border-top 0.1s',
            }}
            tabIndex={0}
            role="option"
            aria-selected={selectedIds.includes(comp.id)}
            draggable
            onDragStart={(e) => handleDragStart(e, comp)}
            onDragOver={(e) => handleDragOver(e, comp)}
            onDragLeave={handleDragLeave}
            onDrop={(e) => handleDrop(e, comp)}
            onDragEnd={handleDragEnd}
          >
            <i className={TYPE_ICONS[comp.type] || 'fa-solid fa-cube'} style={{ fontSize: 11, opacity: 0.5, width: 16 }} />
            {renamingId === comp.id ? (
              <input
                ref={renameRef}
                className={styles.layerRenameInput}
                value={renameValue}
                onChange={(e) => setRenameValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') confirmRename(comp);
                  if (e.key === 'Escape') cancelRename();
                }}
                onBlur={() => confirmRename(comp)}
                onClick={(e) => e.stopPropagation()}
              />
            ) : (
              <span
                className={styles.layerLabel}
                onDoubleClick={(e) => { e.stopPropagation(); startRename(comp); }}
                title="Double-click to rename"
              >
                {comp.label || comp.type}
              </span>
            )}
            <div className={styles.layerActions}>
              <button
                onClick={(e) => { e.stopPropagation(); onUpdateComponent(comp.id, { hidden: !comp.hidden }); }}
                title={comp.hidden ? 'Show' : 'Hide'}
                style={{ opacity: comp.hidden ? 0.3 : 0.6 }}
              >
                <i className={`fa-solid ${comp.hidden ? 'fa-eye-slash' : 'fa-eye'}`} />
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); onUpdateComponent(comp.id, { locked: !comp.locked }); }}
                title={comp.locked ? 'Unlock' : 'Lock'}
                style={{ opacity: comp.locked ? 1 : 0.4, color: comp.locked ? '#f5a623' : undefined }}
              >
                <i className={`fa-solid ${comp.locked ? 'fa-lock' : 'fa-lock-open'}`} />
              </button>
              <button onClick={(e) => { e.stopPropagation(); moveUp(comp); }} title="Move up">
                <i className="fa-solid fa-chevron-up" />
              </button>
              <button onClick={(e) => { e.stopPropagation(); moveDown(comp); }} title="Move down">
                <i className="fa-solid fa-chevron-down" />
              </button>
            </div>
          </div>
        ))}
        {components.length === 0 && (
          <div className={styles.layersEmpty}>No components</div>
        )}
        {components.length > 0 && filtered.length === 0 && searchTerm.trim() && (
          <div className={styles.layersEmpty}>No matching layers</div>
        )}
      </div>
    </div>
  );
};

export default LayersPanel;
