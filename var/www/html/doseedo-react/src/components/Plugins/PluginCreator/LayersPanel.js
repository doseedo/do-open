import React from 'react';
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
};

const LayersPanel = ({ components, selectedId, onSelect, onUpdateComponent, onClose }) => {
  // Sort by zIndex descending (top layer first)
  const sorted = [...components].sort((a, b) => (b.zIndex || 0) - (a.zIndex || 0));

  const moveUp = (comp) => {
    onUpdateComponent(comp.id, { zIndex: (comp.zIndex || 1) + 1 });
  };
  const moveDown = (comp) => {
    onUpdateComponent(comp.id, { zIndex: Math.max(0, (comp.zIndex || 1) - 1) });
  };

  return (
    <div className={styles.layersPanel}>
      <div className={styles.layersHeader}>
        <span>Layers</span>
        <button className={styles.layersCloseBtn} onClick={onClose}>
          <i className="fa-solid fa-xmark" />
        </button>
      </div>
      <div className={styles.layersList}>
        {sorted.map(comp => (
          <div
            key={comp.id}
            className={`${styles.layerItem} ${comp.id === selectedId ? styles.layerItemSelected : ''}`}
            onClick={() => onSelect(comp.id)}
          >
            <i className={TYPE_ICONS[comp.type] || 'fa-solid fa-cube'} style={{ fontSize: 11, opacity: 0.5, width: 16 }} />
            <span className={styles.layerLabel}>{comp.label || comp.type}</span>
            <span className={styles.layerZ}>z{comp.zIndex || 0}</span>
            <div className={styles.layerActions}>
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
      </div>
    </div>
  );
};

export default LayersPanel;
