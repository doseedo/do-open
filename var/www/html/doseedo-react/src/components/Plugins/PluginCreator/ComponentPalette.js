import React from 'react';
import styles from './PluginCreator.module.css';

const CATEGORIES = [
  {
    name: 'Controls',
    items: [
      { type: 'knob',     icon: 'fa-solid fa-circle-dot',    label: 'Knob' },
      { type: 'slider',   icon: 'fa-solid fa-sliders',       label: 'Slider' },
      { type: 'button',   icon: 'fa-solid fa-square',        label: 'Button' },
      { type: 'dropdown', icon: 'fa-solid fa-caret-down',    label: 'Dropdown' },
      { type: 'xy-pad',   icon: 'fa-solid fa-up-down-left-right', label: 'XY Pad' },
    ],
  },
  {
    name: 'Display',
    items: [
      { type: 'label',    icon: 'fa-solid fa-font',          label: 'Label' },
      { type: 'led',      icon: 'fa-solid fa-circle',        label: 'LED' },
      { type: 'meter',    icon: 'fa-solid fa-signal',        label: 'Meter' },
      { type: 'waveform', icon: 'fa-solid fa-wave-square',   label: 'Waveform' },
    ],
  },
  {
    name: 'Layout',
    items: [
      { type: 'panel',    icon: 'fa-solid fa-vector-square', label: 'Panel' },
      { type: 'image',    icon: 'fa-solid fa-image',         label: 'Image' },
    ],
  },
];

const ComponentPalette = ({ onAddComponent }) => (
  <div className={styles.palette}>
    {CATEGORIES.map(cat => (
      <React.Fragment key={cat.name}>
        <span className={styles.paletteCatLabel}>{cat.name}</span>
        {cat.items.map(item => (
          <button
            key={item.type}
            className={styles.paletteBtn}
            onClick={() => onAddComponent(item.type)}
            title={`Add ${item.label}`}
          >
            <i className={item.icon} />
            <span>{item.label}</span>
          </button>
        ))}
      </React.Fragment>
    ))}
  </div>
);

export default ComponentPalette;
