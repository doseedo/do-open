import React from 'react';
import styles from './PluginCreator.module.css';

const CATEGORIES = [
  {
    name: 'Controls',
    items: [
      { type: 'knob',        label: 'Knob' },
      { type: 'slider',      label: 'Slider' },
      { type: 'button',      label: 'Button' },
      { type: 'dropdown',    label: 'Dropdown' },
      { type: 'xy-pad',      label: 'XY Pad' },
      { type: 'click-knob',  label: 'Precision Knob' },
    ],
  },
  {
    name: 'Display',
    items: [
      { type: 'label',             label: 'Label' },
      { type: 'led',               label: 'LED' },
      { type: 'meter',             label: 'Meter' },
      { type: 'waveform',          label: 'Waveform' },
      { type: 'spectrum-analyzer', label: 'Spectrum' },
      { type: 'oscilloscope',      label: 'Oscilloscope' },
    ],
  },
  {
    name: 'Envelopes',
    items: [
      { type: 'mseg-editor',  label: 'MSEG' },
      { type: 'adsr',         label: 'ADSR' },
    ],
  },
  {
    name: 'Routing',
    items: [
      { type: 'mod-matrix',   label: 'Mod Matrix' },
      { type: 'cable',        label: 'Cable' },
    ],
  },
  {
    name: 'Layout',
    items: [
      { type: 'panel', label: 'Panel' },
      { type: 'image', label: 'Image' },
      { type: 'tab-group', label: 'Tab Group' },
    ],
  },
];

/* Small inline SVG thumbnails for each component type */
const Thumb = ({ type }) => {
  const s = { width: 28, height: 28, flexShrink: 0 };
  switch (type) {
    case 'knob':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <circle cx="14" cy="14" r="11" fill="none" stroke="rgba(186,156,255,0.3)" strokeWidth="2"/>
          <path d="M14 5 A9 9 0 1 1 7.5 21" fill="none" stroke="#ba9cff" strokeWidth="2.5" strokeLinecap="round"/>
          <circle cx="14" cy="14" r="3" fill="rgba(186,156,255,0.6)"/>
          <line x1="14" y1="14" x2="14" y2="7" stroke="#fff" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
      );
    case 'click-knob':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <circle cx="14" cy="14" r="11" fill="none" stroke="rgba(102,126,234,0.3)" strokeWidth="2"/>
          <path d="M14 5 A9 9 0 1 1 9 22" fill="none" stroke="#667eea" strokeWidth="2.5" strokeLinecap="round"/>
          <circle cx="14" cy="14" r="2.5" fill="rgba(102,126,234,0.6)"/>
          <line x1="14" y1="14" x2="17" y2="7" stroke="#fff" strokeWidth="1.5" strokeLinecap="round"/>
          {[0,1,2,3,4,5,6,7].map(i => {
            const a = (i / 8) * Math.PI * 2 - Math.PI / 2;
            return <circle key={i} cx={14 + Math.cos(a)*12} cy={14 + Math.sin(a)*12} r="0.8" fill="rgba(102,126,234,0.4)"/>;
          })}
        </svg>
      );
    case 'slider':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <rect x="12" y="3" width="4" height="22" rx="2" fill="rgba(186,156,255,0.15)" stroke="rgba(186,156,255,0.3)" strokeWidth="1"/>
          <rect x="12" y="12" width="4" height="13" rx="2" fill="rgba(186,156,255,0.4)"/>
          <rect x="9" y="10" width="10" height="5" rx="2" fill="#ba9cff"/>
        </svg>
      );
    case 'button':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <rect x="3" y="8" width="22" height="12" rx="4" fill="rgba(102,126,234,0.2)" stroke="rgba(102,126,234,0.5)" strokeWidth="1.5"/>
          <text x="14" y="16.5" textAnchor="middle" fontSize="7" fill="#667eea" fontWeight="600">ON</text>
        </svg>
      );
    case 'dropdown':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <rect x="3" y="8" width="22" height="12" rx="3" fill="rgba(255,255,255,0.06)" stroke="rgba(255,255,255,0.2)" strokeWidth="1"/>
          <text x="8" y="16.5" fontSize="7" fill="rgba(255,255,255,0.5)">Opt</text>
          <path d="M21 12 L23 15 L19 15 Z" fill="rgba(186,156,255,0.6)"/>
        </svg>
      );
    case 'xy-pad':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <rect x="3" y="3" width="22" height="22" rx="3" fill="rgba(102,126,234,0.08)" stroke="rgba(102,126,234,0.3)" strokeWidth="1"/>
          <line x1="14" y1="3" x2="14" y2="25" stroke="rgba(255,255,255,0.08)" strokeWidth="0.5"/>
          <line x1="3" y1="14" x2="25" y2="14" stroke="rgba(255,255,255,0.08)" strokeWidth="0.5"/>
          <circle cx="18" cy="10" r="3" fill="rgba(102,126,234,0.6)" stroke="#667eea" strokeWidth="1"/>
        </svg>
      );
    case 'label':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <text x="4" y="19" fontSize="16" fontWeight="700" fill="rgba(255,255,255,0.5)" fontFamily="sans-serif">Aa</text>
        </svg>
      );
    case 'led':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <circle cx="14" cy="14" r="7" fill="rgba(34,197,94,0.3)"/>
          <circle cx="14" cy="14" r="5" fill="#22c55e"/>
          <circle cx="12" cy="12" r="1.5" fill="rgba(255,255,255,0.4)"/>
        </svg>
      );
    case 'meter':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          {[0,1,2,3,4].map(i => (
            <rect key={i} x={4 + i*4.5} y={i < 3 ? 6 : i < 4 ? 10 : 14} width="3.5" height={22 - (i < 3 ? 6 : i < 4 ? 10 : 14)} rx="1"
              fill={i < 3 ? 'rgba(34,197,94,0.6)' : i < 4 ? 'rgba(234,179,8,0.6)' : 'rgba(239,68,68,0.4)'}/>
          ))}
        </svg>
      );
    case 'waveform':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <path d="M2 14 Q5 6, 8 14 Q11 22, 14 14 Q17 6, 20 14 Q23 22, 26 14" fill="none" stroke="#ba9cff" strokeWidth="1.5"/>
        </svg>
      );
    case 'spectrum-analyzer':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          {[0,1,2,3,4,5,6].map(i => {
            const h = [8,14,18,22,16,10,6][i];
            return <rect key={i} x={2 + i*3.7} y={26 - h} width="2.8" height={h} rx="1" fill={`hsla(${250+i*10},70%,60%,0.6)`}/>;
          })}
        </svg>
      );
    case 'oscilloscope':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <rect x="2" y="4" width="24" height="20" rx="3" fill="rgba(0,0,0,0.3)" stroke="rgba(0,229,255,0.3)" strokeWidth="1"/>
          <path d="M4 14 Q7 6, 10 14 Q13 22, 16 14 Q19 6, 22 14 Q24 18, 26 14" fill="none" stroke="#00e5ff" strokeWidth="1.5"/>
        </svg>
      );
    case 'mseg-editor':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <polyline points="3,22 8,8 14,16 20,6 25,22" fill="none" stroke="#ba9cff" strokeWidth="1.5" strokeLinejoin="round"/>
          {[[3,22],[8,8],[14,16],[20,6],[25,22]].map(([x,y],i) => (
            <circle key={i} cx={x} cy={y} r="2" fill="#ba9cff"/>
          ))}
        </svg>
      );
    case 'adsr':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <polyline points="3,24 8,4 13,12 20,12 25,24" fill="none" stroke="#667eea" strokeWidth="1.5" strokeLinejoin="round"/>
          <text x="5" y="8" fontSize="4" fill="rgba(255,255,255,0.3)">A</text>
          <text x="10" y="16" fontSize="4" fill="rgba(255,255,255,0.3)">D</text>
          <text x="16" y="16" fontSize="4" fill="rgba(255,255,255,0.3)">S</text>
          <text x="22" y="20" fontSize="4" fill="rgba(255,255,255,0.3)">R</text>
        </svg>
      );
    case 'mod-matrix':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          {[0,1,2].map(r => [0,1,2].map(c => (
            <rect key={`${r}${c}`} x={4+c*8} y={4+r*8} width="6" height="6" rx="1"
              fill={(r===0&&c===1)||(r===1&&c===2)||(r===2&&c===0) ? 'rgba(186,156,255,0.5)' : 'rgba(255,255,255,0.06)'}
              stroke="rgba(255,255,255,0.1)" strokeWidth="0.5"/>
          )))}
        </svg>
      );
    case 'cable':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <circle cx="7" cy="10" r="3" fill="rgba(239,68,68,0.3)" stroke="#ef4444" strokeWidth="1"/>
          <circle cx="21" cy="18" r="3" fill="rgba(34,197,94,0.3)" stroke="#22c55e" strokeWidth="1"/>
          <path d="M10 10 C16 10, 12 18, 18 18" fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth="1.5"/>
        </svg>
      );
    case 'panel':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <rect x="3" y="3" width="22" height="22" rx="4" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.2)" strokeWidth="1" strokeDasharray="3,2"/>
        </svg>
      );
    case 'image':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <rect x="3" y="5" width="22" height="18" rx="3" fill="rgba(255,255,255,0.06)" stroke="rgba(255,255,255,0.2)" strokeWidth="1"/>
          <circle cx="10" cy="12" r="2.5" fill="rgba(234,179,8,0.5)"/>
          <path d="M5 20 L11 14 L16 18 L20 13 L25 20 Z" fill="rgba(34,197,94,0.3)"/>
        </svg>
      );
    case 'tab-group':
      return (
        <svg viewBox="0 0 28 28" style={s}>
          <rect x="3" y="8" width="22" height="17" rx="2" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.15)" strokeWidth="1"/>
          <rect x="3" y="4" width="8" height="5" rx="1.5" fill="rgba(186,156,255,0.3)" stroke="rgba(186,156,255,0.4)" strokeWidth="0.5"/>
          <rect x="12" y="4" width="8" height="5" rx="1.5" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5"/>
        </svg>
      );
    default:
      return <i className="fa-solid fa-puzzle-piece" style={{ fontSize: 12, opacity: 0.4 }} />;
  }
};

const ComponentPalette = ({ onAddComponent }) => (
  <div className={styles.palette}>
    <div className={styles.paletteCatRow}>
      {CATEGORIES.map(cat => (
        <div key={cat.name} className={styles.paletteCatGroup}>
          <span className={styles.paletteCatLabel}>{cat.name}</span>
          <div className={styles.paletteCatItems}>
            {cat.items.map(item => (
              <button
                key={item.type}
                className={styles.paletteBtn}
                onClick={() => onAddComponent(item.type)}
                title={`Add ${item.label}`}
              >
                <Thumb type={item.type} />
                <span>{item.label}</span>
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  </div>
);

export default ComponentPalette;
