import React, { useRef, useCallback, useState, useEffect } from 'react';
import Draggable from 'react-draggable';
import { SVGKnobRenderer, SVGSliderRenderer, SVGButtonRenderer, SVGStaticRenderer } from './SVGRenderer';
import { SpriteKnobRenderer, SpriteSliderRenderer, SpriteButtonRenderer } from './SpriteRenderer';
import { generateKnobSVG, generateSliderSVG, generateButtonSVG, MOOG_KNOB_IMAGES } from './svgComponentLibrary';
import { resolveFluxComponent } from './fluxComponentCache';
import XYPad from './XYPad';
import ClickToTypeKnob from './ClickToTypeKnob';
import SpectrumAnalyzer from "./SpectrumAnalyzer";
import MSEGEditor from "./MSEGEditor";
import ModMatrix from "./ModMatrix";
import ModMatrixWithSliders from "./ModMatrixWithSliders";
import Oscilloscope from "./Oscilloscope";
import ADSRDisplay from "./ADSRDisplay";
import EQCurveDisplay from "./EQCurveDisplay";
import { getModulatableTargets, createModConnection } from "./modulationUtils";
import { usePluginContext } from "./PluginContext";
import styles from './PluginCreator.module.css';

/* ---- Interactive knob (test mode) ---- */

const InteractiveKnob = ({ color, size, value, onChange }) => {
  const ref = useRef(null);
  const dragging = useRef(false);
  const startY = useRef(0);
  const startVal = useRef(0);

  const r = size / 2 - 4;
  const cx = size / 2;
  const cy = size / 2;
  // Map value (0-1) to angle (-225 to +45 degrees, 270° range)
  const angle = -225 + value * 270;
  const rad = (angle * Math.PI) / 180;
  const pointerX = cx + (r - 8) * Math.cos(rad);
  const pointerY = cy + (r - 8) * Math.sin(rad);

  useEffect(() => {
    const onMove = (e) => {
      if (!dragging.current) return;
      const dy = startY.current - (e.clientY || e.touches?.[0]?.clientY || 0);
      const newVal = Math.max(0, Math.min(1, startVal.current + dy / 120));
      onChange(newVal);
    };
    const onUp = () => { dragging.current = false; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    window.addEventListener('touchmove', onMove);
    window.addEventListener('touchend', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('touchmove', onMove);
      window.removeEventListener('touchend', onUp);
    };
  }, [onChange]);

  const onDown = (e) => {
    e.stopPropagation();
    dragging.current = true;
    startY.current = e.clientY || e.touches?.[0]?.clientY || 0;
    startVal.current = value;
  };

  return (
    <svg ref={ref} width={size} height={size} viewBox={`0 0 ${size} ${size}`}
         onMouseDown={onDown} onTouchStart={onDown} style={{ cursor: 'ns-resize' }}>
      <circle cx={cx} cy={cy} r={r} fill="#222" stroke={color} strokeWidth="2" />
      <circle cx={cx} cy={cy} r={r - 6} fill="#333" />
      {/* Tick marks */}
      {[...Array(11)].map((_, i) => {
        const a = -225 + (i * 270 / 10);
        const aRad = (a * Math.PI) / 180;
        const x1 = cx + (r + 1) * Math.cos(aRad);
        const y1 = cy + (r + 1) * Math.sin(aRad);
        const x2 = cx + (r - 3) * Math.cos(aRad);
        const y2 = cy + (r - 3) * Math.sin(aRad);
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />;
      })}
      {/* Value arc */}
      {value > 0.01 && (() => {
        const startA = (-225 * Math.PI) / 180;
        const endA = rad;
        const sx = cx + (r - 3) * Math.cos(startA);
        const sy = cy + (r - 3) * Math.sin(startA);
        const ex = cx + (r - 3) * Math.cos(endA);
        const ey = cy + (r - 3) * Math.sin(endA);
        const largeArc = value * 270 > 180 ? 1 : 0;
        return <path d={`M${sx},${sy} A${r - 3},${r - 3} 0 ${largeArc},1 ${ex},${ey}`}
                     fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" opacity="0.7" />;
      })()}
      {/* Pointer */}
      <line x1={cx} y1={cy} x2={pointerX} y2={pointerY} stroke={color} strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  );
};

/* ---- Interactive slider (test mode) ---- */

const InteractiveSlider = ({ color, width, height, value, onChange }) => {
  const dragging = useRef(false);
  const startY = useRef(0);
  const startVal = useRef(0);

  useEffect(() => {
    const onMove = (e) => {
      if (!dragging.current) return;
      const dy = startY.current - (e.clientY || e.touches?.[0]?.clientY || 0);
      const newVal = Math.max(0, Math.min(1, startVal.current + dy / (height - 20)));
      onChange(newVal);
    };
    const onUp = () => { dragging.current = false; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    window.addEventListener('touchmove', onMove);
    window.addEventListener('touchend', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('touchmove', onMove);
      window.removeEventListener('touchend', onUp);
    };
  }, [onChange, height]);

  const onDown = (e) => {
    e.stopPropagation();
    dragging.current = true;
    startY.current = e.clientY || e.touches?.[0]?.clientY || 0;
    startVal.current = value;
  };

  const thumbPos = (1 - value) * (height - 20) + 5;

  return (
    <div style={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'ns-resize' }}
         onMouseDown={onDown} onTouchStart={onDown}>
      <div style={{ width: 4, height: height - 10, background: 'rgba(255,255,255,0.12)', borderRadius: 2, position: 'relative' }}>
        {/* Filled portion */}
        <div style={{
          position: 'absolute', bottom: 0, left: 0, width: '100%',
          height: `${value * 100}%`, background: color, borderRadius: 2, opacity: 0.4,
        }} />
        {/* Thumb */}
        <div style={{
          position: 'absolute', top: thumbPos, left: '50%', transform: 'translateX(-50%)',
          width: 16, height: 8, background: color, borderRadius: 3,
          boxShadow: `0 0 6px ${color}`,
        }} />
      </div>
    </div>
  );
};

/* ---- Interactive XY Pad (test mode) ---- */

const InteractiveXYPad = ({ color, width, height, value, onChange }) => {
  const ref = useRef(null);
  const dragging = useRef(false);

  useEffect(() => {
    const onMove = (e) => {
      if (!dragging.current || !ref.current) return;
      const rect = ref.current.getBoundingClientRect();
      const clientX = e.clientX || e.touches?.[0]?.clientX || 0;
      const clientY = e.clientY || e.touches?.[0]?.clientY || 0;
      const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      const y = Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));
      onChange(x * 0.5 + y * 0.5); // combine to single 0-1 value
    };
    const onUp = () => { dragging.current = false; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    window.addEventListener('touchmove', onMove);
    window.addEventListener('touchend', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('touchmove', onMove);
      window.removeEventListener('touchend', onUp);
    };
  }, [onChange]);

  const onDown = (e) => {
    e.stopPropagation();
    dragging.current = true;
  };

  // Derive x/y position from value for visual
  const dotX = value * width;
  const dotY = (1 - value) * height;
  const gridLines = 4;

  return (
    <svg ref={ref} width={width} height={height} style={{ display: 'block', cursor: 'crosshair' }}
         onMouseDown={onDown} onTouchStart={onDown}>
      <rect width={width} height={height} fill="rgba(0,0,0,0.3)" rx="4" stroke={color} strokeWidth="1" strokeOpacity="0.3" />
      {[...Array(gridLines - 1)].map((_, i) => {
        const gx = ((i + 1) / gridLines) * width;
        const gy = ((i + 1) / gridLines) * height;
        return (
          <g key={i}>
            <line x1={gx} y1={0} x2={gx} y2={height} stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
            <line x1={0} y1={gy} x2={width} y2={gy} stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
          </g>
        );
      })}
      <line x1={dotX} y1={0} x2={dotX} y2={height} stroke={color} strokeWidth="0.5" strokeOpacity="0.4" />
      <line x1={0} y1={dotY} x2={width} y2={dotY} stroke={color} strokeWidth="0.5" strokeOpacity="0.4" />
      <circle cx={dotX} cy={dotY} r={5} fill={color} opacity="0.9" />
      <circle cx={dotX} cy={dotY} r={8} fill="none" stroke={color} strokeWidth="1" opacity="0.4" />
    </svg>
  );
};

/* ---- Static visual sub-components (edit mode) ---- */

const KnobVisual = ({ color, size, knobStyle }) => {
  const r = size / 2 - 4;
  const cx = size / 2;
  const cy = size / 2;
  const gradId = `kg-${size}-${(color || '').replace('#', '')}`;

  if (knobStyle === 'metallic') {
    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <radialGradient id={`${gradId}-m`} cx="35%" cy="35%">
            <stop offset="0%" stopColor="#fff" stopOpacity="0.7" />
            <stop offset="45%" stopColor="#bbb" />
            <stop offset="100%" stopColor="#444" />
          </radialGradient>
        </defs>
        <circle cx={cx} cy={cy} r={r} fill={`url(#${gradId}-m)`} />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(0,0,0,0.3)" strokeWidth="1" />
        {/* Center cap */}
        <circle cx={cx} cy={cy} r={r * 0.3} fill="#555" stroke="rgba(255,255,255,0.15)" strokeWidth="0.5" />
        {/* Indicator */}
        <line x1={cx} y1={cy} x2={cx} y2={cy - r + 6} stroke={color} strokeWidth="2.5" strokeLinecap="round" />
        {/* Tick marks */}
        {[...Array(11)].map((_, i) => {
          const angle = -225 + (i * 270 / 10);
          const aRad = (angle * Math.PI) / 180;
          const x1 = cx + (r + 2) * Math.cos(aRad);
          const y1 = cy + (r + 2) * Math.sin(aRad);
          const x2 = cx + (r - 2) * Math.cos(aRad);
          const y2 = cy + (r - 2) * Math.sin(aRad);
          return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(255,255,255,0.25)" strokeWidth="1" />;
        })}
      </svg>
    );
  }

  if (knobStyle === 'vintage') {
    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={cx} cy={cy} r={r} fill="#3d2b1f" stroke="#5c4033" strokeWidth="2" />
        <circle cx={cx} cy={cy} r={r - 5} fill="#4a3728" />
        {/* Cream pointer */}
        <line x1={cx} y1={cy} x2={cx} y2={cy - r + 8} stroke="#f5e6c8" strokeWidth="3" strokeLinecap="round" />
        {/* Cream tick marks */}
        {[...Array(11)].map((_, i) => {
          const angle = -225 + (i * 270 / 10);
          const aRad = (angle * Math.PI) / 180;
          const x1 = cx + (r + 1) * Math.cos(aRad);
          const y1 = cy + (r + 1) * Math.sin(aRad);
          const x2 = cx + (r - 3) * Math.cos(aRad);
          const y2 = cy + (r - 3) * Math.sin(aRad);
          return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(245,230,200,0.4)" strokeWidth="1" />;
        })}
      </svg>
    );
  }

  if (knobStyle === 'minimal') {
    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Thin ring */}
        <circle cx={cx} cy={cy} r={r - 2} fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="1.5" />
        {/* Dot indicator at default position (top) */}
        <circle cx={cx} cy={cy - r + 6} r={3} fill={color} />
      </svg>
    );
  }

  if (knobStyle === 'led-ring') {
    const segments = 12;
    const arcStart = -225;
    const arcRange = 270;
    const litCount = 8; // preview: ~65% lit
    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={cx} cy={cy} r={r * 0.55} fill="#1a1a1a" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
        {/* LED segments */}
        {[...Array(segments)].map((_, i) => {
          const angle = arcStart + (i * arcRange / (segments - 1));
          const aRad = (angle * Math.PI) / 180;
          const lx = cx + (r - 2) * Math.cos(aRad);
          const ly = cy + (r - 2) * Math.sin(aRad);
          const lit = i < litCount;
          return <circle key={i} cx={lx} cy={ly} r={2.5} fill={lit ? color : 'rgba(255,255,255,0.08)'} opacity={lit ? 0.9 : 0.4} />;
        })}
        {/* Center indicator */}
        <line x1={cx} y1={cy} x2={cx} y2={cy - r * 0.45} stroke="rgba(255,255,255,0.6)" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    );
  }

  // Default style
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={cx} cy={cy} r={r} fill="#222" stroke={color} strokeWidth="2" />
      <circle cx={cx} cy={cy} r={r - 6} fill="#333" />
      <line x1={cx} y1={cy} x2={cx} y2={cy - r + 8} stroke={color} strokeWidth="2" strokeLinecap="round" />
      {[...Array(11)].map((_, i) => {
        const angle = -225 + (i * 270 / 10);
        const aRad = (angle * Math.PI) / 180;
        const x1 = cx + (r + 1) * Math.cos(aRad);
        const y1 = cy + (r + 1) * Math.sin(aRad);
        const x2 = cx + (r - 3) * Math.cos(aRad);
        const y2 = cy + (r - 3) * Math.sin(aRad);
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />;
      })}
    </svg>
  );
};

const SliderVisual = ({ color, width, height, knobStyle }) => {
  const trackW = Math.max(4, width * 0.15);
  const thumbPos = '30%';

  if (knobStyle === 'metallic') {
    return (
      <div style={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{
          width: trackW + 2, height: height - 10, borderRadius: trackW,
          background: 'linear-gradient(90deg, #2a2a2a, #444, #2a2a2a)',
          border: '1px solid rgba(255,255,255,0.15)',
          position: 'relative', boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.6)',
        }}>
          <div style={{
            position: 'absolute', top: thumbPos, left: '50%', transform: 'translateX(-50%)',
            width: trackW + 14, height: trackW + 6,
            background: `linear-gradient(180deg, #eee 0%, #999 40%, #666 100%)`,
            borderRadius: 3, border: '1px solid rgba(0,0,0,0.4)',
            boxShadow: '0 1px 4px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.3)',
          }}>
            <div style={{ position: 'absolute', top: '50%', left: '15%', right: '15%', height: 1, background: 'rgba(0,0,0,0.25)' }} />
          </div>
        </div>
      </div>
    );
  }

  if (knobStyle === 'vintage') {
    return (
      <div style={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{
          width: trackW + 2, height: height - 10, borderRadius: 2,
          background: '#3d2b1a', border: '1px solid rgba(255,200,100,0.2)',
          position: 'relative', boxShadow: 'inset 0 1px 4px rgba(0,0,0,0.5)',
        }}>
          <div style={{
            position: 'absolute', top: thumbPos, left: '50%', transform: 'translateX(-50%)',
            width: trackW + 14, height: trackW + 8,
            background: 'linear-gradient(180deg, #f5e6c8 0%, #c8a87a 100%)',
            borderRadius: 2, border: '1px solid rgba(100,60,20,0.5)',
            boxShadow: '0 1px 3px rgba(0,0,0,0.4)',
          }}>
            <div style={{ position: 'absolute', top: '50%', left: '20%', right: '20%', height: 1, background: 'rgba(100,60,20,0.3)' }} />
          </div>
        </div>
      </div>
    );
  }

  if (knobStyle === 'led-ring') {
    const segments = 8;
    const segH = (height - 20) / segments;
    const litCount = Math.round(0.65 * segments);
    return (
      <div style={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ display: 'flex', flexDirection: 'column-reverse', gap: 2 }}>
          {Array.from({ length: segments }).map((_, i) => (
            <div key={i} style={{
              width: trackW + 6, height: Math.max(3, segH - 2), borderRadius: 1,
              background: i < litCount ? color : 'rgba(255,255,255,0.08)',
              boxShadow: i < litCount ? `0 0 4px ${color}` : 'none',
              transition: 'all 0.15s',
            }} />
          ))}
        </div>
      </div>
    );
  }

  if (knobStyle === 'minimal') {
    return (
      <div style={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{
          width: 2, height: height - 10,
          background: 'rgba(255,255,255,0.15)', borderRadius: 1, position: 'relative',
        }}>
          <div style={{
            position: 'absolute', top: thumbPos, left: '50%', transform: 'translateX(-50%)',
            width: 10, height: 10, borderRadius: '50%',
            background: 'transparent', border: `2px solid ${color}`,
          }} />
        </div>
      </div>
    );
  }

  // Default
  return (
    <div style={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ width: 4, height: height - 10, background: 'rgba(255,255,255,0.12)', borderRadius: 2, position: 'relative' }}>
        <div style={{
          position: 'absolute', top: thumbPos, left: '50%', transform: 'translateX(-50%)',
          width: 16, height: 8, background: color, borderRadius: 3,
        }} />
      </div>
    </div>
  );
};

const ButtonVisual = ({ color, label, width, height, fontSize, borderRadius, pressed, knobStyle }) => {
  const br = borderRadius || 4;
  const fs = fontSize || 11;

  if (knobStyle === 'metallic') {
    return (
      <div style={{
        width, height, borderRadius: br,
        background: pressed
          ? 'linear-gradient(180deg, #888 0%, #aaa 50%, #888 100%)'
          : 'linear-gradient(180deg, #ddd 0%, #aaa 40%, #777 100%)',
        border: '1px solid rgba(0,0,0,0.4)',
        boxShadow: pressed
          ? 'inset 0 2px 4px rgba(0,0,0,0.4)'
          : '0 2px 4px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.3)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: fs, color: '#1a1a1a', fontWeight: 700, letterSpacing: 0.5,
        transform: pressed ? 'scale(0.97)' : undefined,
        transition: 'transform 0.1s, box-shadow 0.1s',
      }}>
        {label}
      </div>
    );
  }

  if (knobStyle === 'vintage') {
    return (
      <div style={{
        width, height, borderRadius: br,
        background: pressed ? '#5a3d20' : 'linear-gradient(180deg, #6b4a2a 0%, #4a3018 100%)',
        border: '1px solid rgba(255,200,100,0.25)',
        boxShadow: pressed
          ? 'inset 0 2px 4px rgba(0,0,0,0.5)'
          : '0 2px 4px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,200,100,0.15)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: fs, color: '#f5e6c8', fontWeight: 600,
        transform: pressed ? 'scale(0.97)' : undefined,
        transition: 'transform 0.1s',
      }}>
        {label}
      </div>
    );
  }

  if (knobStyle === 'led-ring') {
    return (
      <div style={{
        width, height, borderRadius: br,
        background: pressed ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.03)',
        border: `1px solid ${pressed ? color : 'rgba(255,255,255,0.15)'}`,
        boxShadow: pressed ? `0 0 8px ${color}, inset 0 0 8px ${color}33` : 'none',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: fs, color: pressed ? color : 'rgba(255,255,255,0.7)', fontWeight: 600,
        transform: pressed ? 'scale(0.97)' : undefined,
        transition: 'all 0.15s',
      }}>
        {label}
      </div>
    );
  }

  if (knobStyle === 'minimal') {
    return (
      <div style={{
        width, height, borderRadius: br,
        background: pressed ? 'rgba(255,255,255,0.08)' : 'transparent',
        border: `1.5px solid ${pressed ? color : 'rgba(255,255,255,0.2)'}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: fs, color: pressed ? color : 'rgba(255,255,255,0.6)', fontWeight: 500,
        transform: pressed ? 'scale(0.97)' : undefined,
        transition: 'all 0.15s',
      }}>
        {label}
      </div>
    );
  }

  // Default
  return (
    <div style={{
      width, height, background: pressed ? `color-mix(in srgb, ${color} 70%, white)` : color,
      borderRadius: br,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: fs, color: '#fff', fontWeight: 600,
      transform: pressed ? 'scale(0.95)' : undefined,
      transition: 'transform 0.1s, background 0.1s',
    }}>
      {label}
    </div>
  );
};

const LabelVisual = ({ color, label, fontSize, letterSpacing }) => (
  <div style={{ color, fontSize: fontSize || 13, fontWeight: 500, whiteSpace: 'nowrap', userSelect: 'none', letterSpacing: letterSpacing || 'normal' }}>
    {label}
  </div>
);

const LEDVisual = ({ color, size, lit }) => (
  <div style={{
    width: size, height: size, borderRadius: '50%',
    background: lit ? color : `color-mix(in srgb, ${color} 30%, #111)`,
    boxShadow: lit ? `0 0 8px ${color}` : 'none',
    transition: 'all 0.2s',
  }} />
);

/* ---- Dropdown options inference ---- */

const DROPDOWN_OPTIONS = {
  'wave type': ['Sine', 'Saw', 'Square', 'Triangle', 'Noise'],
  'unison': ['1', '2', '4', '8', '16'],
  'filter type': ['Low Pass', 'High Pass', 'Band Pass', 'Notch'],
  'slope': ['12 dB', '24 dB', '48 dB'],
  'mode': ['Poly', 'Mono', 'Legato'],
  'lfo shape': ['Sine', 'Saw', 'Square', 'Triangle', 'S&H'],
  'env mode': ['ADSR', 'AHD', 'Multi-Stage'],
};

const getDropdownOptions = (label) => {
  const key = (label || '').toLowerCase().trim();
  for (const [pattern, options] of Object.entries(DROPDOWN_OPTIONS)) {
    if (key === pattern || key.includes(pattern)) return options;
  }
  return ['Option 1', 'Option 2', 'Option 3', 'Option 4'];
};

/* ---- Interactive dropdown (test mode) ---- */

const InteractiveDropdown = ({ color, label, width, fontSize, value, onChange, options }) => {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef(null);
  const menuRef = useRef(null);
  const [menuPos, setMenuPos] = useState({ top: 0, left: 0, width: 0 });

  const selectedIndex = Math.min(options.length - 1, Math.max(0, Math.round((value ?? 0) * (options.length - 1))));
  const selectedLabel = options[selectedIndex] || options[0];

  const toggleOpen = useCallback((e) => {
    e.stopPropagation();
    if (!open && triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      const spaceBelow = window.innerHeight - rect.bottom;
      const menuH = options.length * 30 + 2;
      const openUp = spaceBelow < menuH && rect.top > menuH;
      setMenuPos({
        top: openUp ? rect.top - menuH - 2 : rect.bottom + 2,
        left: rect.left,
        width: rect.width,
      });
    }
    setOpen(o => !o);
  }, [open, options.length]);

  useEffect(() => {
    if (!open) return;
    const close = (e) => {
      if (menuRef.current?.contains(e.target)) return;
      if (triggerRef.current?.contains(e.target)) return;
      setOpen(false);
    };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [open]);

  return (
    <>
      <div ref={triggerRef} onClick={toggleOpen} style={{
        width, height: 28, background: '#222',
        border: `1px solid ${open ? color : 'rgba(255,255,255,0.2)'}`,
        borderRadius: 4,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 8px', fontSize: fontSize || 11, color: 'rgba(255,255,255,0.8)',
        cursor: 'pointer', userSelect: 'none',
      }}>
        <span>{selectedLabel}</span>
        <i className={`fa-solid fa-chevron-${open ? 'up' : 'down'}`} style={{ fontSize: 9, opacity: 0.5 }} />
      </div>
      {open && (
        <div ref={menuRef} onMouseDown={e => e.stopPropagation()} style={{
          position: 'fixed', top: menuPos.top, left: menuPos.left, width: menuPos.width,
          zIndex: 10000,
          background: '#1a1a1a', border: `1px solid ${color}`, borderRadius: 4,
          overflow: 'hidden', boxShadow: '0 4px 16px rgba(0,0,0,0.6)',
        }}>
          {options.map((opt, i) => (
            <div key={i} onClick={(e) => {
              e.stopPropagation();
              onChange(options.length > 1 ? i / (options.length - 1) : 0);
              setOpen(false);
            }} style={{
              padding: '6px 10px', fontSize: fontSize || 11,
              color: i === selectedIndex ? color : 'rgba(255,255,255,0.7)',
              background: i === selectedIndex ? 'rgba(255,255,255,0.08)' : 'transparent',
              cursor: 'pointer',
            }}
            onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.12)'; }}
            onMouseLeave={e => { e.currentTarget.style.background = i === selectedIndex ? 'rgba(255,255,255,0.08)' : 'transparent'; }}
            >
              {opt}
            </div>
          ))}
        </div>
      )}
    </>
  );
};

/* ---- Static dropdown (edit mode) ---- */

const DropdownVisual = ({ color, label, width, fontSize }) => (
  <div style={{
    width, height: 28, background: '#222',
    border: `1px solid ${color}`, borderRadius: 4,
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '0 8px', fontSize: fontSize || 11, color: 'rgba(255,255,255,0.7)',
  }}>
    <span>{label}</span>
    <i className="fa-solid fa-chevron-down" style={{ fontSize: 9, opacity: 0.5 }} />
  </div>
);

const ImageVisual = ({ width, height, image, borderRadius }) => (
  <div style={{
    width, height, borderRadius: borderRadius || 0,
    background: image ? `url(${image}) center/cover no-repeat` : 'rgba(255,255,255,0.06)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    border: image ? 'none' : '1px dashed rgba(255,255,255,0.2)',
    overflow: 'hidden',
  }}>
    {!image && <i className="fa-solid fa-image" style={{ fontSize: 20, color: 'rgba(255,255,255,0.2)' }} />}
  </div>
);

const PanelVisual = ({ width, height, borderColor, bgColor, borderRadius, bgGradient, backdropBlur, boxShadow }) => (
  <div style={{
    width, height,
    border: `1px solid ${borderColor || 'rgba(255,255,255,0.1)'}`,
    background: bgGradient || bgColor || 'rgba(255,255,255,0.03)',
    borderRadius: borderRadius || 6,
    backdropFilter: backdropBlur ? `blur(${backdropBlur}px)` : undefined,
    WebkitBackdropFilter: backdropBlur ? `blur(${backdropBlur}px)` : undefined,
    boxShadow: boxShadow || undefined,
  }} />
);

const VUNeedleMeter = ({ width, height, level, bodyColor, indicatorColor, accentColor, label }) => {
  const w = width, h = height;
  const val = level ?? 0.35;
  const cx = w / 2, cy = h * 0.75;
  const r = Math.min(w * 0.40, h * 0.58);
  const needleAngle = -135 + val * 90;
  const faceColor = bodyColor || '#f5f0e4';
  const needleColor = indicatorColor || '#1a1a1a';
  const trimColor = accentColor || '#8b6914';
  const labelText = label || 'GAIN REDUCTION dB';
  const fontSize = Math.max(7, Math.min(11, w * 0.035));
  const tickFontSize = Math.max(7, r * 0.13);

  // GR scale: 0, 5, 10, 15, 20 dB
  const grLabels = ['0', '5', '10', '15', '20'];
  const majorMarks = grLabels.map((lbl, i) => {
    const frac = i / (grLabels.length - 1);
    const a = (-135 + frac * 90) * Math.PI / 180;
    return { lbl, a,
      x1: cx + (r + 1) * Math.cos(a), y1: cy + (r + 1) * Math.sin(a),
      x2: cx + (r - 6) * Math.cos(a), y2: cy + (r - 6) * Math.sin(a),
      tx: cx + (r + 14) * Math.cos(a), ty: cy + (r + 14) * Math.sin(a),
    };
  });
  // Minor ticks
  const minorMarks = Array.from({ length: 21 }, (_, i) => {
    if (i % 5 === 0) return null;
    const frac = i / 20;
    const a = (-135 + frac * 90) * Math.PI / 180;
    return {
      x1: cx + r * Math.cos(a), y1: cy + r * Math.sin(a),
      x2: cx + (r - 3) * Math.cos(a), y2: cy + (r - 3) * Math.sin(a),
    };
  }).filter(Boolean);

  // Red zone arc
  const redStartA = (-135 + 72) * Math.PI / 180;
  const redEndA = (-135 + 90) * Math.PI / 180;
  const rs = { x: cx + (r - 1) * Math.cos(redStartA), y: cy + (r - 1) * Math.sin(redStartA) };
  const re = { x: cx + (r - 1) * Math.cos(redEndA), y: cy + (r - 1) * Math.sin(redEndA) };

  // Needle endpoint
  const na = needleAngle * Math.PI / 180;
  const nx = cx + (r - 2) * Math.cos(na), ny = cy + (r - 2) * Math.sin(na);

  return (
    <div style={{ width, height, position: 'relative' }}>
      <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} xmlns="http://www.w3.org/2000/svg">
        {/* Outer housing - dark surround */}
        <rect x="1" y="1" width={w - 2} height={h - 2} rx="6" fill="#2a2520" stroke="#1a1510" strokeWidth="1.5"/>
        <rect x="3" y="3" width={w - 6} height={h - 6} rx="5" fill="#3a3530"/>
        {/* Screw holes */}
        <circle cx={14} cy={14} r={3} fill="#222" stroke="#444" strokeWidth="0.5"/>
        <circle cx={w - 14} cy={14} r={3} fill="#222" stroke="#444" strokeWidth="0.5"/>
        <circle cx={14} cy={h - 14} r={3} fill="#222" stroke="#444" strokeWidth="0.5"/>
        <circle cx={w - 14} cy={h - 14} r={3} fill="#222" stroke="#444" strokeWidth="0.5"/>
        {/* Inner cream face */}
        <rect x={w * 0.1} y={h * 0.08} width={w * 0.8} height={h * 0.72} rx="3" fill={faceColor}
          stroke={trimColor} strokeWidth="0.6"/>
        {/* Subtle inner shadow on face */}
        <rect x={w * 0.1 + 1} y={h * 0.08 + 1} width={w * 0.8 - 2} height={h * 0.72 - 2} rx="2"
          fill="none" stroke="rgba(0,0,0,0.06)" strokeWidth="1"/>
        {/* Major tick marks + labels */}
        {majorMarks.map((m, i) => (
          <g key={`major-${i}`}>
            <line x1={m.x1} y1={m.y1} x2={m.x2} y2={m.y2} stroke="#333" strokeWidth="1.2"/>
            <text x={m.tx} y={m.ty} textAnchor="middle" dominantBaseline="middle"
              fontSize={tickFontSize} fill="#333" fontFamily="serif">{m.lbl}</text>
          </g>
        ))}
        {/* Minor ticks */}
        {minorMarks.map((m, i) => (
          <line key={`minor-${i}`} x1={m.x1} y1={m.y1} x2={m.x2} y2={m.y2} stroke="#555" strokeWidth="0.5"/>
        ))}
        {/* Red zone arc */}
        <path d={`M ${rs.x.toFixed(1)} ${rs.y.toFixed(1)} A ${r - 1} ${r - 1} 0 0 1 ${re.x.toFixed(1)} ${re.y.toFixed(1)}`}
          fill="none" stroke="#cc3333" strokeWidth="2.5" opacity="0.7"/>
        {/* Needle shadow */}
        <line x1={cx + 0.5} y1={cy + 1.5} x2={nx + 0.5} y2={ny + 1.5}
          stroke="rgba(0,0,0,0.12)" strokeWidth="2" strokeLinecap="round"/>
        {/* Needle */}
        <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={needleColor} strokeWidth="1.3" strokeLinecap="round"/>
        {/* Pivot point */}
        <circle cx={cx} cy={cy} r={Math.max(4, r * 0.07)} fill="#222"/>
        <circle cx={cx} cy={cy} r={Math.max(2, r * 0.035)} fill="#555"/>
        <circle cx={cx - 1} cy={cy - 1} r={Math.max(1, r * 0.015)} fill="rgba(255,255,255,0.2)"/>
        {/* Label */}
        <text x={cx} y={cy + Math.max(16, r * 0.25)} textAnchor="middle" fontSize={fontSize}
          fill="#333" fontFamily="serif" fontWeight="600" letterSpacing="0.05em">
          {labelText}
        </text>
      </svg>
    </div>
  );
};

const MeterVisual = ({ color, width, height, level, svgStyle, bodyColor, indicatorColor, accentColor, label }) => {
  if (svgStyle === 'vu-needle') {
    return <VUNeedleMeter width={width} height={height} level={level} bodyColor={bodyColor} indicatorColor={indicatorColor} accentColor={accentColor} label={label} />;
  }
  const segments = 12;
  const segH = (height - 6) / segments;
  const litCount = Math.round((level ?? 0.65) * segments);
  return (
    <div style={{
      width, height, background: '#111', borderRadius: 3,
      padding: 3, display: 'flex', flexDirection: 'column-reverse', gap: 1,
      border: '1px solid rgba(255,255,255,0.1)',
    }}>
      {[...Array(segments)].map((_, i) => {
        const lit = i < litCount;
        let segColor = color;
        const ratio = i / segments;
        if (ratio > 0.85) segColor = '#ff4444';
        else if (ratio > 0.7) segColor = '#ffaa00';
        return (
          <div key={i} style={{
            flex: 1, minHeight: segH - 1, borderRadius: 1,
            background: lit ? segColor : 'rgba(255,255,255,0.06)',
            opacity: lit ? 0.9 : 0.3,
          }} />
        );
      })}
    </div>
  );
};

// ── Compute waveform points (shared across all visual styles) ──
const computeWavePoints = (width, height, waveType, phase, warp, attack, decay, sustain, release) => {
  const points = [];
  const steps = 100;
  const type = (waveType || 'sine').toLowerCase();
  const amp = height * 0.35;
  const mid = height / 2;
  const cycles = 3;
  const ph = phase ?? 0;
  const wp = warp ?? 0;

  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const x = t * width;
    let y;
    switch (type) {
      case 'saw': {
        let p = (t * cycles + ph) % 1;
        if (wp > 0) p = Math.pow(p, 1 + wp * 2);
        y = mid - (p * 2 - 1) * amp;
        break;
      }
      case 'square': {
        const p = (t * cycles + ph) % 1;
        const duty = 0.5 - wp * 0.35;
        y = p < duty ? mid - amp : mid + amp;
        break;
      }
      case 'triangle': {
        let p = (t * cycles + ph) % 1;
        const skew = 0.5 + wp * 0.4;
        const v = p < skew ? (p / skew) * 2 - 1 : 1 - ((p - skew) / (1 - skew)) * 2;
        y = mid - v * amp;
        break;
      }
      case 'noise': {
        const seed = Math.sin(i * 127.1 + i * i * 37.7) * 0.5 + Math.sin(i * 269.5) * 0.3 + Math.sin(i * 53.3) * 0.2;
        y = mid + seed * amp;
        break;
      }
      case 'lowpass': {
        const cutoff = 0.55;
        let gain;
        if (t < cutoff) { gain = 0.85; } else { gain = 0.85 * Math.exp(-(t - cutoff) * 6); }
        gain += 0.2 * Math.exp(-(t - cutoff) * (t - cutoff) * 600);
        y = mid - gain * amp;
        break;
      }
      case 'highpass': {
        const cutoff = 0.4;
        let gain;
        if (t > cutoff) { gain = 0.85; } else { gain = 0.85 * Math.exp(-(cutoff - t) * 6); }
        gain += 0.2 * Math.exp(-(t - cutoff) * (t - cutoff) * 600);
        y = mid - gain * amp;
        break;
      }
      case 'bandpass': {
        const center = 0.5, bw = 0.15;
        const dist = (t - center) / bw;
        const gain = 0.9 * Math.exp(-dist * dist * 0.5);
        y = mid - gain * amp;
        break;
      }
      case 'notch': {
        const center = 0.5, bw = 0.12;
        const dist = (t - center) / bw;
        const gain = 0.85 * (1 - 0.85 * Math.exp(-dist * dist * 0.5));
        y = mid - gain * amp;
        break;
      }
      case 'adsr': {
        const a = 0.04 + (attack ?? 0.2) * 0.32;
        const d = 0.04 + (decay ?? 0.3) * 0.28;
        const sLvl = sustain ?? 0.7;
        const r = 0.04 + (release ?? 0.3) * 0.32;
        const aEnd = a, dEnd = a + d, sEnd = Math.max(dEnd + 0.02, 1 - r);
        let v;
        if (t < aEnd) v = t / aEnd;
        else if (t < dEnd) v = 1 - (1 - sLvl) * ((t - aEnd) / (d || 0.01));
        else if (t < sEnd) v = sLvl;
        else v = sLvl * Math.max(0, 1 - (t - sEnd) / (r || 0.01));
        y = mid - Math.max(0, v) * amp * 1.8;
        break;
      }
      default: {
        const p = t * cycles + ph;
        let v = Math.sin(p * 2 * Math.PI);
        if (wp > 0) v = Math.tanh(v * (1 + wp * 4)) / Math.tanh(1 + wp * 4);
        y = mid - v * amp;
        break;
      }
    }
    points.push({ x, y, t });
  }
  return { points, mid, amp, type };
};

// ── Helper: parse hex/named color to rgba string ──
const colorToRgba = (c, alpha) => {
  if (!c) return `rgba(120,120,255,${alpha})`;
  if (c.startsWith('rgba')) return c;
  if (c.startsWith('rgb')) return c.replace('rgb', 'rgba').replace(')', `,${alpha})`);
  // hex
  const hex = c.replace('#', '');
  const r = parseInt(hex.substring(0, 2), 16) || 120;
  const g = parseInt(hex.substring(2, 4), 16) || 120;
  const b = parseInt(hex.substring(4, 6), 16) || 255;
  return `rgba(${r},${g},${b},${alpha})`;
};

const WaveformVisual = ({ color, width, height, waveType, phase, warp, attack, decay, sustain, release, waveformStyle, waveformSeed }) => {
  const { points, mid, amp, type } = computeWavePoints(width, height, waveType, phase, warp, attack, decay, sustain, release);
  const pointStr = points.map(p => `${p.x},${p.y}`).join(' ');
  const isFilter = ['lowpass', 'highpass', 'bandpass', 'notch'].includes(type);
  const isEnv = type === 'adsr';
  const gridCount = isFilter ? 5 : isEnv ? 4 : 0;
  const style = waveformStyle || 'minimal-line';
  const uid = `wf-${Math.random().toString(36).slice(2, 8)}`;
  // Seed-derived params — each s[i] is 0..1, unique per component instance
  const s = waveformSeed || [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

  // ── Helper: shift hue of the accent color by degrees ──
  const shiftHue = (hexColor, degrees) => {
    const hex = (hexColor || '#667eea').replace('#', '');
    let r = parseInt(hex.substring(0, 2), 16) / 255;
    let g = parseInt(hex.substring(2, 4), 16) / 255;
    let b = parseInt(hex.substring(4, 6), 16) / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, sat, l = (max + min) / 2;
    if (max === min) { h = 0; sat = 0; } else {
      const d = max - min;
      sat = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
      else if (max === g) h = ((b - r) / d + 2) / 6;
      else h = ((r - g) / d + 4) / 6;
    }
    h = ((h * 360 + degrees) % 360) / 360;
    const hue2rgb = (p2, q2, t2) => { if (t2 < 0) t2 += 1; if (t2 > 1) t2 -= 1; if (t2 < 1/6) return p2 + (q2 - p2) * 6 * t2; if (t2 < 1/2) return q2; if (t2 < 2/3) return p2 + (q2 - p2) * (2/3 - t2) * 6; return p2; };
    let rr, gg, bb;
    if (sat === 0) { rr = gg = bb = l; } else {
      const q = l < 0.5 ? l * (1 + sat) : l + sat - l * sat;
      const p = 2 * l - q;
      rr = hue2rgb(p, q, h + 1/3); gg = hue2rgb(p, q, h); bb = hue2rgb(p, q, h - 1/3);
    }
    return `#${Math.round(rr * 255).toString(16).padStart(2, '0')}${Math.round(gg * 255).toString(16).padStart(2, '0')}${Math.round(bb * 255).toString(16).padStart(2, '0')}`;
  };

  // ── Grid + center line helpers ──
  const gridLines = gridCount > 0 ? [...Array(gridCount)].map((_, gi) => {
    const gx = ((gi + 1) / (gridCount + 1)) * width;
    return <line key={gi} x1={gx} y1={4} x2={gx} y2={height - 4} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />;
  }) : null;

  const centerLine = <line x1={0} y1={mid} x2={width} y2={mid} stroke="rgba(255,255,255,0.08)" strokeWidth="1" />;

  // ── 3D WAVETABLE — stacked waveform copies with perspective ──
  if (style === '3d-wavetable') {
    const layers = 4 + Math.floor(s[0] * 5);               // 4-8 layers
    const offsetY = Math.max(2, height * (0.03 + s[1] * 0.04)); // varied spacing
    const phaseSpread = 0.06 + s[2] * 0.14;                 // how different each layer is
    const hueShift = s[3] * 40 - 20;                        // slight color shift back layers
    const fillOpacity = 0.15 + s[4] * 0.25;                 // fill density
    const bgDark = 0.3 + s[5] * 0.2;
    return (
      <svg width={width} height={height} style={{ display: 'block' }}>
        <defs>
          <linearGradient id={`${uid}-bg`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={`rgba(0,0,0,${bgDark * 0.4})`} />
            <stop offset="100%" stopColor={`rgba(0,0,0,${bgDark})`} />
          </linearGradient>
        </defs>
        <rect width={width} height={height} fill={`url(#${uid}-bg)`} rx="4" />
        {gridLines}
        {centerLine}
        {[...Array(layers)].map((_, li) => {
          const layerIdx = layers - 1 - li;
          const dy = layerIdx * offsetY;
          const opacity = 0.06 + (li / layers) * 0.58;
          const phaseShift = layerIdx * phaseSpread;
          const layerColor = layerIdx > 0 ? shiftHue(color, hueShift * (layerIdx / layers)) : color;
          const lp = computeWavePoints(width, height, waveType, (phase ?? 0) + phaseShift, warp, attack, decay, sustain, release).points;
          const lpStr = lp.map(p => `${p.x},${p.y - dy}`).join(' ');
          const fillStr = lpStr + ` ${width},${height - dy} 0,${height - dy}`;
          return (
            <g key={li}>
              <polyline points={fillStr} fill={colorToRgba(layerColor, opacity * fillOpacity)} stroke="none" />
              <polyline points={lpStr} fill="none" stroke={colorToRgba(layerColor, opacity)} strokeWidth={li === layers - 1 ? 2 : 1} strokeLinejoin="round" />
            </g>
          );
        })}
      </svg>
    );
  }

  // ── NEON GLOW — glowing phosphor trace ──
  if (style === 'neon-glow') {
    const blur1 = 2 + s[0] * 4;           // inner glow 2-6
    const blur2 = 4 + s[1] * 8;           // outer glow 4-12
    const strokeW = 1.5 + s[2] * 1.5;     // main stroke 1.5-3
    const glowW = 4 + s[3] * 6;           // glow stroke 4-10
    const bgAlpha = 0.4 + s[4] * 0.2;     // bg darkness
    const highlight = s[5] > 0.5;          // whether to add highlight line
    const secondColor = s[6] > 0.6 ? shiftHue(color, 30 + s[7] * 60) : null; // dual glow
    return (
      <svg width={width} height={height} style={{ display: 'block' }}>
        <defs>
          <filter id={`${uid}-glow`} x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur in="SourceGraphic" stdDeviation={blur1} result="blur1" />
            <feGaussianBlur in="SourceGraphic" stdDeviation={blur2} result="blur2" />
            <feMerge>
              <feMergeNode in="blur2" />
              <feMergeNode in="blur1" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <rect width={width} height={height} fill={`rgba(0,0,0,${bgAlpha})`} rx="4" />
        {gridLines}
        {centerLine}
        {isFilter && <polyline points={pointStr} fill={colorToRgba(color, 0.06)} stroke="none" />}
        {secondColor && <polyline points={pointStr} fill="none" stroke={colorToRgba(secondColor, 0.12)} strokeWidth={glowW + 2} strokeLinejoin="round" filter={`url(#${uid}-glow)`} />}
        <polyline points={pointStr} fill="none" stroke={colorToRgba(color, 0.15)} strokeWidth={glowW} strokeLinejoin="round" filter={`url(#${uid}-glow)`} />
        <polyline points={pointStr} fill="none" stroke={color} strokeWidth={strokeW} strokeLinejoin="round" filter={`url(#${uid}-glow)`} />
        {highlight && <polyline points={pointStr} fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="0.5" strokeLinejoin="round" />}
      </svg>
    );
  }

  // ── GRADIENT FILL — filled waveform with vertical gradient ──
  if (style === 'gradient-fill') {
    const fillPoints = pointStr + ` ${width},${height} 0,${height}`;
    const topOpacity = 0.3 + s[0] * 0.4;           // top gradient 0.3-0.7
    const midOpacity = 0.05 + s[1] * 0.2;           // mid gradient
    const gradAngle = s[2] > 0.7;                   // diagonal gradient variant
    const strokeW = 1.5 + s[3] * 1;                 // stroke width 1.5-2.5
    const secondFill = s[4] > 0.5;                  // dual gradient fill
    const secondHue = 40 + s[5] * 120;              // shift for second gradient
    const highlightOpacity = 0.15 + s[6] * 0.25;
    return (
      <svg width={width} height={height} style={{ display: 'block' }}>
        <defs>
          <linearGradient id={`${uid}-grad`} x1={gradAngle ? "0" : "0"} y1="0" x2={gradAngle ? "0.3" : "0"} y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={topOpacity} />
            <stop offset="60%" stopColor={color} stopOpacity={midOpacity} />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
          {secondFill && (
            <linearGradient id={`${uid}-grad2`} x1="1" y1="0" x2="0.7" y2="1">
              <stop offset="0%" stopColor={shiftHue(color, secondHue)} stopOpacity={topOpacity * 0.5} />
              <stop offset="100%" stopColor={shiftHue(color, secondHue)} stopOpacity="0" />
            </linearGradient>
          )}
        </defs>
        <rect width={width} height={height} fill="rgba(0,0,0,0.3)" rx="4" />
        {gridLines}
        {centerLine}
        {isEnv && <line x1={0} y1={mid - amp * 1.8} x2={width} y2={mid - amp * 1.8} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />}
        <polyline points={fillPoints} fill={`url(#${uid}-grad)`} stroke="none" />
        {secondFill && <polyline points={fillPoints} fill={`url(#${uid}-grad2)`} stroke="none" />}
        <polyline points={pointStr} fill="none" stroke={color} strokeWidth={strokeW} strokeLinejoin="round" />
        <polyline points={pointStr} fill="none" stroke={`rgba(255,255,255,${highlightOpacity})`} strokeWidth="0.5" strokeLinejoin="round" />
      </svg>
    );
  }

  // ── RETRO CRT — phosphor oscilloscope ──
  if (style === 'retro-crt') {
    // Seed-driven phosphor color: green, amber, blue, or cyan
    const phosphorColors = ['#00ff41', '#ffb000', '#00bfff', '#00ffcc', '#ff6b35', '#39ff14'];
    const phosphorIdx = Math.floor(s[0] * phosphorColors.length);
    const phosphor = phosphorColors[Math.min(phosphorIdx, phosphorColors.length - 1)];
    const scanH = 2 + Math.floor(s[1] * 3);          // scan line height 2-4
    const scanOpacity = 0.08 + s[2] * 0.15;           // scan line darkness
    const vGridCount = 4 + Math.floor(s[3] * 5);      // vertical grid lines 4-8
    const hGridCount = 3 + Math.floor(s[4] * 4);      // horizontal grid lines 3-6
    const bloom = 1 + s[5] * 2;                        // bloom intensity
    const vigStrength = 0.25 + s[6] * 0.3;             // vignette
    const strokeW = 1 + s[7] * 1.5;                    // stroke 1-2.5
    const gridAlpha = 0.04 + s[2] * 0.06;
    return (
      <svg width={width} height={height} style={{ display: 'block' }}>
        <defs>
          <filter id={`${uid}-crt`} x="-10%" y="-10%" width="120%" height="120%">
            <feGaussianBlur in="SourceGraphic" stdDeviation={bloom} result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <pattern id={`${uid}-scan`} width={width} height={scanH + 1} patternUnits="userSpaceOnUse">
            <rect width={width} height="1" fill={`rgba(0,0,0,${scanOpacity})`} />
            <rect y="1" width={width} height={scanH} fill="transparent" />
          </pattern>
          <radialGradient id={`${uid}-vig`} cx="50%" cy="50%" r="70%">
            <stop offset="0%" stopColor="transparent" />
            <stop offset="100%" stopColor={`rgba(0,0,0,${vigStrength})`} />
          </radialGradient>
        </defs>
        <rect width={width} height={height} fill="#0a0a0a" rx="4" />
        {[...Array(vGridCount)].map((_, gi) => {
          const gx = ((gi + 1) / (vGridCount + 1)) * width;
          return <line key={`v${gi}`} x1={gx} y1={0} x2={gx} y2={height} stroke={colorToRgba(phosphor, gridAlpha)} strokeWidth="0.5" />;
        })}
        {[...Array(hGridCount)].map((_, gi) => {
          const gy = ((gi + 1) / (hGridCount + 1)) * height;
          return <line key={`h${gi}`} x1={0} y1={gy} x2={width} y2={gy} stroke={colorToRgba(phosphor, gridAlpha)} strokeWidth="0.5" />;
        })}
        <line x1={0} y1={mid} x2={width} y2={mid} stroke={colorToRgba(phosphor, 0.1)} strokeWidth="0.5" />
        <polyline points={pointStr} fill="none" stroke={phosphor} strokeWidth={strokeW} strokeLinejoin="round" filter={`url(#${uid}-crt)`} />
        <rect width={width} height={height} fill={`url(#${uid}-scan)`} rx="4" />
        <rect width={width} height={height} fill={`url(#${uid}-vig)`} rx="4" />
      </svg>
    );
  }

  // ── GLASS PANEL — frosted glass with depth ──
  if (style === 'glass-panel') {
    const borderR = 4 + Math.floor(s[0] * 5);          // roundedness 4-8
    const shineAngle = s[1] > 0.5;                      // top or side shine
    const glassAlpha = 0.04 + s[2] * 0.08;              // glass fill opacity
    const innerShadow = 1 + s[3] * 2;                   // inner shadow depth
    const strokeOpacity = 0.5 + s[4] * 0.4;             // waveform opacity
    const bezelColor = s[5] > 0.5 ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.04)';
    const strokeW = 1 + s[6] * 1;                       // 1-2
    return (
      <svg width={width} height={height} style={{ display: 'block' }}>
        <defs>
          <linearGradient id={`${uid}-glass`} x1={shineAngle ? "0.2" : "0"} y1="0" x2={shineAngle ? "0.8" : "0"} y2="1">
            <stop offset="0%" stopColor={`rgba(255,255,255,${glassAlpha * 2})`} />
            <stop offset="50%" stopColor={`rgba(255,255,255,${glassAlpha * 0.5})`} />
            <stop offset="100%" stopColor={`rgba(0,0,0,${0.1 + s[7] * 0.1})`} />
          </linearGradient>
          <filter id={`${uid}-inner`}>
            <feDropShadow dx="0" dy="1" stdDeviation={innerShadow} floodColor={`rgba(0,0,0,${0.3 + s[3] * 0.3})`} />
          </filter>
        </defs>
        <rect width={width} height={height} fill="rgba(0,0,0,0.35)" rx={borderR} />
        <rect x="1" y="1" width={width - 2} height={height - 2} fill={`url(#${uid}-glass)`} rx={borderR - 1} stroke={bezelColor} strokeWidth="0.5" />
        <rect x="3" y="3" width={width - 6} height={height - 6} fill="rgba(0,0,0,0.2)" rx={borderR - 2} filter={`url(#${uid}-inner)`} />
        {gridLines}
        {centerLine}
        {isEnv && <line x1={0} y1={mid - amp * 1.8} x2={width} y2={mid - amp * 1.8} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />}
        {isFilter && <polyline points={pointStr} fill={colorToRgba(color, 0.08)} stroke="none" />}
        <polyline points={pointStr} fill="none" stroke={colorToRgba(color, strokeOpacity)} strokeWidth={strokeW} strokeLinejoin="round" />
        <line x1={borderR} y1="2" x2={width - borderR} y2="2" stroke="rgba(255,255,255,0.08)" strokeWidth="0.5" />
      </svg>
    );
  }

  // ── HOLOGRAPHIC — rainbow-shifting waveform trace ──
  if (style === 'holographic') {
    const segCount = points.length - 1;
    const hueStart = Math.floor(s[0] * 360);         // starting hue — makes each one unique
    const hueRange = 200 + s[1] * 160;               // how much of the rainbow (200-360)
    const sat = 70 + s[2] * 25;                       // saturation 70-95
    const light = 55 + s[3] * 20;                     // lightness 55-75
    const blurAmt = 0.5 + s[4] * 1.5;                 // glow 0.5-2
    const strokeW = 1.5 + s[5] * 1.5;                 // stroke 1.5-3
    const layers = s[6] > 0.6 ? 2 : 1;                // single or dual trace
    const offsetAmt = layers > 1 ? 1 + s[7] * 2 : 0;  // offset between traces
    return (
      <svg width={width} height={height} style={{ display: 'block' }}>
        <defs>
          <filter id={`${uid}-holo`} x="-10%" y="-10%" width="120%" height="120%">
            <feGaussianBlur in="SourceGraphic" stdDeviation={blurAmt} result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <rect width={width} height={height} fill="rgba(0,0,0,0.4)" rx="4" />
        {gridLines}
        {centerLine}
        {layers > 1 && points.slice(0, -1).map((p, i) => {
          const next = points[i + 1];
          const hue = (hueStart + (i / segCount) * hueRange) % 360;
          return (
            <line key={`s${i}`} x1={p.x} y1={p.y + offsetAmt} x2={next.x} y2={next.y + offsetAmt}
              stroke={`hsl(${hue}, ${sat}%, ${light - 15}%)`} strokeWidth={strokeW + 1} strokeLinecap="round"
              opacity="0.3" filter={`url(#${uid}-holo)`} />
          );
        })}
        {points.slice(0, -1).map((p, i) => {
          const next = points[i + 1];
          const hue = (hueStart + (i / segCount) * hueRange) % 360;
          return (
            <line key={i} x1={p.x} y1={p.y} x2={next.x} y2={next.y}
              stroke={`hsl(${hue}, ${sat}%, ${light}%)`} strokeWidth={strokeW} strokeLinecap="round"
              filter={`url(#${uid}-holo)`} />
          );
        })}
        {points.slice(0, -1).map((p, i) => {
          const next = points[i + 1];
          const hue = (hueStart + (i / segCount) * hueRange) % 360;
          return (
            <line key={`t${i}`} x1={p.x} y1={p.y} x2={next.x} y2={next.y}
              stroke={`hsl(${hue}, 95%, ${light + 15}%)`} strokeWidth="0.5" strokeLinecap="round" />
          );
        })}
      </svg>
    );
  }

  // ── LED MATRIX — dot-matrix display ──
  if (style === 'led-matrix') {
    const density = 3.5 + s[0] * 3;  // dot spacing 3.5-6.5px
    const cols = Math.min(48, Math.max(12, Math.floor(width / density)));
    const rows = Math.min(20, Math.max(5, Math.floor(height / density)));
    const dotScale = 0.25 + s[1] * 0.15;  // dot radius ratio 0.25-0.4
    const dotR = Math.min(width / cols, height / rows) * dotScale;
    const padX = width / cols;
    const padY = height / rows;
    const useSquares = s[2] > 0.7;         // round vs square dots
    const litOpacity = 0.7 + s[3] * 0.25;
    const dimOpacity = 0.02 + s[4] * 0.04;
    const gradientLit = s[5] > 0.4;        // color gradient across columns
    const hueStart = Math.floor(s[6] * 360);
    const hueRange = 40 + s[7] * 80;

    const colValues = [];
    for (let c = 0; c < cols; c++) {
      const t = c / (cols - 1);
      const idx = Math.round(t * (points.length - 1));
      colValues.push(1 - (points[idx].y / height));
    }

    const dots = [];
    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < rows; r++) {
        const normRow = 1 - (r + 0.5) / rows;
        const isLit = normRow <= colValues[c];
        const cx = (c + 0.5) * padX;
        const cy = (r + 0.5) * padY;
        let fill;
        if (isLit) {
          if (gradientLit) {
            const hue = (hueStart + (c / cols) * hueRange) % 360;
            fill = `hsla(${hue}, 80%, 60%, ${litOpacity})`;
          } else {
            fill = colorToRgba(color, litOpacity);
          }
        } else {
          fill = `rgba(255,255,255,${dimOpacity})`;
        }
        if (useSquares) {
          dots.push(
            <rect key={`${c}-${r}`} x={cx - dotR} y={cy - dotR} width={dotR * 2} height={dotR * 2} rx={dotR * 0.2} fill={fill} />
          );
        } else {
          dots.push(
            <circle key={`${c}-${r}`} cx={cx} cy={cy} r={dotR} fill={fill} />
          );
        }
      }
    }

    return (
      <svg width={width} height={height} style={{ display: 'block' }}>
        <rect width={width} height={height} fill={`rgba(0,0,0,${0.4 + s[0] * 0.15})`} rx="4" />
        {dots}
      </svg>
    );
  }

  // ── MINIMAL LINE — clean default (fallback) ──
  const bgTopAlpha = 0.02 + s[0] * 0.04;
  const bgBotAlpha = 0.2 + s[1] * 0.2;
  const strokeW = 1 + s[2] * 1;           // 1-2
  const fillUnder = s[3] > 0.5;
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <defs>
        <linearGradient id={`${uid}-mbg`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={`rgba(255,255,255,${bgTopAlpha})`} />
          <stop offset="100%" stopColor={`rgba(0,0,0,${bgBotAlpha})`} />
        </linearGradient>
      </defs>
      <rect width={width} height={height} fill={`url(#${uid}-mbg)`} rx="4" />
      {gridLines}
      {centerLine}
      {isEnv && <line x1={0} y1={mid - amp * 1.8} x2={width} y2={mid - amp * 1.8} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />}
      {isFilter && <polyline points={pointStr} fill={colorToRgba(color, 0.08)} stroke="none" />}
      {fillUnder && <polyline points={pointStr + ` ${width},${height} 0,${height}`} fill={colorToRgba(color, 0.06)} stroke="none" />}
      <polyline points={pointStr} fill="none" stroke={color} strokeWidth={strokeW} strokeLinejoin="round" />
    </svg>
  );
};

const XYPadVisual = ({ color, width, height }) => {
  const gridLines = 4;
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <rect width={width} height={height} fill="rgba(0,0,0,0.3)" rx="4" stroke={color} strokeWidth="1" strokeOpacity="0.3" />
      {[...Array(gridLines - 1)].map((_, i) => {
        const x = ((i + 1) / gridLines) * width;
        const y = ((i + 1) / gridLines) * height;
        return (
          <g key={i}>
            <line x1={x} y1={0} x2={x} y2={height} stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
            <line x1={0} y1={y} x2={width} y2={y} stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
          </g>
        );
      })}
      <line x1={width * 0.6} y1={0} x2={width * 0.6} y2={height} stroke={color} strokeWidth="0.5" strokeOpacity="0.4" />
      <line x1={0} y1={height * 0.4} x2={width} y2={height * 0.4} stroke={color} strokeWidth="0.5" strokeOpacity="0.4" />
      <circle cx={width * 0.6} cy={height * 0.4} r={5} fill={color} opacity="0.9" />
      <circle cx={width * 0.6} cy={height * 0.4} r={8} fill="none" stroke={color} strokeWidth="1" opacity="0.4" />
    </svg>
  );
};

/* ---- Resize Handles ---- */

const ResizeHandles = ({ component, onUpdate, onDragStop }) => {
  const startResize = useCallback((corner, clientX, clientY) => {
    const startX = clientX;
    const startY = clientY;
    const startW = component.width;
    const startH = component.height;
    const startCompX = component.x;
    const startCompY = component.y;

    const doResize = (cx, cy) => {
      const dx = cx - startX;
      const dy = cy - startY;
      let newW = startW, newH = startH, newX = startCompX, newY = startCompY;

      if (corner.includes('e')) newW = Math.max(10, startW + dx);
      if (corner.includes('w')) { newW = Math.max(10, startW - dx); newX = startCompX + dx; }
      if (corner.includes('s')) newH = Math.max(10, startH + dy);
      if (corner.includes('n')) { newH = Math.max(10, startH - dy); newY = startCompY + dy; }

      // For knobs keep square
      if (component.type === 'knob' || component.type === 'led') {
        const size = Math.max(newW, newH);
        newW = size;
        newH = size;
      }

      onUpdate(component.id, { width: Math.round(newW), height: Math.round(newH), x: Math.round(newX), y: Math.round(newY) });
    };

    const onMouseMove = (me) => doResize(me.clientX, me.clientY);
    const onTouchMove = (te) => { te.preventDefault(); doResize(te.touches[0].clientX, te.touches[0].clientY); };
    const cleanup = () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', cleanup);
      window.removeEventListener('touchmove', onTouchMove);
      window.removeEventListener('touchend', cleanup);
      if (onDragStop) onDragStop();
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', cleanup);
    window.addEventListener('touchmove', onTouchMove, { passive: false });
    window.addEventListener('touchend', cleanup);
  }, [component, onUpdate, onDragStop]);

  const handleMouseDown = useCallback((corner, e) => {
    e.stopPropagation();
    e.preventDefault();
    startResize(corner, e.clientX, e.clientY);
  }, [startResize]);

  const handleTouchStart = useCallback((corner, e) => {
    e.stopPropagation();
    e.preventDefault();
    const t = e.touches[0];
    startResize(corner, t.clientX, t.clientY);
  }, [startResize]);

  const handleStyle = (cursor) => ({
    position: 'absolute',
    width: 8, height: 8,
    background: 'rgba(186, 156, 255, 0.9)',
    border: '1px solid rgba(255,255,255,0.8)',
    borderRadius: 2,
    cursor,
    zIndex: 9999,
    touchAction: 'none',
  });

  return (
    <>
      <div style={{ ...handleStyle('nw-resize'), top: -5, left: -5 }} onMouseDown={(e) => handleMouseDown('nw', e)} onTouchStart={(e) => handleTouchStart('nw', e)} />
      <div style={{ ...handleStyle('ne-resize'), top: -5, right: -5 }} onMouseDown={(e) => handleMouseDown('ne', e)} onTouchStart={(e) => handleTouchStart('ne', e)} />
      <div style={{ ...handleStyle('sw-resize'), bottom: -5, left: -5 }} onMouseDown={(e) => handleMouseDown('sw', e)} onTouchStart={(e) => handleTouchStart('sw', e)} />
      <div style={{ ...handleStyle('se-resize'), bottom: -5, right: -5 }} onMouseDown={(e) => handleMouseDown('se', e)} onTouchStart={(e) => handleTouchStart('se', e)} />
    </>
  );
};

/* ---- Main CanvasComponent ---- */

const CanvasComponent = ({ component, isSelected, onSelect, onUpdate, onDragStop, editorMode, paramValue, onParamChange, onContextMenu, allComponents, allParamValues, boundParamId, isGenerating, onGroupHover, groupVisible, engine }) => {
  // Get param controller from context for ModMatrix etc
  const { paramController } = usePluginContext() || {};
  const nodeRef = useRef(null);
  const [buttonPressed, setButtonPressed] = useState(false);
  const [liveMeterLevel, setLiveMeterLevel] = useState(null);
  const meterAnimRef = useRef(null);

  const isTestMode = editorMode === 'test';
  const isInteractive = isTestMode && ['knob', 'slider', 'xy-pad', 'button', 'click-knob'].includes(component.type);

  // Live meter animation: poll engine for compressor reduction or output level
  useEffect(() => {
    if (!isTestMode || component.type !== 'meter' || !engine) {
      setLiveMeterLevel(null);
      return;
    }
    const tick = () => {
      const reduction = engine.getReduction?.() || 0; // negative dB
      if (reduction < -0.1) {
        // Gain reduction meter: map -30..0 dB → 0..1
        setLiveMeterLevel(Math.min(1, Math.abs(reduction) / 30));
      } else {
        // No compression active — show output level
        const level = engine.getOutputLevel?.() || 0;
        setLiveMeterLevel(level);
      }
      meterAnimRef.current = requestAnimationFrame(tick);
    };
    meterAnimRef.current = requestAnimationFrame(tick);
    return () => { if (meterAnimRef.current) cancelAnimationFrame(meterAnimRef.current); };
  }, [isTestMode, component.type, engine]);

  // Async Flux image resolution — replaces placeholder SVG with photorealistic image
  useEffect(() => {
    if (component.svgStyle === 'flux' && component.fluxPrompt && !component._fluxResolved && !component._fluxLoading && !component._fluxFailed) {
      onUpdate(component.id, { _fluxLoading: true });
      resolveFluxComponent(component).then(result => {
        if (result?.sprite) {
          onUpdate(component.id, { sprite: result.sprite, flatRing: result.flatRing || null, _fluxLoading: false, _fluxResolved: true });
        } else if (result) {
          onUpdate(component.id, { svg: result, _fluxLoading: false, _fluxResolved: true });
        } else {
          // Generation returned null — mark failed so we don't retry infinitely
          console.warn('[Flux] Generation returned null for:', component.type, component.fluxPrompt);
          onUpdate(component.id, { _fluxLoading: false, _fluxFailed: true });
        }
      }).catch(err => {
        console.error('[Flux] Resolution error:', err.message);
        onUpdate(component.id, { _fluxLoading: false, _fluxFailed: true });
      });
    }
  }, [component.id, component.svgStyle, component.fluxPrompt, component._fluxResolved, component._fluxLoading, component._fluxFailed, onUpdate]);

  const handleDrag = useCallback((e, data) => {
    onUpdate(component.id, { x: data.x, y: data.y });
  }, [component.id, onUpdate]);

  const handleDragStop = useCallback(() => {
    if (onDragStop) onDragStop();
  }, [onDragStop]);

  const handleClick = useCallback((e) => {
    e.stopPropagation();
    if (!isTestMode) {
      onSelect(component.id, e.shiftKey);
    }
  }, [component.id, onSelect, isTestMode]);

  const handleRightClick = useCallback((e) => {
    if (!isTestMode && onContextMenu) {
      onContextMenu(e, component.id);
    }
  }, [component.id, isTestMode, onContextMenu]);

  const handleParamChange = useCallback((val) => {
    if (onParamChange) onParamChange(component.id, val);
  }, [component.id, onParamChange]);

  // Determine the waveform shape + linked knob values using section (panel) containment
  const getWaveformContext = () => {
    const result = { shape: 'sine', phase: 0, warp: 0, attack: 0.2, decay: 0.3, sustain: 0.7, release: 0.3 };
    if (!allComponents || !allParamValues) return result;
    const cx = component.x + component.width / 2;
    const cy = component.y + component.height / 2;

    // Find the smallest panel containing this waveform (= section boundary)
    let panel = null, panelArea = Infinity;
    for (const c of allComponents) {
      if (c.type !== 'panel') continue;
      if (cx >= c.x && cx <= c.x + c.width && cy >= c.y && cy <= c.y + c.height) {
        const area = c.width * c.height;
        if (area < panelArea) { panelArea = area; panel = c; }
      }
    }

    // Check if a component's center is inside the same panel
    const inSection = (other) => {
      if (!panel) return true; // no panel → allow all
      const ox = other.x + other.width / 2;
      const oy = other.y + other.height / 2;
      return ox >= panel.x && ox <= panel.x + panel.width &&
             oy >= panel.y && oy <= panel.y + panel.height;
    };

    // Find nearest component of a given type + label pattern within this section
    const findInSection = (type, pattern) => {
      let closest = null, minDist = Infinity;
      for (const c of allComponents) {
        if (c.type !== type || !pattern.test(c.label || '') || !inSection(c)) continue;
        const dx = (c.x + c.width / 2) - cx;
        const dy = (c.y + c.height / 2) - cy;
        const d = dx * dx + dy * dy;
        if (d < minDist) { minDist = d; closest = c; }
      }
      return closest;
    };

    // --- Oscillator: "Wave Type" dropdown in this section ---
    const waveDD = findInSection('dropdown', /wave\s*type/i);
    if (waveDD) {
      const options = getDropdownOptions(waveDD.label);
      const val = allParamValues[waveDD.id] ?? 0;
      const idx = Math.min(options.length - 1, Math.max(0, Math.round(val * (options.length - 1))));
      result.shape = (options[idx] || 'Sine').toLowerCase();
      const phaseKnob = findInSection('knob', /^phase$/i);
      const warpKnob = findInSection('knob', /^warp$/i);
      if (phaseKnob) result.phase = allParamValues[phaseKnob.id] ?? 0;
      if (warpKnob) result.warp = allParamValues[warpKnob.id] ?? 0;
      return result;
    }

    // --- Filter: "Filter Type" dropdown in this section ---
    const filterDD = findInSection('dropdown', /filter\s*type/i);
    if (filterDD) {
      const options = getDropdownOptions(filterDD.label);
      const val = allParamValues[filterDD.id] ?? 0;
      const idx = Math.min(options.length - 1, Math.max(0, Math.round(val * (options.length - 1))));
      const ft = (options[idx] || '').toLowerCase();
      if (ft.includes('high')) result.shape = 'highpass';
      else if (ft.includes('band')) result.shape = 'bandpass';
      else if (ft.includes('notch')) result.shape = 'notch';
      else result.shape = 'lowpass';
      return result;
    }

    // --- Envelope: ADSR knobs in this section ---
    const attackK = findInSection('knob', /^attack$/i);
    const decayK = findInSection('knob', /^decay$/i);
    const sustainK = findInSection('knob', /^sustain$/i);
    const releaseK = findInSection('knob', /^release$/i);
    if (attackK || decayK || sustainK || releaseK) {
      result.shape = 'adsr';
      if (attackK) result.attack = allParamValues[attackK.id] ?? 0.2;
      if (decayK) result.decay = allParamValues[decayK.id] ?? 0.3;
      if (sustainK) result.sustain = allParamValues[sustainK.id] ?? 0.7;
      if (releaseK) result.release = allParamValues[releaseK.id] ?? 0.3;
      return result;
    }

    return result;
  };

  const renderVisual = () => {
    // Priority 1: Sprite image (resolved DALL-E URL)
    if (component.sprite && typeof component.sprite === 'string') {
      const iR = component.innerRotates ?? false;
      const oR = component.outerRotates ?? true;
      const iRS = component.innerRotateSpeed ?? 1;
      const oRS = component.outerRotateSpeed ?? 1;
      if (isTestMode) {
        if (component.type === 'knob') return <SpriteKnobRenderer spriteUrl={component.sprite} flatRingUrl={component.flatRing} size={component.width} value={paramValue} isTestMode onChange={handleParamChange} indicatorColor={component.indicatorColor || component.color} innerRotates={iR} outerRotates={oR} innerRotateSpeed={iRS} outerRotateSpeed={oRS} />;
        if (component.type === 'slider') return <SpriteSliderRenderer spriteUrl={component.sprite} width={component.width} height={component.height} value={paramValue} isTestMode onChange={handleParamChange} />;
        if (component.type === 'button') return <SpriteButtonRenderer spriteUrl={component.sprite} pressedSpriteUrl={component.pressedSprite} width={component.width} height={component.height} pressed={buttonPressed} />;
      }
      if (component.type === 'knob') return <SpriteKnobRenderer spriteUrl={component.sprite} flatRingUrl={component.flatRing} size={component.width} value={0.65} indicatorColor={component.indicatorColor || component.color} innerRotates={iR} outerRotates={oR} innerRotateSpeed={iRS} outerRotateSpeed={oRS} />;
      if (component.type === 'slider') return <SpriteSliderRenderer spriteUrl={component.sprite} width={component.width} height={component.height} value={0.3} />;
      if (component.type === 'button') return <SpriteButtonRenderer spriteUrl={component.sprite} pressedSpriteUrl={component.pressedSprite} width={component.width} height={component.height} pressed={false} />;
      // For non-knob/slider/button sprites, render as image
      return <ImageVisual width={component.width} height={component.height} image={component.sprite} borderRadius={component.borderRadius} />;
    }

    // Priority 1.5a: Moog-style knobs with pre-baked images → use arc-indicator renderer
    if (component.type === 'knob' && MOOG_KNOB_IMAGES[component.svgStyle]) {
      const moogData = MOOG_KNOB_IMAGES[component.svgStyle];
      const iColor = component.indicatorColor || component.color;
      if (isTestMode) return <SpriteKnobRenderer spriteUrl={moogData.full} flatRingUrl={moogData.flat} size={component.width} value={paramValue} isTestMode onChange={handleParamChange} indicatorColor={iColor} />;
      return <SpriteKnobRenderer spriteUrl={moogData.full} flatRingUrl={moogData.flat} size={component.width} value={0.65} indicatorColor={iColor} />;
    }

    // Priority 1.5b: Lazy-generate SVG from svgStyle if not already resolved
    if (component.svgStyle && !component.svg && !component.sprite) {
      const uid = (component.id || 'c').replace(/[^a-zA-Z0-9]/g, '').slice(0, 12);
      const params = { width: component.width, height: component.height,
        bodyColor: component.bodyColor || component.color || '#333',
        indicatorColor: component.indicatorColor || component.color || '#fff',
        accentColor: component.accentColor || component.color || '#888', uid, label: component.label || '' };
      if (component.type === 'knob') component.svg = generateKnobSVG(component.svgStyle, params);
      else if (component.type === 'slider') component.svg = generateSliderSVG(component.svgStyle, params);
      else if (component.type === 'button') component.svg = generateButtonSVG(component.svgStyle, params);
    }

    // Priority 2: Custom SVG (LLM-generated or library-generated)
    if (component.svg) {
      if (isTestMode) {
        switch (component.type) {
          case 'knob':
            return <SVGKnobRenderer svgString={component.svg} size={component.width} value={paramValue} color={component.color} isTestMode onChange={handleParamChange} knobImages={MOOG_KNOB_IMAGES[component.svgStyle]} />;
          case 'slider':
            return <SVGSliderRenderer svgString={component.svg} width={component.width} height={component.height} value={paramValue} color={component.color} isTestMode onChange={handleParamChange} />;
          case 'button':
            return <SVGButtonRenderer svgString={component.svg} width={component.width} height={component.height} pressed={buttonPressed} color={component.color} />;
          default:
            return <SVGStaticRenderer svgString={component.svg} width={component.width} height={component.height} />;
        }
      }
      switch (component.type) {
        case 'knob':
          return <SVGKnobRenderer svgString={component.svg} size={component.width} value={0.65} color={component.color} knobImages={MOOG_KNOB_IMAGES[component.svgStyle]} />;
        case 'slider':
          return <SVGSliderRenderer svgString={component.svg} width={component.width} height={component.height} value={0.3} color={component.color} />;
        case 'button':
          return <SVGButtonRenderer svgString={component.svg} width={component.width} height={component.height} pressed={false} color={component.color} />;
        default:
          return <SVGStaticRenderer svgString={component.svg} width={component.width} height={component.height} />;
      }
    }

    // Priority 3: Legacy knobStyle preset rendering
    if (isTestMode) {
      switch (component.type) {
        case 'knob':
          return <InteractiveKnob color={component.color} size={component.width} value={paramValue} onChange={handleParamChange} />;
        case 'slider':
          return <InteractiveSlider color={component.color} width={component.width} height={component.height} value={paramValue} onChange={handleParamChange} />;
        case 'button':
          return <ButtonVisual color={component.color} label={component.label} width={component.width} height={component.height} fontSize={component.fontSize} borderRadius={component.borderRadius} pressed={buttonPressed} knobStyle={component.knobStyle} />;
        case 'xy-pad':
          return <XYPad color={component.color} width={component.width} height={component.height} x={paramValue} y={paramValue} onChange={handleParamChange} />;
        case 'click-knob':
          return (
            <ClickToTypeKnob value={paramValue} onChange={handleParamChange} color={component.color} width={component.width} label={component.label}>
              <KnobVisual color={component.color} size={component.width} knobStyle={component.knobStyle} />
            </ClickToTypeKnob>
          );
        case 'spectrum-analyzer':
          return <SpectrumAnalyzer width={component.width} height={component.height} color={component.color} />;
        case 'mseg-editor':
          return <MSEGEditor width={component.width} height={component.height} color={component.color} paramController={paramController} />;
        case 'oscilloscope':
          return <Oscilloscope width={component.width} height={component.height} color={component.color} />;
        case 'adsr':
          return <ADSRDisplay width={component.width} height={component.height} color={component.color} attack={paramController?.getBaseValue?.("attack") ?? 0.1} decay={paramController?.getBaseValue?.("decay") ?? 0.3} sustain={paramController?.getBaseValue?.("sustain") ?? 0.7} release={paramController?.getBaseValue?.("release") ?? 0.5} onParamChange={(k,v) => paramController?.setBaseValue?.(k,v)} />;
        case 'eq-curve':
          return <EQCurveDisplay width={component.width} height={component.height} color={component.color} paramController={paramController} />;
        case 'mod-matrix':
          return <ModMatrixWithSliders color={component.color} connections={paramController?.getConnections?.()} onChange={(c) => paramController?.setModConnections?.(c)} />;
        // END NEW CASES
        case 'spectrum-analyzer-dup':
          return <SpectrumAnalyzer width={component.width} height={component.height} color={component.color} />;
        case 'meter':
          return <MeterVisual color={component.color} width={component.width} height={component.height} level={liveMeterLevel ?? paramValue} svgStyle={component.svgStyle} bodyColor={component.bodyColor} indicatorColor={component.indicatorColor} accentColor={component.accentColor} label={component.label} />;
        case 'led':
          return <LEDVisual color={component.color} size={component.width} lit={paramValue > 0.5} />;
        case 'dropdown': {
          const options = getDropdownOptions(component.label);
          return <InteractiveDropdown color={component.color} label={component.label} width={component.width} fontSize={component.fontSize} value={paramValue} onChange={handleParamChange} options={options} />;
        }
        case 'waveform': {
          const wCtx = getWaveformContext();
          return <WaveformVisual color={component.color} width={component.width} height={component.height} waveType={wCtx.shape} phase={wCtx.phase} warp={wCtx.warp} attack={wCtx.attack} decay={wCtx.decay} sustain={wCtx.sustain} release={wCtx.release} waveformStyle={component.waveformStyle} waveformSeed={component.waveformSeed} />;
        }
        case 'wavetable_3d': {
          const wCtx3d = getWaveformContext();
          return <WaveformVisual color={component.color} width={component.width} height={component.height} waveType={wCtx3d.shape} phase={wCtx3d.phase} warp={wCtx3d.warp} waveformStyle="3d-wavetable" waveformSeed={component.waveformSeed} />;
        }
        default:
          break; // fall through to static rendering
      }
    }

    switch (component.type) {
      case 'knob':
        return <KnobVisual color={component.color} size={component.width} knobStyle={component.knobStyle} />;
      case 'slider':
        return <SliderVisual color={component.color} width={component.width} height={component.height} knobStyle={component.knobStyle} />;
      case 'button':
        return <ButtonVisual color={component.color} label={component.label} width={component.width} height={component.height} fontSize={component.fontSize} borderRadius={component.borderRadius} knobStyle={component.knobStyle} />;
      case 'label':
        return <LabelVisual color={component.color} label={component.label} fontSize={component.fontSize} letterSpacing={component.letterSpacing} />;
      case 'led':
        return <LEDVisual color={component.color} size={component.width} />;
      case 'dropdown':
        return <DropdownVisual color={component.color} label={component.label} width={component.width} fontSize={component.fontSize} />;
      case 'image':
        return <ImageVisual width={component.width} height={component.height} image={component.image} borderRadius={component.borderRadius} />;
      case 'panel':
        return <PanelVisual width={component.width} height={component.height} borderColor={component.borderColor} bgColor={component.bgColor} borderRadius={component.borderRadius} bgGradient={component.bgGradient} backdropBlur={component.backdropBlur} boxShadow={component.boxShadow} />;
      case 'meter':
        return <MeterVisual color={component.color} width={component.width} height={component.height} svgStyle={component.svgStyle} bodyColor={component.bodyColor} indicatorColor={component.indicatorColor} accentColor={component.accentColor} label={component.label} />;
      case 'waveform': {
        const wCtx = getWaveformContext();
        return <WaveformVisual color={component.color} width={component.width} height={component.height} waveType={wCtx.shape} phase={wCtx.phase} warp={wCtx.warp} waveformStyle={component.waveformStyle} waveformSeed={component.waveformSeed} />;
      }
      case 'wavetable_3d': {
        const wCtx3d = getWaveformContext();
        return <WaveformVisual color={component.color} width={component.width} height={component.height} waveType={wCtx3d.shape} phase={wCtx3d.phase} warp={wCtx3d.warp} waveformStyle="3d-wavetable" waveformSeed={component.waveformSeed} />;
      }
      case 'xy-pad':
        return <XYPadVisual color={component.color} width={component.width} height={component.height} />;
      case 'click-knob':
        return <KnobVisual color={component.color} size={component.width} knobStyle={component.knobStyle} />;
      case 'spectrum-analyzer':
          return <SpectrumAnalyzer width={component.width} height={component.height} color={component.color} />;
        case 'mseg-editor':
          return <MSEGEditor width={component.width} height={component.height} color={component.color} paramController={paramController} />;
        case 'oscilloscope':
          return <Oscilloscope width={component.width} height={component.height} color={component.color} />;
        case 'adsr':
          return <ADSRDisplay width={component.width} height={component.height} color={component.color} attack={paramController?.getBaseValue?.("attack") ?? 0.1} decay={paramController?.getBaseValue?.("decay") ?? 0.3} sustain={paramController?.getBaseValue?.("sustain") ?? 0.7} release={paramController?.getBaseValue?.("release") ?? 0.5} onParamChange={(k,v) => paramController?.setBaseValue?.(k,v)} />;
        case 'eq-curve':
          return <EQCurveDisplay width={component.width} height={component.height} color={component.color} paramController={paramController} />;
        case 'mod-matrix':
          return <ModMatrixWithSliders color={component.color} connections={paramController?.getConnections?.()} onChange={(c) => paramController?.setModConnections?.(c)} />;
        // END NEW CASES
        case 'spectrum-analyzer-dup':
        return <SpectrumAnalyzer width={component.width} height={component.height} color={component.color} />;
      default:
        return null;
    }
  };

  const showSubLabel = component.label && component.type !== 'label' && component.type !== 'button' && component.type !== 'panel';

  // groupVisible: undefined = normal, true = fade in, false = fade out
  const isGroupFaded = groupVisible === false;
  const wrapperStyle = {
    opacity: isGroupFaded ? 0 : (component.opacity ?? 1),
    transform: component.rotation ? `rotate(${component.rotation}deg)` : undefined,
    zIndex: component.zIndex ?? 1,
    boxShadow: component.type !== 'panel' ? (component.boxShadow || undefined) : undefined,
    position: isTestMode ? 'absolute' : undefined,
    left: isTestMode ? component.x : undefined,
    top: isTestMode ? component.y : undefined,
    transition: groupVisible !== undefined ? 'opacity 0.18s ease' : undefined,
    pointerEvents: isGroupFaded ? 'none' : undefined,
  };

  // Build ARIA attributes for test mode interactive components
  const getAriaProps = () => {
    if (!isTestMode) return {};
    const type = component.type;
    if (type === 'knob' || type === 'slider' || type === 'click-knob') {
      return {
        role: 'slider',
        'aria-valuenow': Math.round(paramValue * 100),
        'aria-valuemin': 0,
        'aria-valuemax': 100,
        'aria-label': component.label || `${type} control`,
      };
    }
    if (type === 'button') {
      return {
        role: 'button',
        'aria-label': component.label || 'button',
        'aria-pressed': buttonPressed,
      };
    }
    if (type === 'label') {
      return {
        'aria-label': component.label || 'label',
      };
    }
    if (type === 'dropdown') {
      return {
        role: 'listbox',
        'aria-label': component.label || 'dropdown',
      };
    }
    if (type === 'xy-pad') {
      return {
        role: 'slider',
        'aria-valuenow': Math.round(paramValue * 100),
        'aria-valuemin': 0,
        'aria-valuemax': 100,
        'aria-label': component.label || 'XY pad',
      };
    }
    // Display-only components: meter, waveform, led, spectrum-analyzer, oscilloscope, adsr, eq-curve, etc.
    if (['meter', 'waveform', 'led', 'spectrum-analyzer', 'oscilloscope', 'adsr', 'eq-curve', 'spectrum-analyzer-dup'].includes(type)) {
      return {
        role: 'img',
        'aria-label': component.label || `${type} display`,
      };
    }
    return {};
  };

  // In test mode, render without Draggable
  if (isTestMode) {
    const ariaProps = getAriaProps();
    return (
      <div
        ref={nodeRef}
        className={styles.canvasComp}
        style={wrapperStyle}
        onMouseDown={component.type === 'button' ? () => setButtonPressed(true) : undefined}
        onMouseUp={component.type === 'button' ? () => setButtonPressed(false) : undefined}
        onMouseLeave={component.type === 'button' ? () => { setButtonPressed(false); if (component.isGroupParent && onGroupHover) onGroupHover(null); } : (component.isGroupParent && onGroupHover ? () => onGroupHover(null) : undefined)}
        onMouseEnter={component.isGroupParent && onGroupHover ? () => onGroupHover(component.groupId) : undefined}
        {...ariaProps}
      >
        {renderVisual()}
        {showSubLabel && (
          <div className={styles.compLabel} aria-label={component.label}>
            {component.label}
            {['knob', 'slider'].includes(component.type) && (
              <span className={styles.paramValueLabel}> {Math.round(paramValue * 100)}%</span>
            )}
          </div>
        )}
      </div>
    );
  }

  // Hidden components are not rendered in edit mode
  if (component.hidden && !isTestMode) {
    return null;
  }

  // Edit mode — draggable with resize handles
  return (
    <Draggable
      nodeRef={nodeRef}
      position={{ x: component.x, y: component.y }}
      onDrag={handleDrag}
      onStop={handleDragStop}
      bounds="parent"
      disabled={!!component.locked}
    >
      <div
        ref={nodeRef}
        className={`${styles.canvasComp} ${isSelected ? styles.canvasCompSelected : ''}`}
        onClick={handleClick}
        onContextMenu={handleRightClick}
        style={{
          ...wrapperStyle,
          cursor: component.locked ? 'not-allowed' : undefined,
          opacity: component.locked ? 0.7 : (wrapperStyle.opacity ?? 1),
        }}
      >
        {renderVisual()}
        {showSubLabel && <div className={styles.compLabel}>{component.label}</div>}
        {/* Image generation loading indicator */}
        {(isGenerating || component._fluxLoading) && (
          <div style={{
            position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(0,0,0,0.4)', borderRadius: 'inherit', zIndex: 9997,
            pointerEvents: 'none',
          }}>
            <i className="fa-solid fa-spinner fa-spin" style={{ fontSize: Math.min(component.width, component.height) * 0.3, color: 'rgba(186,156,255,0.8)' }} />
          </div>
        )}
        {/* Lock indicator + locked overlay */}
        {component.locked && (
          <>
            <div style={{
              position: 'absolute', top: -3, left: -3, width: 14, height: 14,
              borderRadius: '50%', background: 'rgba(245,166,35,0.9)', display: 'flex',
              alignItems: 'center', justifyContent: 'center', zIndex: 10000,
            }}>
              <i className="fa-solid fa-lock" style={{ fontSize: 7, color: '#000' }} />
            </div>
            <div style={{
              position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
              pointerEvents: 'none', zIndex: 9998, borderRadius: 'inherit',
              background: 'repeating-linear-gradient(135deg, transparent, transparent 4px, rgba(245,166,35,0.06) 4px, rgba(245,166,35,0.06) 8px)',
            }} />
          </>
        )}
        {/* Parameter binding indicator — hidden by default, shown on hover only */}
        {isSelected && !component.locked && <ResizeHandles component={component} onUpdate={onUpdate} onDragStop={onDragStop} />}
      </div>
    </Draggable>
  );
};

export default CanvasComponent;
