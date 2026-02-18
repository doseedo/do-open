import React, { useRef, useCallback, useState, useEffect } from 'react';
import Draggable from 'react-draggable';
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

const KnobVisual = ({ color, size }) => {
  const r = size / 2 - 4;
  const cx = size / 2;
  const cy = size / 2;
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

const SliderVisual = ({ color, width, height }) => (
  <div style={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <div style={{ width: 4, height: height - 10, background: 'rgba(255,255,255,0.12)', borderRadius: 2, position: 'relative' }}>
      <div style={{
        position: 'absolute', top: '30%', left: '50%', transform: 'translateX(-50%)',
        width: 16, height: 8, background: color, borderRadius: 3,
      }} />
    </div>
  </div>
);

const ButtonVisual = ({ color, label, width, height, fontSize, borderRadius, pressed }) => (
  <div style={{
    width, height, background: pressed ? `color-mix(in srgb, ${color} 70%, white)` : color,
    borderRadius: borderRadius || 4,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: fontSize || 11, color: '#fff', fontWeight: 600,
    transform: pressed ? 'scale(0.95)' : undefined,
    transition: 'transform 0.1s, background 0.1s',
  }}>
    {label}
  </div>
);

const LabelVisual = ({ color, label, fontSize }) => (
  <div style={{ color, fontSize: fontSize || 13, fontWeight: 500, whiteSpace: 'nowrap', userSelect: 'none' }}>
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

const MeterVisual = ({ color, width, height, level }) => {
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

const WaveformVisual = ({ color, width, height }) => {
  const points = [];
  const steps = 40;
  for (let i = 0; i <= steps; i++) {
    const x = (i / steps) * width;
    const y = height / 2 + Math.sin(i * 0.5) * (height * 0.35) * (0.5 + 0.5 * Math.sin(i * 0.12));
    points.push(`${x},${y}`);
  }
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <rect width={width} height={height} fill="rgba(0,0,0,0.3)" rx="4" />
      <line x1={0} y1={height / 2} x2={width} y2={height / 2} stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
      <polyline points={points.join(' ')} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
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
  const handleMouseDown = useCallback((corner, e) => {
    e.stopPropagation();
    e.preventDefault();
    const startX = e.clientX;
    const startY = e.clientY;
    const startW = component.width;
    const startH = component.height;
    const startCompX = component.x;
    const startCompY = component.y;

    const onMove = (me) => {
      const dx = me.clientX - startX;
      const dy = me.clientY - startY;
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
    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      if (onDragStop) onDragStop();
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, [component, onUpdate, onDragStop]);

  const handleStyle = (cursor) => ({
    position: 'absolute',
    width: 8, height: 8,
    background: 'rgba(186, 156, 255, 0.9)',
    border: '1px solid rgba(255,255,255,0.8)',
    borderRadius: 2,
    cursor,
    zIndex: 9999,
  });

  return (
    <>
      <div style={{ ...handleStyle('nw-resize'), top: -5, left: -5 }} onMouseDown={(e) => handleMouseDown('nw', e)} />
      <div style={{ ...handleStyle('ne-resize'), top: -5, right: -5 }} onMouseDown={(e) => handleMouseDown('ne', e)} />
      <div style={{ ...handleStyle('sw-resize'), bottom: -5, left: -5 }} onMouseDown={(e) => handleMouseDown('sw', e)} />
      <div style={{ ...handleStyle('se-resize'), bottom: -5, right: -5 }} onMouseDown={(e) => handleMouseDown('se', e)} />
    </>
  );
};

/* ---- Main CanvasComponent ---- */

const CanvasComponent = ({ component, isSelected, onSelect, onUpdate, onDragStop, editorMode, paramValue, onParamChange }) => {
  const nodeRef = useRef(null);
  const [buttonPressed, setButtonPressed] = useState(false);

  const isTestMode = editorMode === 'test';
  const isInteractive = isTestMode && ['knob', 'slider', 'xy-pad', 'button'].includes(component.type);

  const handleDrag = useCallback((e, data) => {
    onUpdate(component.id, { x: data.x, y: data.y });
  }, [component.id, onUpdate]);

  const handleDragStop = useCallback(() => {
    if (onDragStop) onDragStop();
  }, [onDragStop]);

  const handleClick = useCallback((e) => {
    e.stopPropagation();
    if (!isTestMode) {
      onSelect(component.id);
    }
  }, [component.id, onSelect, isTestMode]);

  const handleParamChange = useCallback((val) => {
    if (onParamChange) onParamChange(component.id, val);
  }, [component.id, onParamChange]);

  const renderVisual = () => {
    // In test mode, use interactive versions for supported types
    if (isTestMode) {
      switch (component.type) {
        case 'knob':
          return <InteractiveKnob color={component.color} size={component.width} value={paramValue} onChange={handleParamChange} />;
        case 'slider':
          return <InteractiveSlider color={component.color} width={component.width} height={component.height} value={paramValue} onChange={handleParamChange} />;
        case 'button':
          return <ButtonVisual color={component.color} label={component.label} width={component.width} height={component.height} fontSize={component.fontSize} borderRadius={component.borderRadius} pressed={buttonPressed} />;
        case 'xy-pad':
          return <InteractiveXYPad color={component.color} width={component.width} height={component.height} value={paramValue} onChange={handleParamChange} />;
        case 'meter':
          return <MeterVisual color={component.color} width={component.width} height={component.height} level={paramValue} />;
        case 'led':
          return <LEDVisual color={component.color} size={component.width} lit={paramValue > 0.5} />;
        default:
          break; // fall through to static rendering
      }
    }

    switch (component.type) {
      case 'knob':
        return <KnobVisual color={component.color} size={component.width} />;
      case 'slider':
        return <SliderVisual color={component.color} width={component.width} height={component.height} />;
      case 'button':
        return <ButtonVisual color={component.color} label={component.label} width={component.width} height={component.height} fontSize={component.fontSize} borderRadius={component.borderRadius} />;
      case 'label':
        return <LabelVisual color={component.color} label={component.label} fontSize={component.fontSize} />;
      case 'led':
        return <LEDVisual color={component.color} size={component.width} />;
      case 'dropdown':
        return <DropdownVisual color={component.color} label={component.label} width={component.width} fontSize={component.fontSize} />;
      case 'image':
        return <ImageVisual width={component.width} height={component.height} image={component.image} borderRadius={component.borderRadius} />;
      case 'panel':
        return <PanelVisual width={component.width} height={component.height} borderColor={component.borderColor} bgColor={component.bgColor} borderRadius={component.borderRadius} bgGradient={component.bgGradient} backdropBlur={component.backdropBlur} boxShadow={component.boxShadow} />;
      case 'meter':
        return <MeterVisual color={component.color} width={component.width} height={component.height} />;
      case 'waveform':
        return <WaveformVisual color={component.color} width={component.width} height={component.height} />;
      case 'xy-pad':
        return <XYPadVisual color={component.color} width={component.width} height={component.height} />;
      default:
        return null;
    }
  };

  const showSubLabel = component.label && component.type !== 'label' && component.type !== 'button' && component.type !== 'panel';

  const wrapperStyle = {
    opacity: component.opacity ?? 1,
    transform: component.rotation ? `rotate(${component.rotation}deg)` : undefined,
    zIndex: component.zIndex ?? 1,
    boxShadow: component.type !== 'panel' ? (component.boxShadow || undefined) : undefined,
    position: isTestMode ? 'absolute' : undefined,
    left: isTestMode ? component.x : undefined,
    top: isTestMode ? component.y : undefined,
  };

  // In test mode, render without Draggable
  if (isTestMode) {
    return (
      <div
        ref={nodeRef}
        className={styles.canvasComp}
        style={wrapperStyle}
        onMouseDown={component.type === 'button' ? () => setButtonPressed(true) : undefined}
        onMouseUp={component.type === 'button' ? () => setButtonPressed(false) : undefined}
        onMouseLeave={component.type === 'button' ? () => setButtonPressed(false) : undefined}
      >
        {renderVisual()}
        {showSubLabel && (
          <div className={styles.compLabel}>
            {component.label}
            {['knob', 'slider'].includes(component.type) && (
              <span className={styles.paramValueLabel}> {Math.round(paramValue * 100)}%</span>
            )}
          </div>
        )}
      </div>
    );
  }

  // Edit mode — draggable with resize handles
  return (
    <Draggable
      nodeRef={nodeRef}
      position={{ x: component.x, y: component.y }}
      onDrag={handleDrag}
      onStop={handleDragStop}
      bounds="parent"
    >
      <div
        ref={nodeRef}
        className={`${styles.canvasComp} ${isSelected ? styles.canvasCompSelected : ''}`}
        onClick={handleClick}
        style={wrapperStyle}
      >
        {renderVisual()}
        {showSubLabel && <div className={styles.compLabel}>{component.label}</div>}
        {isSelected && <ResizeHandles component={component} onUpdate={onUpdate} onDragStop={onDragStop} />}
      </div>
    </Draggable>
  );
};

export default CanvasComponent;
