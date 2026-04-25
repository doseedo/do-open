import React, { useRef, useCallback, useState, useEffect } from 'react';
import CanvasComponent from './CanvasComponent';
import WebAudioDSPEngine from '../../../audio/WebAudioDSPEngine';
import styles from './PluginCreator.module.css';

const guideLineStyle = {
  position: 'absolute',
  pointerEvents: 'none',
  zIndex: 9998,
};

// Piano keyboard constants — 2 octaves + 1 (C3 to C5)
const KEYBOARD_NOTES = [];
for (let oct = 0; oct < 2; oct++) {
  const base = 48 + oct * 12; // C3=48, C4=60
  KEYBOARD_NOTES.push(
    { note: base,     black: false, label: `C${oct + 3}`,  ariaLabel: `C${oct + 3}` },
    { note: base + 1, black: true,  label: 'C#',           ariaLabel: `C#${oct + 3}` },
    { note: base + 2, black: false, label: `D${oct + 3}`,  ariaLabel: `D${oct + 3}` },
    { note: base + 3, black: true,  label: 'D#',           ariaLabel: `D#${oct + 3}` },
    { note: base + 4, black: false, label: `E${oct + 3}`,  ariaLabel: `E${oct + 3}` },
    { note: base + 5, black: false, label: `F${oct + 3}`,  ariaLabel: `F${oct + 3}` },
    { note: base + 6, black: true,  label: 'F#',           ariaLabel: `F#${oct + 3}` },
    { note: base + 7, black: false, label: `G${oct + 3}`,  ariaLabel: `G${oct + 3}` },
    { note: base + 8, black: true,  label: 'G#',           ariaLabel: `G#${oct + 3}` },
    { note: base + 9, black: false, label: `A${oct + 3}`,  ariaLabel: `A${oct + 3}` },
    { note: base + 10, black: true, label: 'A#',           ariaLabel: `A#${oct + 3}` },
    { note: base + 11, black: false, label: `B${oct + 3}`, ariaLabel: `B${oct + 3}` },
  );
}
KEYBOARD_NOTES.push({ note: 72, black: false, label: 'C5', ariaLabel: 'C5' }); // final C

const PluginCanvas = ({ config, components, selectedIds, onSelect, onDeselect, onUpdateComponent, onDragStop, snapToGrid, gridSize, editorMode, paramValues, onParamChange, smartGuides, onContextMenu, rubberBand, onRubberBandChange, onRubberBandSelect, overlay, reverseParamMap, engine, frameRef, generatingImages }) => {
  const canvasRef = useRef(null);
  const [rbStart, setRbStart] = useState(null);
  const [activeNotes, setActiveNotes] = useState(new Set());
  const [activeTab, setActiveTab] = useState(0);
  const voiceMapRef = useRef({});
  const mouseHeldRef = useRef(false);
  const currentNoteRef = useRef(null);
  const pianoContainerRef = useRef(null);

  const isInstrument = editorMode === 'test' && engine?.isInstrument;

  // Detect if this is a tabbed layout (any component has tabIndex defined)
  const hasTabLayout = components.some(c => c.tabIndex !== undefined);
  // Reset active tab when components change significantly (new layout applied)
  const tabCountRef = useRef(0);
  const currentTabCount = hasTabLayout ? components.filter(c => c.isTabButton).length : 0;
  useEffect(() => {
    if (currentTabCount !== tabCountRef.current) {
      tabCountRef.current = currentTabCount;
      setActiveTab(0);
    }
  }, [currentTabCount]);

  // Computer keyboard → piano notes
  useEffect(() => {
    if (!isInstrument || !engine) return;
    const kbMap = WebAudioDSPEngine.KEYBOARD_MAP;

    const onKeyDown = (e) => {
      if (e.repeat) return;
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      const note = kbMap[e.key.toLowerCase()];
      // R6: noteOn no longer returns a voiceId — key tracking is by the
      // MIDI note number directly. We still index voiceMapRef by keyboard
      // key so a held QWERTY key only triggers once.
      if (note == null || voiceMapRef.current[e.key] != null) return;
      engine._ensureContext();
      if (!engine.masterGain) engine._buildGraph();
      engine.noteOn(note, 0.8);
      voiceMapRef.current[e.key] = note;
      setActiveNotes(prev => new Set([...prev, note]));
    };

    const onKeyUp = (e) => {
      const note = voiceMapRef.current[e.key];
      if (note != null) {
        // R6: noteOff is keyed on the MIDI note number, not a voiceId.
        engine.noteOff(note);
        delete voiceMapRef.current[e.key];
        setActiveNotes(prev => { const s = new Set(prev); s.delete(note); return s; });
      }
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      for (const note of Object.values(voiceMapRef.current)) engine.noteOff(note);
      voiceMapRef.current = {};
    };
  }, [isInstrument, engine]);

  const handleNoteOn = useCallback((note) => {
    if (!engine) return;
    engine._ensureContext();
    if (!engine.masterGain) engine._buildGraph();
    // R6: ignore the voiceId return; noteOff takes the MIDI note number.
    engine.noteOn(note, 0.8);
    voiceMapRef.current['m' + note] = note;
    setActiveNotes(prev => new Set([...prev, note]));
  }, [engine]);

  const handleNoteOff = useCallback((note) => {
    const tracked = voiceMapRef.current['m' + note];
    if (tracked != null && engine) {
      // R6: noteOff is keyed on the MIDI note number.
      engine.noteOff(tracked);
      delete voiceMapRef.current['m' + note];
    }
    setActiveNotes(prev => { const s = new Set(prev); s.delete(note); return s; });
  }, [engine]);

  // --- Glissando support: mouse drag across piano keys ---
  const handlePianoKeyMouseDown = useCallback((note) => {
    mouseHeldRef.current = true;
    currentNoteRef.current = note;
    handleNoteOn(note);
  }, [handleNoteOn]);

  const handlePianoKeyMouseEnter = useCallback((note) => {
    if (!mouseHeldRef.current) return;
    // Stop previous note if different
    if (currentNoteRef.current != null && currentNoteRef.current !== note) {
      handleNoteOff(currentNoteRef.current);
    }
    currentNoteRef.current = note;
    handleNoteOn(note);
  }, [handleNoteOn, handleNoteOff]);

  const handlePianoMouseUp = useCallback(() => {
    if (mouseHeldRef.current && currentNoteRef.current != null) {
      handleNoteOff(currentNoteRef.current);
    }
    mouseHeldRef.current = false;
    currentNoteRef.current = null;
  }, [handleNoteOff]);

  // Window-level mouseup to catch release outside the piano
  useEffect(() => {
    if (!isInstrument) return;
    const onGlobalMouseUp = () => {
      if (mouseHeldRef.current && currentNoteRef.current != null) {
        handleNoteOff(currentNoteRef.current);
      }
      mouseHeldRef.current = false;
      currentNoteRef.current = null;
    };
    window.addEventListener('mouseup', onGlobalMouseUp);
    return () => window.removeEventListener('mouseup', onGlobalMouseUp);
  }, [isInstrument, handleNoteOff]);

  // --- Touch glissando: slide finger across piano keys ---
  const handlePianoKeyTouchStart = useCallback((e, note) => {
    e.preventDefault();
    mouseHeldRef.current = true;
    currentNoteRef.current = note;
    handleNoteOn(note);
  }, [handleNoteOn]);

  useEffect(() => {
    if (!isInstrument) return;
    const container = pianoContainerRef.current;
    if (!container) return;

    const onTouchMove = (e) => {
      if (!mouseHeldRef.current) return;
      e.preventDefault();
      const touch = e.touches[0];
      if (!touch) return;
      const el = document.elementFromPoint(touch.clientX, touch.clientY);
      if (!el) return;
      // Find the key element with data-note attribute
      const keyEl = el.closest('[data-note]');
      if (!keyEl) return;
      const note = parseInt(keyEl.getAttribute('data-note'), 10);
      if (isNaN(note)) return;
      if (note !== currentNoteRef.current) {
        if (currentNoteRef.current != null) {
          handleNoteOff(currentNoteRef.current);
        }
        currentNoteRef.current = note;
        handleNoteOn(note);
      }
    };

    const onTouchEnd = () => {
      if (mouseHeldRef.current && currentNoteRef.current != null) {
        handleNoteOff(currentNoteRef.current);
      }
      mouseHeldRef.current = false;
      currentNoteRef.current = null;
    };

    container.addEventListener('touchmove', onTouchMove, { passive: false });
    container.addEventListener('touchend', onTouchEnd);
    container.addEventListener('touchcancel', onTouchEnd);
    return () => {
      container.removeEventListener('touchmove', onTouchMove);
      container.removeEventListener('touchend', onTouchEnd);
      container.removeEventListener('touchcancel', onTouchEnd);
    };
  }, [isInstrument, handleNoteOn, handleNoteOff]);

  const handleCanvasMouseDown = useCallback((e) => {
    // Only on direct canvas clicks (not on components), left button only
    if (e.target !== canvasRef.current || e.button !== 0) return;
    if (editorMode !== 'edit') return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setRbStart({ x, y });
    if (onRubberBandChange) onRubberBandChange({ startX: x, startY: y, currentX: x, currentY: y });
  }, [editorMode, onRubberBandChange]);

  const handleCanvasMouseMove = useCallback((e) => {
    if (!rbStart || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(config.width, e.clientX - rect.left));
    const y = Math.max(0, Math.min(config.height, e.clientY - rect.top));
    if (onRubberBandChange) onRubberBandChange({ startX: rbStart.x, startY: rbStart.y, currentX: x, currentY: y });
  }, [rbStart, config.width, config.height, onRubberBandChange]);

  const handleCanvasMouseUp = useCallback(() => {
    if (!rbStart || !rubberBand) {
      setRbStart(null);
      return;
    }

    // Calculate selection rectangle
    const left = Math.min(rubberBand.startX, rubberBand.currentX);
    const right = Math.max(rubberBand.startX, rubberBand.currentX);
    const top = Math.min(rubberBand.startY, rubberBand.currentY);
    const bottom = Math.max(rubberBand.startY, rubberBand.currentY);

    // Only select if the rectangle is at least 5px in some dimension (not a click)
    if (right - left > 5 || bottom - top > 5) {
      const selected = components
        .filter(c => {
          const cx = c.x + c.width / 2;
          const cy = c.y + c.height / 2;
          return cx >= left && cx <= right && cy >= top && cy <= bottom;
        })
        .map(c => c.id);
      if (onRubberBandSelect && selected.length > 0) {
        onRubberBandSelect(selected);
      }
    }

    setRbStart(null);
    if (onRubberBandChange) onRubberBandChange(null);
  }, [rbStart, rubberBand, components, onRubberBandSelect, onRubberBandChange]);

  const handleCanvasClick = useCallback((e) => {
    if (e.target === canvasRef.current && !rbStart) {
      onDeselect();
    }
  }, [onDeselect, rbStart]);

  const handleCanvasContextMenu = useCallback((e) => {
    // Only fire for empty canvas clicks (not on components)
    if (e.target === canvasRef.current && onContextMenu) {
      onContextMenu(e, null);
    }
  }, [onContextMenu]);

  // Touch handlers for rubber-band selection
  const handleTouchStart = useCallback((e) => {
    if (e.target !== canvasRef.current || e.touches.length !== 1) return;
    if (editorMode !== 'edit') return;
    const touch = e.touches[0];
    const rect = canvasRef.current.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    setRbStart({ x, y });
    if (onRubberBandChange) onRubberBandChange({ startX: x, startY: y, currentX: x, currentY: y });
  }, [editorMode, onRubberBandChange]);

  const handleTouchMove = useCallback((e) => {
    if (!rbStart || !canvasRef.current || e.touches.length !== 1) return;
    const touch = e.touches[0];
    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(config.width, touch.clientX - rect.left));
    const y = Math.max(0, Math.min(config.height, touch.clientY - rect.top));
    if (onRubberBandChange) onRubberBandChange({ startX: rbStart.x, startY: rbStart.y, currentX: x, currentY: y });
  }, [rbStart, config.width, config.height, onRubberBandChange]);

  const handleTouchEnd = useCallback(() => {
    handleCanvasMouseUp();
  }, [handleCanvasMouseUp]);

  // ── Group hover state for slider-group reveal ──
  const [hoveredGroupId, setHoveredGroupId] = useState(null);

  // Filter components by active tab and group visibility, then sort by zIndex
  const visibleComponents = (hasTabLayout
    ? components.filter(c => {
        if (c.isTabButton) return true;
        if (c.tabIndex === -1) return true;
        if (c.tabIndex === activeTab) return true;
        return false;
      }).map(c => {
        if (c.isTabButton) {
          return { ...c, opacity: c.tabTargetIndex === activeTab ? 1 : 0.5 };
        }
        if (c.tabIndex === activeTab) {
          return { ...c, opacity: c.opacity === 0 ? 1 : c.opacity };
        }
        return c;
      })
    : components
  ).map(c => {
    // Group hover — compute visibility for smooth fade transitions
    if (c.groupHidden) {
      const visible = hoveredGroupId ? c.groupId === hoveredGroupId : !!c.groupDefault;
      return { ...c, _groupVisible: visible };
    }
    return c;
  });
  const sorted = [...visibleComponents].sort((a, b) => (a.zIndex || 0) - (b.zIndex || 0));

  // Rubber band rect
  const rbRect = rubberBand ? {
    left: Math.min(rubberBand.startX, rubberBand.currentX),
    top: Math.min(rubberBand.startY, rubberBand.currentY),
    width: Math.abs(rubberBand.currentX - rubberBand.startX),
    height: Math.abs(rubberBand.currentY - rubberBand.startY),
  } : null;

  return (
    <div className={styles.pluginFrame} ref={frameRef}>
      {/* macOS-style title bar */}
      <div className={styles.titleBar} style={{ background: config.titleBarColor }}>
        <div className={styles.titleBarDots}>
          <span className={styles.dot} style={{ background: '#ff5f57' }} />
          <span className={styles.dot} style={{ background: '#febc2e' }} />
          <span className={styles.dot} style={{ background: '#28c840' }} />
        </div>
        <span className={styles.titleBarText}>{config.name}</span>
        {editorMode === 'test' && (
          <span className={styles.testModeBadge}>TEST MODE</span>
        )}
      </div>
      {/* Canvas body */}
      <div
        ref={canvasRef}
        className={styles.canvasBody}
        style={{
          width: config.width,
          height: config.height,
          background: config.bgImage
            ? `url(${config.bgImage}) center/cover no-repeat, ${config.bgColor}`
            : config.bgColor,
          position: 'relative',
          overflow: 'hidden',
        }}
        onClick={handleCanvasClick}
        onContextMenu={handleCanvasContextMenu}
        onMouseDown={handleCanvasMouseDown}
        onMouseMove={handleCanvasMouseMove}
        onMouseUp={handleCanvasMouseUp}
        onMouseLeave={handleCanvasMouseUp}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        {/* Grid overlay (edit mode only) */}
        {editorMode === 'edit' && snapToGrid && gridSize && (
          <div className={styles.gridOverlay} style={{
            width: '100%', height: '100%',
            backgroundSize: `${gridSize}px ${gridSize}px`,
            backgroundImage: 'linear-gradient(to right, rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.04) 1px, transparent 1px)',
            position: 'absolute', top: 0, left: 0, pointerEvents: 'none', zIndex: 0,
          }} />
        )}

        {/* Smart guide lines */}
        {smartGuides && smartGuides.map((guide, i) => (
          guide.type === 'vertical' ? (
            <div key={`guide-${i}`} style={{
              ...guideLineStyle,
              left: guide.position,
              top: 0,
              width: 1,
              height: '100%',
              borderLeft: '1px dashed rgba(0, 200, 255, 0.7)',
            }} />
          ) : (
            <div key={`guide-${i}`} style={{
              ...guideLineStyle,
              top: guide.position,
              left: 0,
              height: 1,
              width: '100%',
              borderTop: '1px dashed rgba(0, 200, 255, 0.7)',
            }} />
          )
        ))}

        {/* Rubber band selection rectangle */}
        {rbRect && rbRect.width > 2 && rbRect.height > 2 && (
          <div style={{
            position: 'absolute',
            left: rbRect.left,
            top: rbRect.top,
            width: rbRect.width,
            height: rbRect.height,
            border: '1px dashed rgba(186, 156, 255, 0.7)',
            background: 'rgba(186, 156, 255, 0.08)',
            pointerEvents: 'none',
            zIndex: 9999,
          }} />
        )}

        {sorted.map(comp => (
          comp.isTabButton ? (
            <div
              key={comp.id}
              onClick={(e) => { e.stopPropagation(); setActiveTab(comp.tabTargetIndex); }}
              style={{
                position: 'absolute',
                left: comp.x, top: comp.y,
                width: comp.width, height: comp.height,
                background: comp.tabTargetIndex === activeTab
                  ? (comp.color || '#667eea')
                  : 'rgba(255,255,255,0.08)',
                color: comp.tabTargetIndex === activeTab ? '#fff' : 'rgba(255,255,255,0.5)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: comp.fontSize || 11, fontWeight: 600,
                letterSpacing: '0.05em',
                borderRadius: comp.borderRadius || 4,
                cursor: 'pointer',
                zIndex: comp.zIndex || 5,
                transition: 'background 0.15s, color 0.15s',
                userSelect: 'none',
              }}
            >
              {comp.label}
            </div>
          ) : (
            <CanvasComponent
              key={comp.id}
              component={comp}
              isSelected={selectedIds.includes(comp.id)}
              onSelect={onSelect}
              onUpdate={onUpdateComponent}
              onDragStop={onDragStop}
              canvasWidth={config.width}
              canvasHeight={config.height}
              editorMode={editorMode}
              paramValue={paramValues?.[comp.id] ?? (comp.type === 'dropdown' ? 0 : 0.5)}
              onParamChange={onParamChange}
              onContextMenu={onContextMenu}
              allComponents={components}
              allParamValues={paramValues}
              boundParamId={reverseParamMap?.[comp.id] || null}
              isGenerating={generatingImages?.has?.(comp.id) || false}
              onGroupHover={setHoveredGroupId}
              groupVisible={comp._groupVisible}
              engine={engine}
            />
          )
        ))}

        {/* Structure mode overlay — renders on top of canvas content */}
        {overlay}
      </div>

      {/* Piano keyboard — instrument test mode */}
      {isInstrument && (() => {
        const whiteKeys = KEYBOARD_NOTES.filter(k => !k.black);
        const totalWhite = whiteKeys.length;
        const kbW = config.width;
        const whiteW = Math.floor(kbW / totalWhite);
        const blackW = Math.floor(whiteW * 0.6);
        const whiteH = 52;
        const blackH = 32;

        // Map computer keys for labels
        const kbMap = WebAudioDSPEngine.KEYBOARD_MAP;
        const noteToKey = {};
        for (const [k, n] of Object.entries(kbMap)) noteToKey[n] = k.toUpperCase();

        let whiteIdx = 0;
        const keyElements = [];

        for (const nk of KEYBOARD_NOTES) {
          const isActive = activeNotes.has(nk.note);
          if (!nk.black) {
            const x = whiteIdx * whiteW;
            keyElements.push(
              <div
                key={nk.note}
                data-note={nk.note}
                aria-label={nk.ariaLabel}
                role="button"
                onMouseDown={() => handlePianoKeyMouseDown(nk.note)}
                onMouseEnter={() => handlePianoKeyMouseEnter(nk.note)}
                onMouseUp={handlePianoMouseUp}
                onTouchStart={(e) => handlePianoKeyTouchStart(e, nk.note)}
                style={{
                  position: 'absolute', left: x, top: 0,
                  width: whiteW - 1, height: whiteH,
                  background: isActive
                    ? 'linear-gradient(180deg, #667eea 0%, #764ba2 100%)'
                    : 'linear-gradient(180deg, #f5f5f5 0%, #d8d8d8 100%)',
                  borderRadius: '0 0 4px 4px',
                  border: '1px solid rgba(0,0,0,0.15)',
                  borderTop: 'none',
                  cursor: 'pointer',
                  display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'flex-end',
                  paddingBottom: 3, userSelect: 'none',
                  zIndex: 1, boxSizing: 'border-box',
                  transition: 'background 0.05s',
                }}
              >
                {noteToKey[nk.note] && (
                  <span style={{ fontSize: 8, color: isActive ? '#fff' : 'rgba(0,0,0,0.25)', lineHeight: 1 }}>
                    {noteToKey[nk.note]}
                  </span>
                )}
                <span style={{ fontSize: 7, color: isActive ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.15)', lineHeight: 1 }}>
                  {nk.label}
                </span>
              </div>
            );
            whiteIdx++;
          } else {
            // Black key positioned between previous and next white keys
            const x = whiteIdx * whiteW - Math.floor(blackW / 2);
            keyElements.push(
              <div
                key={nk.note}
                data-note={nk.note}
                aria-label={nk.ariaLabel}
                role="button"
                onMouseDown={(e) => { e.stopPropagation(); handlePianoKeyMouseDown(nk.note); }}
                onMouseEnter={() => handlePianoKeyMouseEnter(nk.note)}
                onMouseUp={handlePianoMouseUp}
                onTouchStart={(e) => { e.stopPropagation(); handlePianoKeyTouchStart(e, nk.note); }}
                style={{
                  position: 'absolute', left: x, top: 0,
                  width: blackW, height: blackH,
                  background: isActive
                    ? 'linear-gradient(180deg, #667eea 0%, #4a3f8a 100%)'
                    : 'linear-gradient(180deg, #333 0%, #111 100%)',
                  borderRadius: '0 0 3px 3px',
                  border: '1px solid #000',
                  borderTop: 'none',
                  cursor: 'pointer',
                  display: 'flex', alignItems: 'flex-end', justifyContent: 'center',
                  paddingBottom: 2, userSelect: 'none',
                  zIndex: 2, boxSizing: 'border-box',
                  transition: 'background 0.05s',
                }}
              >
                {noteToKey[nk.note] && (
                  <span style={{ fontSize: 7, color: isActive ? '#fff' : 'rgba(255,255,255,0.2)' }}>
                    {noteToKey[nk.note]}
                  </span>
                )}
              </div>
            );
          }
        }

        return (
          <div ref={pianoContainerRef} style={{
            position: 'relative',
            width: kbW,
            height: whiteH,
            background: 'linear-gradient(180deg, #1a1a1a 0%, #252525 100%)',
            borderTop: '2px solid rgba(0,0,0,0.4)',
            borderRadius: '0 0 6px 6px',
            overflow: 'hidden',
            touchAction: 'none',
          }}>
            {keyElements}
          </div>
        );
      })()}
    </div>
  );
};

export default PluginCanvas;
