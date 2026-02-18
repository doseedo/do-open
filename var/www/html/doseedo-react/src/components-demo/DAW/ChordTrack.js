import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { useApp } from '../../context/AppContext';

/**
 * ChordTrack Component with Form Sections
 *
 * Shows song structure: Intro (4 bars), Verse (8 bars), Chorus (8 bars), Outro (rest)
 * Hover reveals chords (normal mode) or automation shapes (automation mode)
 */
const ChordTrack = ({ totalDuration, zoomLevel, onBeatSelect }) => {
  const { state, dispatch } = useApp();
  const [hoveredSection, setHoveredSection] = useState(null);
  const chordTrackRef = useRef(null);

  const bpm = state.bpm || 120;
  const selectedSection = state.selectedFormSection || null;
  const automationMode = state.drumAutomationMode;
  const drumAutomation = state.drumAutomation || {};

  const containerWidth = state.timelineWidth || 700;
  const chordTrackContentWidth = containerWidth * zoomLevel;
  const secondsPerBeat = 60 / bpm;
  const secondsPerBar = secondsPerBeat * 4;

  // Form sections definition (in bars)
  // Intro: 4 bars, Verse: 8 bars, Chorus: 10 bars, Outro: rest
  const formSections = useMemo(() => {
    const totalBars = Math.ceil(totalDuration / secondsPerBar);
    return [
      { id: 'intro', name: 'INTRO', startBar: 0, endBar: Math.min(4, totalBars) },
      { id: 'verse', name: 'VERSE', startBar: 4, endBar: Math.min(12, totalBars) },
      { id: 'chorus', name: 'CHORUS', startBar: 12, endBar: Math.min(22, totalBars) },  // 10 bars
      { id: 'outro', name: 'OUTRO', startBar: 22, endBar: totalBars }
    ].filter(s => s.startBar < totalBars && s.endBar > s.startBar);
  }, [totalDuration, secondsPerBar]);

  // Convert bar positions to pixels
  const getSectionStyle = useCallback((section) => {
    const startTime = section.startBar * secondsPerBar;
    const endTime = section.endBar * secondsPerBar;
    const left = (startTime / totalDuration) * chordTrackContentWidth;
    const width = ((endTime - startTime) / totalDuration) * chordTrackContentWidth;
    return { left, width };
  }, [totalDuration, chordTrackContentWidth, secondsPerBar]);

  // Generate chord cells per bar
  const barCells = useMemo(() => {
    const cells = [];
    const width = chordTrackContentWidth;
    const totalBars = Math.ceil(totalDuration / secondsPerBar);

    for (let barNumber = 0; barNumber < totalBars; barNumber++) {
      const startTime = barNumber * secondsPerBar;
      const endTime = Math.min((barNumber + 1) * secondsPerBar, totalDuration);
      const position = (startTime / totalDuration) * width;
      const cellWidth = ((endTime - startTime) / totalDuration) * width;
      const beatIndex = barNumber * 4;
      const chord = state.chordTrack?.chords?.[beatIndex] || null;

      cells.push({
        id: `bar-${barNumber}`,
        barNumber,
        position,
        width: cellWidth,
        chord
      });
    }
    return cells;
  }, [chordTrackContentWidth, totalDuration, secondsPerBar, state.chordTrack]);

  // Check if a bar is in the hovered or selected section
  const isBarHighlighted = useCallback((barNumber) => {
    const activeSection = hoveredSection || selectedSection;
    if (!activeSection) return false;
    const section = formSections.find(s => s.id === activeSection);
    return section && barNumber >= section.startBar && barNumber < section.endBar;
  }, [hoveredSection, selectedSection, formSections]);

  // Get section for a bar
  const getSectionForBar = useCallback((barNumber) => {
    return formSections.find(s => barNumber >= s.startBar && barNumber < s.endBar);
  }, [formSections]);

  // Handle section click
  const handleSectionClick = useCallback((sectionId) => {
    if (selectedSection === sectionId) {
      dispatch({ type: 'SET_SELECTED_FORM_SECTION', payload: null });
    } else {
      dispatch({ type: 'SET_SELECTED_FORM_SECTION', payload: sectionId });
    }
  }, [selectedSection, dispatch]);

  // Handle automation shape click
  const handleShapeClick = useCallback((sectionId, shape) => {
    if (!automationMode?.parameter) return;
    dispatch({
      type: 'SET_DRUM_AUTOMATION_SHAPE',
      payload: {
        section: sectionId,
        parameter: automationMode.parameter,
        shape
      }
    });
  }, [automationMode, dispatch]);

  // Handle chord cell click - also selects the section
  const handleCellClick = useCallback((barNumber) => {
    const beatNumber = barNumber * 4 + 1;
    onBeatSelect?.(beatNumber);

    // Also select the section containing this bar
    const section = formSections.find(s => barNumber >= s.startBar && barNumber < s.endBar);
    if (section && !automationMode?.enabled) {
      if (selectedSection === section.id) {
        dispatch({ type: 'SET_SELECTED_FORM_SECTION', payload: null });
      } else {
        dispatch({ type: 'SET_SELECTED_FORM_SECTION', payload: section.id });
      }
    }
  }, [onBeatSelect, formSections, automationMode, selectedSection, dispatch]);

  // Get automation shape for a section
  const getAutomationShape = useCallback((sectionId) => {
    if (!automationMode?.parameter) return null;
    return drumAutomation[sectionId]?.[automationMode.parameter] || null;
  }, [automationMode, drumAutomation]);

  // Force width using ref
  useEffect(() => {
    if (chordTrackRef.current) {
      chordTrackRef.current.style.width = `${chordTrackContentWidth}px`;
      chordTrackRef.current.style.minWidth = `${chordTrackContentWidth}px`;
    }
  }, [chordTrackContentWidth]);

  const activeSection = hoveredSection || selectedSection;
  const isAutomationMode = automationMode?.enabled;

  // Render automation shape SVG - always white
  const renderAutomationShape = (shape, width, height) => {
    if (!shape) return null;

    const paths = {
      'ramp-up': `M 0 ${height} L ${width} 0`,
      'ramp-down': `M 0 0 L ${width} ${height}`,
      'flat': `M 0 ${height / 2} L ${width} ${height / 2}`
    };

    return (
      <svg
        width={width}
        height={height}
        style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
      >
        <path
          d={paths[shape]}
          stroke="rgba(255, 255, 255, 0.6)"
          strokeWidth="2"
          fill="none"
        />
      </svg>
    );
  };

  return (
    <div
      ref={chordTrackRef}
      style={{
        position: 'absolute',
        left: 0,
        top: 0,
        width: `${chordTrackContentWidth}px`,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'visible'
      }}
    >
      {/* Form Sections Row - Top */}
      <div style={{
        position: 'relative',
        height: '50%',
        width: '100%'
      }}>
        {formSections.map((section) => {
          const style = getSectionStyle(section);
          const isHovered = hoveredSection === section.id;
          const isSelected = selectedSection === section.id;
          const isActive = isHovered || isSelected;
          const currentShape = getAutomationShape(section.id);

          return (
            <div
              key={section.id}
              onMouseEnter={() => setHoveredSection(section.id)}
              onMouseLeave={() => setHoveredSection(null)}
              onClick={() => !isAutomationMode && handleSectionClick(section.id)}
              style={{
                position: 'absolute',
                left: `${style.left}px`,
                top: 0,
                width: `${style.width}px`,
                height: '100%',
                background: isActive ? 'rgba(255, 255, 255, 0.08)' : 'transparent',
                borderLeft: `1px solid ${isActive ? 'rgba(255, 255, 255, 0.4)' : 'rgba(255, 255, 255, 0.1)'}`,
                borderRight: '1px solid rgba(255, 255, 255, 0.05)',
                borderTop: 'none',
                borderBottom: `1px solid ${isActive ? 'rgba(255, 255, 255, 0.2)' : 'rgba(255, 255, 255, 0.05)'}`,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '4px',
                transition: 'background 0.1s ease',
                boxSizing: 'border-box'
              }}
            >
              {/* Automation Shape Buttons - show when in automation mode and hovered */}
              {isAutomationMode && isHovered ? (
                <div style={{ display: 'flex', gap: '3px' }}>
                  {[
                    { id: 'ramp-up', icon: 'fa-arrow-trend-up' },
                    { id: 'ramp-down', icon: 'fa-arrow-trend-down' },
                    { id: 'flat', icon: 'fa-minus' }
                  ].map(shape => (
                    <button
                      key={shape.id}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleShapeClick(section.id, shape.id);
                      }}
                      style={{
                        width: '20px',
                        height: '16px',
                        border: currentShape === shape.id ? '1px solid rgba(255,255,255,0.6)' : '1px solid rgba(255,255,255,0.2)',
                        borderRadius: '3px',
                        background: currentShape === shape.id ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.3)',
                        color: currentShape === shape.id ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.5)',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '8px',
                        padding: 0
                      }}
                    >
                      <i className={`fa-solid ${shape.icon}`} />
                    </button>
                  ))}
                </div>
              ) : (
                <>
                  <span style={{
                    fontSize: '9px',
                    fontWeight: 600,
                    color: isActive ? 'rgba(255, 255, 255, 0.9)' : 'rgba(255, 255, 255, 0.4)',
                    letterSpacing: '0.5px',
                    transition: 'color 0.1s ease'
                  }}>
                    {section.name}
                  </span>
                  {isSelected && !isAutomationMode && (
                    <i
                      className="fa-solid fa-check"
                      style={{
                        fontSize: '7px',
                        color: 'rgba(255, 255, 255, 0.8)',
                        opacity: 0.8
                      }}
                    />
                  )}
                </>
              )}
            </div>
          );
        })}
      </div>

      {/* Chord/Automation Cells Row - Bottom */}
      <div style={{
        position: 'relative',
        height: '50%',
        width: '100%'
      }}>
        {/* Render automation shapes for each section in automation mode */}
        {isAutomationMode && formSections.map((section) => {
          const style = getSectionStyle(section);
          const shape = getAutomationShape(section.id);
          const isActive = hoveredSection === section.id || selectedSection === section.id;

          if (!shape && !isActive) return null;

          return (
            <div
              key={`auto-${section.id}`}
              style={{
                position: 'absolute',
                left: `${style.left}px`,
                top: 0,
                width: `${style.width}px`,
                height: '100%',
                pointerEvents: 'none',
                overflow: 'hidden'
              }}
            >
              {shape && renderAutomationShape(shape, style.width, 20)}
            </div>
          );
        })}

        {/* Bar cells */}
        {barCells.map((cell) => {
          const highlighted = isBarHighlighted(cell.barNumber);
          const cellSection = getSectionForBar(cell.barNumber);

          return (
            <div
              key={cell.id}
              onClick={() => handleCellClick(cell.barNumber)}
              onMouseEnter={() => cellSection && setHoveredSection(cellSection.id)}
              onMouseLeave={() => setHoveredSection(null)}
              style={{
                position: 'absolute',
                left: `${cell.position}px`,
                top: 0,
                width: `${cell.width}px`,
                height: '100%',
                borderLeft: `1px solid ${highlighted ? 'rgba(255, 255, 255, 0.3)' : 'rgba(255, 255, 255, 0.1)'}`,
                borderRight: '1px solid rgba(255, 255, 255, 0.03)',
                borderTop: 'none',
                borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                backgroundColor: highlighted ? 'rgba(255, 255, 255, 0.05)' : 'transparent',
                transition: 'background 0.1s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                boxSizing: 'border-box'
              }}
            >
              {/* Chord label - only show when section is active and NOT in automation mode */}
              {!isAutomationMode && cell.chord && activeSection && (
                <div style={{
                  fontSize: '9px',
                  fontWeight: 500,
                  color: highlighted ? 'rgba(255, 255, 255, 0.7)' : 'rgba(255, 255, 255, 0.35)',
                  whiteSpace: 'nowrap',
                  opacity: highlighted ? 0.9 : 0.5,
                  transition: 'opacity 0.1s ease'
                }}>
                  {cell.chord}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ChordTrack;
