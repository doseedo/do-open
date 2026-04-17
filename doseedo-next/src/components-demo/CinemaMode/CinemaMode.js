import React, { useCallback, useRef, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import styles from './CinemaMode.module.css';

/**
 * CinemaMode - Fullscreen video mode with auto-hiding panels
 *
 * Features:
 * - Video centered in fullscreen
 * - Left panel (Generation) slides in on left edge hover
 * - Right panel (TrackInfo) slides in on right edge hover
 * - Bottom panel (DAW) slides up on bottom edge hover
 * - Video shifts 50% of panel width when panels reveal
 * - Track selection opens both DAW and right sidebar together
 */
const CinemaMode = ({
  videoContent,
  leftPanel,
  rightPanel,
  bottomPanel,
  leftPanelWidth = 304,
  rightPanelWidth = 320,
  bottomPanelHeight = 300
}) => {
  const { state, dispatch } = useApp();
  const hoverTimeoutRef = useRef({});
  const prevSelectedTrackRef = useRef(null);

  // When a track is selected, show both bottom (DAW) and right (TrackInfo) panels
  useEffect(() => {
    if (state.cinemaMode && state.selectedTrack && state.selectedTrack !== prevSelectedTrackRef.current) {
      // Clear any pending hide timeouts
      if (hoverTimeoutRef.current.bottom) clearTimeout(hoverTimeoutRef.current.bottom);
      if (hoverTimeoutRef.current.right) clearTimeout(hoverTimeoutRef.current.right);

      // Close left panel since right is opening (mutually exclusive)
      if (state.cinemaPanels.left) {
        dispatch({ type: 'SET_CINEMA_PANEL', payload: { panel: 'left', visible: false } });
      }

      // Show both panels
      dispatch({ type: 'SET_CINEMA_PANEL', payload: { panel: 'bottom', visible: true } });
      dispatch({ type: 'SET_CINEMA_PANEL', payload: { panel: 'right', visible: true } });
    }
    prevSelectedTrackRef.current = state.selectedTrack;
  }, [state.selectedTrack, state.cinemaMode, state.cinemaPanels.left, dispatch]);

  // Handle mouse enter on hover zones
  const handleZoneEnter = useCallback((panel) => {
    // Clear any pending hide timeout for this panel
    if (hoverTimeoutRef.current[panel]) {
      clearTimeout(hoverTimeoutRef.current[panel]);
    }

    // Left and right are mutually exclusive (but bottom stays independent)
    if (panel === 'left') {
      // Close right panel, but keep bottom open
      if (state.cinemaPanels.right) {
        dispatch({ type: 'SET_CINEMA_PANEL', payload: { panel: 'right', visible: false } });
      }
      // Cancel any pending bottom timeout so DAW stays open
      if (hoverTimeoutRef.current.bottom) {
        clearTimeout(hoverTimeoutRef.current.bottom);
      }
    } else if (panel === 'right') {
      // Close left panel, but keep bottom open
      if (state.cinemaPanels.left) {
        dispatch({ type: 'SET_CINEMA_PANEL', payload: { panel: 'left', visible: false } });
      }
      // Cancel any pending bottom timeout so DAW stays open
      if (hoverTimeoutRef.current.bottom) {
        clearTimeout(hoverTimeoutRef.current.bottom);
      }
    }

    dispatch({ type: 'SET_CINEMA_PANEL', payload: { panel, visible: true } });
  }, [dispatch, state.cinemaPanels.left, state.cinemaPanels.right]);

  // Handle mouse leave from panels (with delay)
  // Each panel hides independently after mouse leaves
  const handlePanelLeave = useCallback((panel) => {
    hoverTimeoutRef.current[panel] = setTimeout(() => {
      dispatch({ type: 'SET_CINEMA_PANEL', payload: { panel, visible: false } });
    }, 500);
  }, [dispatch]);

  // Handle mouse enter on panels (cancel hide)
  const handlePanelEnter = useCallback((panel) => {
    // Cancel any pending hide timeout for this panel
    if (hoverTimeoutRef.current[panel]) {
      clearTimeout(hoverTimeoutRef.current[panel]);
    }

    // Keep bottom (DAW) open when moving between panels
    // Cancel bottom timeout when entering any panel
    if (hoverTimeoutRef.current.bottom) {
      clearTimeout(hoverTimeoutRef.current.bottom);
    }
  }, []);

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      Object.values(hoverTimeoutRef.current).forEach(timeout => {
        if (timeout) clearTimeout(timeout);
      });
    };
  }, []);

  // Calculate video shrink - only 30% of panel width (panels overlap video by 70%)
  const leftShrink = state.cinemaPanels.left ? leftPanelWidth * 0.3 : 0;
  const rightShrink = state.cinemaPanels.right ? rightPanelWidth * 0.3 : 0;
  const bottomShrink = state.cinemaPanels.bottom ? bottomPanelHeight * 0.5 : 0;

  // ESC key to exit cinema mode
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && state.cinemaMode) {
        dispatch({ type: 'TOGGLE_CINEMA_MODE' });
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [state.cinemaMode, dispatch]);

  if (!state.cinemaMode) return null;

  return (
    <div className={styles.cinemaOverlay}>
      {/* Video Container - Shrinks by 50% of panel width, panels overlap by 50% */}
      <div
        className={styles.videoContainer}
        style={{
          width: `calc(100% - ${leftShrink}px - ${rightShrink}px)`,
          height: `calc(100% - ${bottomShrink}px)`,
          marginLeft: `${leftShrink}px`,
          marginTop: 0
        }}
      >
        {videoContent}
      </div>

      {/* Left Collapsed Mini-Sidebar */}
      <div
        className={`${styles.collapsedLeft} ${state.cinemaPanels.left ? styles.hidden : ''}`}
        onMouseEnter={() => handleZoneEnter('left')}
      >
        <div className={styles.collapsedIcon}><i className="fa-solid fa-bars"></i></div>
        <div className={styles.collapsedIcon}><i className="fa-solid fa-wand-magic-sparkles"></i></div>
        <div className={styles.collapsedIcon}><i className="fa-solid fa-bookmark"></i></div>
      </div>

      {/* Left Panel */}
      <div
        className={`${styles.leftPanel} ${state.cinemaPanels.left ? styles.visible : ''}`}
        style={{ width: `${leftPanelWidth}px` }}
        onMouseEnter={() => handlePanelEnter('left')}
        onMouseLeave={() => handlePanelLeave('left')}
      >
        <div className={styles.panelContent}>
          {leftPanel}
        </div>
      </div>

      {/* Right Collapsed Mini-Sidebar */}
      <div
        className={`${styles.collapsedRight} ${state.cinemaPanels.right ? styles.hidden : ''}`}
        onMouseEnter={() => handleZoneEnter('right')}
      >
        <div className={styles.collapsedIcon}><i className="fa-solid fa-sliders"></i></div>
        <div className={styles.collapsedIcon}><i className="fa-solid fa-wave-square"></i></div>
      </div>

      {/* Right Panel */}
      <div
        className={`${styles.rightPanel} ${state.cinemaPanels.right ? styles.visible : ''}`}
        style={{ width: `${rightPanelWidth}px` }}
        onMouseEnter={() => handlePanelEnter('right')}
        onMouseLeave={() => handlePanelLeave('right')}
      >
        <div className={styles.panelContent}>
          {rightPanel}
        </div>
      </div>

      {/* Bottom Collapsed Timeline Bar */}
      <div
        className={`${styles.collapsedBottom} ${state.cinemaPanels.bottom ? styles.hidden : ''}`}
        onMouseEnter={() => handleZoneEnter('bottom')}
      >
        <div className={styles.miniTimeline}>
          <div className={styles.miniPlayhead}></div>
          <div className={styles.miniTimeMarks}>
            <span>0:00</span>
            <span>0:30</span>
            <span>1:00</span>
            <span>1:30</span>
            <span>2:00</span>
          </div>
        </div>
      </div>

      {/* Bottom Panel */}
      <div
        className={`${styles.bottomPanel} ${state.cinemaPanels.bottom ? styles.visible : ''}`}
        style={{ height: `${bottomPanelHeight}px` }}
        onMouseEnter={() => handlePanelEnter('bottom')}
        onMouseLeave={() => handlePanelLeave('bottom')}
      >
        <div className={styles.panelContent}>
          {bottomPanel}
        </div>
      </div>

    </div>
  );
};

export default CinemaMode;
