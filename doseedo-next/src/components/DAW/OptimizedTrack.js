import React, { useCallback, useRef, useState, useMemo, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { useTimeline } from '../../hooks/useTimeline';
import { useWaveform } from '../../hooks/useWaveform';
import { useThemeColor } from '../../hooks/useThemeColor';
import MIDITrackVisualization from './MIDITrackVisualization';
import PlaceholderWaveform from './PlaceholderWaveform';
import styles from './DAW.module.css';

/**
 * OptimizedTrack - Track component using GPU-accelerated transforms
 *
 * Key Optimizations:
 * 1. Uses CSS transforms instead of top/left (GPU accelerated)
 * 2. Drag handled with translate3d for 60fps performance
 * 3. Resize uses CSS custom properties for efficient updates
 * 4. No react-draggable dependency (native implementation is lighter)
 */
const OptimizedTrack = React.memo(({ track, busId, index, isExpanded, isSelected, isMasterView = false, trackHeight = 72 }) => {
  const { state, dispatch } = useApp();
  const trackRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(null); // null | 'left' | 'right'
  const dragStartRef = useRef({ x: 0, y: 0, startPosition: 0, startCrop: 0, startWidth: 0 });
  const dragDirectionRef = useRef(null); // null | 'x' | 'y'

  // Check if this track is in multi-select
  const isMultiSelected = useMemo(() => {
    return state.selectedTracks?.some(t => t.id === track.id) || false;
  }, [state.selectedTracks, track.id]);

  // Inpaint selection state
  const [inpaintSelection, setInpaintSelection] = useState(null); // { startX, width }
  const inpaintSelectionRef = useRef({ isSelecting: false, startX: 0 });

  // Check if this track is in inpaint mode
  const isInInpaintMode = state.inpaintMode?.enabled && state.inpaintMode?.trackId === track.id;

  // Resolve relative audio URL to full download URL for plugin bridge
  const getFullAudioUrl = useCallback((audioUrl) => {
    if (!audioUrl) return null;
    if (audioUrl.startsWith('http') || audioUrl.startsWith('blob:')) return audioUrl;
    return `${window.location.origin}${audioUrl}`;
  }, []);

  // Helper to communicate with JUCE plugin host for drag-to-DAW
  // Uses dodrag:// URL scheme that the JUCE plugin intercepts to initiate native OS drag
  const initiatePluginDrag = useCallback(() => {
    if (!track.audioUrl) return;

    const fullUrl = getFullAudioUrl(track.audioUrl);
    const fileName = `${track.name || 'audio'}.wav`;

    // 1. dodrag:// URL scheme — JUCE plugin intercepts this navigation,
    //    downloads the file to temp, and calls performExternalDragDropOfFiles()
    const encodedUrl = encodeURIComponent(fullUrl);
    console.log('🎵 Initiating plugin drag via dodrag://', fileName, fullUrl);
    window.location.href = 'dodrag://' + encodedUrl;

    // Also try additional bridges as fallback (JUCE may support these too)
    if (window.juce && window.juce.startDragFile) {
      window.juce.startDragFile(fullUrl, fileName);
    } else if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.dragFile) {
      window.webkit.messageHandlers.dragFile.postMessage({
        url: fullUrl,
        fileName: fileName,
        trackId: track.id
      });
    } else if (window.chrome && window.chrome.webview) {
      window.chrome.webview.postMessage({
        type: 'dragFile',
        url: fullUrl,
        fileName: fileName,
        trackId: track.id
      });
    }
  }, [track.audioUrl, track.name, track.id, getFullAudioUrl]);

  // Snap to beat/sub-beat helper function (only in BPM mode)
  const snapToGrid = useCallback((timeInSeconds) => {
    if (!state.isBPMMode) return timeInSeconds; // No snap in seconds mode

    const bpm = state.bpm || 120;
    const subdivisionLevel = state.subdivisionLevel || 1;
    const sceneTempos = state.video?.sceneTempos || [];
    const sceneChanges = state.video?.sceneChanges || [];

    // Use scene tempos if available, otherwise constant BPM
    const useSceneTempos = sceneTempos.length > 0 && sceneChanges.length > 1;

    if (useSceneTempos) {
      // Find which scene we're in
      let sceneBPM = bpm;
      for (let i = 0; i < sceneChanges.length; i++) {
        if (timeInSeconds >= sceneChanges[i] && (i === sceneChanges.length - 1 || timeInSeconds < sceneChanges[i + 1])) {
          sceneBPM = sceneTempos[i];
          break;
        }
      }

      const secondsPerBeat = 60 / sceneBPM;
      const secondsPerSubdivision = secondsPerBeat / subdivisionLevel;

      // Snap to nearest subdivision
      const subdivisionIndex = Math.round(timeInSeconds / secondsPerSubdivision);
      return subdivisionIndex * secondsPerSubdivision;
    } else {
      // Constant BPM
      const secondsPerBeat = 60 / bpm;
      const secondsPerSubdivision = secondsPerBeat / subdivisionLevel;

      // Snap to nearest subdivision
      const subdivisionIndex = Math.round(timeInSeconds / secondsPerSubdivision);
      return subdivisionIndex * secondsPerSubdivision;
    }
  }, [state.isBPMMode, state.bpm, state.subdivisionLevel, state.video]);

  // Get timeline calculations - use actual timeline width from state
  const { pixelsPerSecond } = useTimeline(
    state.totalDuration || 10,
    state.zoomLevel || 1.0,
    state.timelineWidth || 950
  );

  // Calculate visual selection from global state time values
  const confirmedSelection = useMemo(() => {
    if (!state.inpaintSelection || state.inpaintSelection.trackId !== track.id) {
      return null;
    }

    // Convert time to pixels in timeline coordinates
    // The visible track starts at cropStart in the full audio
    const startX = (state.inpaintSelection.startTime - (track.cropStart || 0)) * pixelsPerSecond;
    const endX = (state.inpaintSelection.endTime - (track.cropStart || 0)) * pixelsPerSecond;

    return {
      startX: Math.max(0, startX),
      width: Math.max(0, endX - startX)
    };
  }, [state.inpaintSelection, track.id, track.cropStart, pixelsPerSecond]);

  // Calculate full audio duration for waveform rendering
  // No tempo scaling - notes are already converted to timeline BPM
  const fullAudioWidth = useMemo(() => {
    return (track.duration || 10) * pixelsPerSecond;
  }, [track.duration, pixelsPerSecond]);

  // Get waveform color from trackRef's CSS variable (set by :hover and .selected states)
  const [waveformColor, setWaveformColor] = useState('#667eea');
  const [isHovering, setIsHovering] = useState(false);

  // Update waveform color when selection or hover changes
  useEffect(() => {
    if (trackRef.current) {
      const computedStyle = getComputedStyle(trackRef.current);
      const color = computedStyle.getPropertyValue('--waveform-color').trim();
      if (color) {
        setWaveformColor(color);
      }
    }
  }, [isSelected, isHovering]);

  // Visual gain: the waveform bar heights are scaled by this so volume
  // slider moves are reflected in the waveform in real time. Muted tracks
  // collapse to a flat center line (gain=0). When this track is inside a
  // collapsed bus we don't scale here — the composite waveform handles
  // aggregation; individual collapsed-overlay instances use gain=1 so
  // they still render meaningfully (though they're no longer used once
  // CompositeBusWaveform is wired in).
  const visualGain = track.isMuted ? 0 : (track.gain ?? 1.0);

  const { canvasRef, duration: actualDuration } = useWaveform(
    track.type === 'midi' ? null : track.audioUrl, // Skip waveform for MIDI tracks
    fullAudioWidth,
    trackHeight,
    waveformColor,
    0, // No crop on rendering
    0, // No crop on rendering
    track.metadata?.envelopeData || null,  // latent_visual or v4-small instant envelope
    track.metadata?.envelopeFps || 25,     // v4 outputs ≈31.25 Hz, latent_visual is 25
    visualGain,
    // Show noise loop whenever there's nothing painted yet (loading, generating, placeholder)
    !!(track.isPlaceholder || track.metadata?.generating),
  );

  // Update track duration when audio loads — only when the delta is
  // meaningful (>10ms). The Web Audio decoded duration often differs
  // from the metadata duration by microseconds, which would otherwise
  // trigger an UPDATE_TRACK on every load and cascade re-renders
  // (and re-fetches) across every track in the project.
  useEffect(() => {
    if (!actualDuration) return;
    if (Math.abs(actualDuration - (track.duration || 0)) < 0.01) return;
    dispatch({
      type: 'UPDATE_TRACK',
      payload: {
        busId,
        trackId: track.id,
        updates: { duration: actualDuration },
        skipHistory: true,
      }
    });
  }, [actualDuration, track.duration, track.id, busId, dispatch]);

  // Calculate transform-based position
  const trackTransform = useMemo(() => {
    const x = (track.startPosition || 0) * pixelsPerSecond;
    // Tracks are positioned within busTracks container
    // When collapsed: tracks overlay at y=0 (master view)
    // When expanded: busTracks.expanded has padding-top: trackHeight to offset for bus header
    //   So tracks can start at y=0 and spread with index * trackHeight
    const y = isExpanded ? (index * trackHeight) : 0;
    // Use translate3d for GPU acceleration
    return `translate3d(${x}px, ${y}px, 0)`;
  }, [track.startPosition, pixelsPerSecond, index, isExpanded, trackHeight]);

  // Calculate track container width based on visible audio (after cropping)
  // No tempo scaling - notes are already converted to timeline BPM
  const trackWidth = useMemo(() => {
    const visibleDuration = (track.duration || 10) - (track.cropStart || 0) - (track.cropEnd || 0);
    return visibleDuration * pixelsPerSecond;
  }, [track.duration, track.cropStart, track.cropEnd, pixelsPerSecond]);

  // Calculate waveform offset inside container (for cropping)
  // No tempo scaling - notes are already converted to timeline BPM
  const waveformOffset = useMemo(() => {
    return -(track.cropStart || 0) * pixelsPerSecond;
  }, [track.cropStart, pixelsPerSecond]);

  // Handle track selection
  const handleSelect = useCallback((e) => {
    if (isDragging || isResizing) return;
    e.stopPropagation();

    // Select the track
    dispatch({ type: 'SELECT_TRACK', payload: { trackId: track.id } });

    // Open the sidebar if it's collapsed
    if (state.stemsSidebar?.isCollapsed) {
      dispatch({ type: 'TOGGLE_STEMS_SIDEBAR' });
    }
  }, [dispatch, track.id, isDragging, isResizing, state.stemsSidebar]);

  // Plugin mode: intercept HTML5 dragstart to use dodrag:// scheme instead
  const handlePluginDragStart = useCallback((e) => {
    if (!state.pluginMode || !track.audioUrl) return;
    e.preventDefault();
    initiatePluginDrag();
  }, [state.pluginMode, track.audioUrl, initiatePluginDrag]);

  // Plugin mode: hold-to-drag timer ref
  const pluginHoldTimerRef = useRef(null);

  // Drag handling - uses native mouse events for timeline movement (X-axis only in plugin mode)
  const handleMouseDragStart = useCallback((e) => {
    if (e.button !== 0) return; // Only left mouse button
    if (e.target.classList.contains(styles.resizeHandle)) return; // Don't drag when resizing

    e.preventDefault();
    setIsDragging(true);

    dragStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      startPosition: track.startPosition || 0,
      startCrop: track.cropStart || 0,
      startWidth: track.width || 800
    };

    // Reset drag direction
    dragDirectionRef.current = null;

    // Plugin mode: 200ms hold-to-drag — if user holds without significant X movement, trigger plugin drag
    if (state.pluginMode && track.audioUrl) {
      clearTimeout(pluginHoldTimerRef.current);
      pluginHoldTimerRef.current = setTimeout(() => {
        if (!dragDirectionRef.current) {
          setIsDragging(false);
          initiatePluginDrag();
        }
      }, 200);
    }

    // Save to history ONCE at drag start
    dispatch({
      type: 'SAVE_HISTORY_SNAPSHOT'
    });

    // Attach global listeners for smooth dragging
    const handleDragMove = (moveEvent) => {
      const deltaX = moveEvent.clientX - dragStartRef.current.x;
      const deltaY = moveEvent.clientY - dragStartRef.current.y;

      // Determine drag direction if not yet determined
      if (!dragDirectionRef.current && (Math.abs(deltaX) > 5 || Math.abs(deltaY) > 5)) {
        dragDirectionRef.current = Math.abs(deltaY) > Math.abs(deltaX) ? 'y' : 'x';
        // Clear hold-to-drag timer once direction is determined
        clearTimeout(pluginHoldTimerRef.current);

        // If Y-axis drag in plugin mode, trigger plugin drag
        if (dragDirectionRef.current === 'y' && state.pluginMode && track.audioUrl) {
          setIsDragging(false);
          document.removeEventListener('mousemove', handleDragMove);
          document.removeEventListener('mouseup', handleDragEnd);
          initiatePluginDrag();
          return;
        }
      }

      // Only continue with X-axis timeline movement
      if (!dragDirectionRef.current || dragDirectionRef.current === 'x') {
        const deltaSeconds = deltaX / pixelsPerSecond;
        const rawPosition = Math.max(0, dragStartRef.current.startPosition + deltaSeconds);

        // Apply snap to beat/sub-beat (only in BPM mode)
        const newPosition = snapToGrid(rawPosition);

        if (isMasterView) {
          // In master view, move ALL tracks in the bus together
          const deltaPosition = newPosition - dragStartRef.current.startPosition;
          dispatch({
            type: 'UPDATE_ALL_TRACKS_POSITION',
            payload: {
              busId,
              deltaPosition,
              skipHistory: true  // Don't save history during drag
            }
          });
        } else {
          // Normal mode: update just this track
          dispatch({
            type: 'UPDATE_TRACK',
            payload: {
              busId,
              trackId: track.id,
              updates: { startPosition: newPosition },
              skipHistory: true  // Don't save history during drag
            }
          });
        }
      }
    };

    const handleDragEnd = () => {
      setIsDragging(false);
      dragDirectionRef.current = null;
      clearTimeout(pluginHoldTimerRef.current);
      document.removeEventListener('mousemove', handleDragMove);
      document.removeEventListener('mouseup', handleDragEnd);
    };

    document.addEventListener('mousemove', handleDragMove);
    document.addEventListener('mouseup', handleDragEnd);
  }, [track.id, track.startPosition, track.cropStart, track.width, busId, dispatch, pixelsPerSecond, snapToGrid, isMasterView, state.pluginMode, track.audioUrl, initiatePluginDrag]);

  // Resize handling
  const handleResizeStart = useCallback((side) => (e) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizing(side);

    dragStartRef.current = {
      x: e.clientX,
      startPosition: track.startPosition || 0,
      startCropStart: track.cropStart || 0,
      startCropEnd: track.cropEnd || 0
    };

    // Save to history ONCE at resize start
    dispatch({
      type: 'SAVE_HISTORY_SNAPSHOT'
    });

    const handleResizeMove = (moveEvent) => {
      const deltaX = moveEvent.clientX - dragStartRef.current.x;
      const deltaTime = deltaX / pixelsPerSecond;

      if (side === 'left') {
        // Left edge: move startPosition and increase cropStart by same amount
        const rawCropStart = Math.max(0, Math.min(dragStartRef.current.startCropStart + deltaTime, track.duration - (track.cropEnd || 0) - 0.1));
        const rawStartPosition = dragStartRef.current.startPosition + (rawCropStart - dragStartRef.current.startCropStart);

        // Apply snap to the position (which determines cropStart)
        const snappedPosition = snapToGrid(rawStartPosition);
        const snappedDelta = snappedPosition - dragStartRef.current.startPosition;
        const newCropStart = dragStartRef.current.startCropStart + snappedDelta;

        dispatch({
          type: 'UPDATE_TRACK',
          payload: {
            busId,
            trackId: track.id,
            updates: {
              cropStart: newCropStart,
              startPosition: snappedPosition
            },
            skipHistory: true  // Don't save history during resize
          }
        });
      } else {
        // Right edge: increase cropEnd (shrink from right)
        const newCropEnd = Math.max(0, Math.min(dragStartRef.current.startCropEnd - deltaTime, track.duration - (track.cropStart || 0) - 0.1));

        dispatch({
          type: 'UPDATE_TRACK',
          payload: {
            busId,
            trackId: track.id,
            updates: { cropEnd: newCropEnd },
            skipHistory: true  // Don't save history during resize
          }
        });
      }
    };

    const handleResizeEnd = () => {
      setIsResizing(null);
      document.removeEventListener('mousemove', handleResizeMove);
      document.removeEventListener('mouseup', handleResizeEnd);
    };

    document.addEventListener('mousemove', handleResizeMove);
    document.addEventListener('mouseup', handleResizeEnd);
  }, [track.id, track.startPosition, track.cropStart, track.cropEnd, track.width, track.duration, busId, dispatch, pixelsPerSecond, snapToGrid]);

  // Inpaint selection handling
  const handleInpaintMouseDown = useCallback((e) => {
    if (!isInInpaintMode) return;
    if (e.button !== 0) return; // Only left mouse button

    e.preventDefault();
    e.stopPropagation();

    const rect = trackRef.current.getBoundingClientRect();
    // Account for 1px border
    const borderWidth = 1;
    const startX = e.clientX - rect.left - borderWidth;

    inpaintSelectionRef.current = {
      isSelecting: true,
      startX: startX,
      startClientX: e.clientX
    };

    setInpaintSelection({ startX, width: 0 });

    const handleInpaintMouseMove = (moveEvent) => {
      if (!inpaintSelectionRef.current.isSelecting) return;

      // Get fresh bounding rect on each move to account for scrolling
      const currentRect = trackRef.current.getBoundingClientRect();
      // Account for 1px border
      const borderWidth = 1;
      const currentX = moveEvent.clientX - currentRect.left - borderWidth;

      // Recalculate startX based on current rect (in case of scrolling)
      const adjustedStartX = inpaintSelectionRef.current.startClientX - currentRect.left - borderWidth;
      const width = currentX - adjustedStartX;

      setInpaintSelection({
        startX: width >= 0 ? adjustedStartX : currentX,
        width: Math.abs(width)
      });
    };

    const handleInpaintMouseUp = (upEvent) => {
      if (!inpaintSelectionRef.current.isSelecting) return;

      inpaintSelectionRef.current.isSelecting = false;

      // Get final bounding rect
      const finalRect = trackRef.current.getBoundingClientRect();
      // Account for 1px border
      const borderWidth = 1;
      const finalX = upEvent.clientX - finalRect.left - borderWidth;
      const adjustedStartX = inpaintSelectionRef.current.startClientX - finalRect.left - borderWidth;

      const actualStartX = Math.min(adjustedStartX, finalX);
      const actualEndX = Math.max(adjustedStartX, finalX);
      const actualWidth = actualEndX - actualStartX;

      // Convert pixels to time in the FULL audio (not cropped)
      // The visible track starts at cropStart in the full audio
      const startTimeInFullAudio = (actualStartX / pixelsPerSecond) + (track.cropStart || 0);
      const endTimeInFullAudio = (actualEndX / pixelsPerSecond) + (track.cropStart || 0);

      // Clamp to actual audio bounds
      const clampedStart = Math.max(track.cropStart || 0, Math.min(startTimeInFullAudio, track.duration - (track.cropEnd || 0)));
      const clampedEnd = Math.max(track.cropStart || 0, Math.min(endTimeInFullAudio, track.duration - (track.cropEnd || 0)));

      // Dispatch selection to global state
      if (clampedEnd - clampedStart >= 0.1) { // Minimum 0.1 second selection
        dispatch({
          type: 'SET_INPAINT_SELECTION',
          payload: {
            trackId: track.id,
            startTime: clampedStart,
            endTime: clampedEnd
          }
        });
      } else {
        // Selection too small, clear it
        setInpaintSelection(null);
      }

      // Clear local selection after dispatching
      setInpaintSelection(null);

      document.removeEventListener('mousemove', handleInpaintMouseMove);
      document.removeEventListener('mouseup', handleInpaintMouseUp);
    };

    document.addEventListener('mousemove', handleInpaintMouseMove);
    document.addEventListener('mouseup', handleInpaintMouseUp);
  }, [isInInpaintMode, track.id, track.duration, track.cropStart, track.cropEnd, pixelsPerSecond, dispatch]);

  return (
    <div
      ref={trackRef}
      data-track-id={track.id}
      className={`${styles.track} ${isSelected ? styles.selected : ''} ${isMultiSelected ? styles.multiSelected : ''} ${isDragging ? styles.dragging : ''} ${isInInpaintMode ? styles.inpaintMode : ''} ${track.isMuted ? styles.muted : ''} ${track.isPlaceholder ? styles.placeholder : ''}`}
      style={{
        transform: trackTransform,
        width: `${trackWidth}px`,
        height: `${trackHeight}px`,
        opacity: track.isMuted ? 0.6 : 1
      }}
      draggable={state.pluginMode && !!track.audioUrl}
      onDragStart={state.pluginMode ? handlePluginDragStart : undefined}
      onMouseDown={isInInpaintMode ? handleInpaintMouseDown : handleMouseDragStart}
      onClick={handleSelect}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => { setIsHovering(false); clearTimeout(pluginHoldTimerRef.current); }}
    >
      {/* Visualization - MIDI piano roll OR canvas (waveform / noise) */}
      {track.type === 'midi' && !track.metadata?.generating && !track.isPlaceholder ? (
        <div style={{ transform: `translateX(${waveformOffset}px)` }}>
          <MIDITrackVisualization
            midiData={track.midiData}
            width={fullAudioWidth}
            height={trackHeight}
            pixelsPerSecond={pixelsPerSecond}
            startTime={track.cropStart || 0}
            endTime={(track.duration || 10) - (track.cropEnd || 0)}
            timelineBpm={state.bpm || 120}
            f0Contour={track.f0Contour}
          />
        </div>
      ) : (
        <>
          <canvas
            ref={canvasRef}
            className={styles.waveform}
            style={{
              transform: `translateX(${waveformOffset}px)`,
              height: `${trackHeight}px`,
            }}
          />
        </>
      )}

      {/* Inpaint selection overlay - show either active drag or confirmed selection */}
      {isInInpaintMode && (inpaintSelection || confirmedSelection) && (
        <div
          className={styles.inpaintSelectionOverlay}
          style={{
            left: `${(inpaintSelection || confirmedSelection).startX + 1}px`, // Add 1px border offset
            width: `${(inpaintSelection || confirmedSelection).width}px`
          }}
        />
      )}

      {/* Resize handles - hidden in inpaint mode */}
      {!isInInpaintMode && (
        <>
          <div
            className={`${styles.resizeHandle} ${styles.resizeHandleLeft}`}
            onMouseDown={handleResizeStart('left')}
          />
          <div
            className={`${styles.resizeHandle} ${styles.resizeHandleRight}`}
            onMouseDown={handleResizeStart('right')}
          />
        </>
      )}
    </div>
  );
});

OptimizedTrack.displayName = 'OptimizedTrack';

export default OptimizedTrack;
