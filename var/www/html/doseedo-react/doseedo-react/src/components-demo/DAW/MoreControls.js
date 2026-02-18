import React, { useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import Button from '../common/Button';
import Slider from '../common/Slider';
import TransportControls from './TransportControls';

/**
 * MoreControls Component - Automation and zoom controls (MEMOIZED)
 * Uses reusable Button and Slider components
 */
const MoreControls = React.memo(() => {
  const { state, dispatch } = useApp();

  // Memoize callbacks to prevent recreation
  const handlePlayPause = useCallback(() => {
    dispatch({ type: 'TOGGLE_PLAY' });
  }, [dispatch]);

  const handleStop = useCallback(() => {
    dispatch({ type: 'SET_PLAYING', payload: false });
    dispatch({ type: 'RESET_PLAYHEAD' });
  }, [dispatch]);

  const toggleAutomation = useCallback(() => {
    dispatch({ type: 'TOGGLE_AUTOMATION_WINDOW' });
  }, [dispatch]);

  const clearAutomation = useCallback(() => {
    if (!window.confirm('Clear all automation points?')) {
      return;
    }

    // Keep only edge points
    const edgePoints = state.automationWindow.points.filter(p => p.isEdge);
    dispatch({
      type: 'UPDATE_AUTOMATION_POINTS',
      payload: edgePoints
    });
  }, [dispatch, state.automationWindow.points]);

  const restoreSceneAutomation = useCallback(() => {
    if (!state.video.sceneChanges || state.video.sceneChanges.length === 0) {
      alert('No scene changes detected. Upload a video with scene detection first.');
      return;
    }

    console.log('🔄 Restoring scene automation points...');

    // Keep edge points
    const edgePoints = state.automationWindow.points.filter(p => p.isEdge);

    // Get the default volume (50% / 0.5)
    const defaultVolume = 0.5;
    const midVolume = edgePoints.find(p => p.time === 0)?.volume || defaultVolume;

    // Create new points at each scene change
    const scenePoints = state.video.sceneChanges
      .filter(time => time > 0 && time < state.totalDuration)
      .map(time => ({
        time,
        volume: midVolume,
        isEdge: false
      }));

    // Combine edge points and scene points
    const updatedPoints = [...edgePoints, ...scenePoints].sort((a, b) => a.time - b.time);

    console.log(`✅ Restored ${scenePoints.length} scene points`);

    dispatch({
      type: 'UPDATE_AUTOMATION_POINTS',
      payload: updatedPoints
    });
  }, [dispatch, state.automationWindow.points, state.video.sceneChanges, state.totalDuration]);

  const handleZoomChange = useCallback((value) => {
    dispatch({
      type: 'UPDATE_ZOOM_LEVEL',
      payload: value
    });
  }, [dispatch]);

  return (
    <div className="morecontrols" style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
      <TransportControls
        isPlaying={state.isPlaying}
        playheadPosition={state.playheadPosition}
        onPlayPause={handlePlayPause}
        onStop={handleStop}
      />

      {/* Automation button - hidden for now
      <Button
        id="autobtn"
        icon="fa-solid fa-chart-simple"
        onClick={toggleAutomation}
        isActive={state.automationWindow.isVisible}
        title="Toggle automation"
      />

      {state.automationWindow.isVisible && (
        <>
          <Button
            id="auto-clear-btn"
            icon="fa-solid fa-eraser"
            onClick={clearAutomation}
            title="Clear automation points"
          />

          <Button
            id="auto-restore-btn"
            icon="fa-solid fa-rotate-left"
            onClick={restoreSceneAutomation}
            title="Restore scene points"
          />
        </>
      )}
      */}

      <div
        className="zoom-controls"
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          marginLeft: '20px',
          gap: '10px'
        }}
      >
        <i className="fa-solid fa-magnifying-glass-plus" style={{ color: '#aaa', fontSize: '14px' }}></i>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: 0, color: '#aaa', fontSize: '13px' }}>
          <span>Zoom:</span>
          <Slider
            id="zoom-slider"
            value={state.zoomLevel || 1.0}
            min={1}
            max={10}
            step={0.1}
            onChange={handleZoomChange}
          />
          <span id="zoom-value" style={{ minWidth: '35px', textAlign: 'right' }}>
            {(state.zoomLevel || 1.0).toFixed(1)}x
          </span>
        </label>
      </div>
    </div>
  );
});

MoreControls.displayName = 'MoreControls';

export default MoreControls;
