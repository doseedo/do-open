import React, { useRef, useCallback } from 'react';
import CanvasComponent from './CanvasComponent';
import styles from './PluginCreator.module.css';

const PluginCanvas = ({ config, components, selectedId, onSelect, onDeselect, onUpdateComponent, onDragStop, snapToGrid, gridSize, editorMode, paramValues, onParamChange }) => {
  const canvasRef = useRef(null);

  const handleCanvasClick = useCallback((e) => {
    if (e.target === canvasRef.current) {
      onDeselect();
    }
  }, [onDeselect]);

  // Sort components by zIndex for rendering order
  const sorted = [...components].sort((a, b) => (a.zIndex || 0) - (b.zIndex || 0));

  return (
    <div className={styles.pluginFrame}>
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
        {sorted.map(comp => (
          <CanvasComponent
            key={comp.id}
            component={comp}
            isSelected={comp.id === selectedId}
            onSelect={onSelect}
            onUpdate={onUpdateComponent}
            onDragStop={onDragStop}
            canvasWidth={config.width}
            canvasHeight={config.height}
            editorMode={editorMode}
            paramValue={paramValues?.[comp.id] ?? 0.5}
            onParamChange={onParamChange}
          />
        ))}
      </div>
    </div>
  );
};

export default PluginCanvas;
