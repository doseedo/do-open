import React from 'react';
import styles from './ZoomControls.module.css';

/**
 * ZoomControls Component
 * Reusable zoom in/out controls with level display
 *
 * @param {number} zoomLevel - Current zoom level (0.2 to 5.0)
 * @param {Function} onZoomIn - Callback for zoom in button (horizontal)
 * @param {Function} onZoomOut - Callback for zoom out button (horizontal)
 * @param {number} trackHeight - Current track height in pixels
 * @param {Function} onZoomYIn - Callback for vertical zoom in button
 * @param {Function} onZoomYOut - Callback for vertical zoom out button
 */
const ZoomControls = React.memo(({
  zoomLevel,
  onZoomIn,
  onZoomOut,
  trackHeight,
  onZoomYIn,
  onZoomYOut
}) => {
  return (
    <div className={styles.zoomControls}>
      {/* Horizontal Zoom Controls */}
      <button
        className={styles.button}
        onClick={onZoomOut}
        title="Zoom Out (Horizontal)"
      >
        <i className="fa-solid fa-magnifying-glass-minus"></i>
      </button>
      <button
        className={styles.button}
        onClick={onZoomIn}
        title="Zoom In (Horizontal)"
      >
        <i className="fa-solid fa-magnifying-glass-plus"></i>
      </button>

      {/* Vertical Zoom Controls (Track Height) */}
      {onZoomYIn && onZoomYOut && (
        <>
          <button
            className={styles.button}
            onClick={onZoomYOut}
            title="Decrease Track Height"
          >
            <i className="fa-solid fa-up-down"></i>
            <i className="fa-solid fa-minus" style={{ fontSize: '10px', marginLeft: '-4px' }}></i>
          </button>
          <button
            className={styles.button}
            onClick={onZoomYIn}
            title="Increase Track Height"
          >
            <i className="fa-solid fa-up-down"></i>
            <i className="fa-solid fa-plus" style={{ fontSize: '10px', marginLeft: '-4px' }}></i>
          </button>
        </>
      )}
    </div>
  );
});

ZoomControls.displayName = 'ZoomControls';

export default ZoomControls;
