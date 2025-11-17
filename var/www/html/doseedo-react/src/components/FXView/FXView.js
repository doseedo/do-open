import React from 'react';
import FXPanel from './FXPanel';
import styles from './FXView.module.css';

/**
 * FXView - Displayed when FX mode is selected
 * Shows FX panel controls in the main content area (replaces video container)
 */
const FXView = () => {
  return (
    <div className={styles.fxViewContainer}>
      <div className={styles.fxContent}>
        <div className={styles.fxHeader}>
          <div className={styles.fxIcon}>
            <i className="fa-solid fa-sliders"></i>
          </div>
          <h2 className={styles.fxTitle}>FX Controls</h2>
          <p className={styles.fxDescription}>
            Control track effects and reverb sends for all tracks.
          </p>
        </div>

        {/* FX Panel */}
        <FXPanel />

        <div className={styles.fxHint}>
          <i className="fa-solid fa-lightbulb"></i>
          <span>Adjust reverb sends for each track and master reverb level</span>
        </div>
      </div>
    </div>
  );
};

export default FXView;
