import React from 'react';
import styles from './WhatsNew.module.css';

/**
 * WhatsNew Component
 * Display latest updates and features
 */
const WhatsNew = () => {
  const updates = [
    {
      version: 'v1.2.0',
      date: 'November 2024',
      title: 'URL Routing & Session Management',
      features: [
        'Added URL-based navigation (/dashboard, /projects, /studio, etc.)',
        'Implemented session save/load functionality',
        'Auto-save sessions every 3 seconds',
        'Spotify-style sessions list with waveform previews'
      ]
    },
    {
      version: 'v1.1.0',
      date: 'November 2024',
      title: 'UI Improvements',
      features: [
        'Redesigned My Sessions page',
        'Added smooth sidebar transitions',
        'Improved responsive layout',
        'Enhanced theme system'
      ]
    },
    {
      version: 'v1.0.0',
      date: 'October 2024',
      title: 'Initial Release',
      features: [
        'AI-powered music generation',
        'Multi-track DAW interface',
        'Real-time audio playback',
        'Video to music conversion'
      ]
    }
  ];

  return (
    <div className={styles.whatsNewContainer}>
      <div className={styles.header}>
        <h1 className={styles.title}>What's New</h1>
        <p className={styles.subtitle}>Latest updates and features</p>
      </div>

      <div className={styles.updatesList}>
        {updates.map((update, index) => (
          <div key={index} className={styles.updateCard}>
            <div className={styles.updateHeader}>
              <div>
                <h2 className={styles.updateTitle}>{update.title}</h2>
                <div className={styles.updateMeta}>
                  <span className={styles.version}>{update.version}</span>
                  <span className={styles.date}>{update.date}</span>
                </div>
              </div>
            </div>

            <ul className={styles.featuresList}>
              {update.features.map((feature, idx) => (
                <li key={idx} className={styles.feature}>
                  <i className="fa-solid fa-check"></i>
                  <span>{feature}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
};

export default WhatsNew;
