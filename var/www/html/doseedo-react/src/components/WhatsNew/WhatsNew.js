import React from 'react';
import styles from './WhatsNew.module.css';

/**
 * WhatsNew Component
 * Display latest updates and features
 */
const WhatsNew = () => {
  const updates = [
    {
      version: 'v1.0',
      date: 'February 2026',
      title: 'Brass Mute V1',
      features: [
        'Multiple mute types: Straight, Cup, Harmon, Plunger, Bucket',
        'Real-time acoustic modeling for authentic mute tones',
        'Low-latency processing suitable for live performance',
        'VST3 and AU formats included',
        'Adjustable mute depth and resonance controls',
      ]
    },
    {
      version: 'Private Beta',
      date: 'November 2025',
      title: 'Private Beta Launch',
      features: [
        'Video to Music - Transform your videos into professional music',
        'Lyric Edit - Edit and generate lyrics with AI assistance',
        'Voice to Instrument - Convert voice recordings to instrumental tracks',
        'Sample Regenerator - Regenerate and enhance audio samples',
        'Stem Separation - Separate audio into individual stems',
        'Beat Generator - Generate custom beats and rhythms'
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
