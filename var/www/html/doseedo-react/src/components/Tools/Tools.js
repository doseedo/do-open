import React, { useState, useCallback } from 'react';
import styles from './Tools.module.css';
import VocalHarmonizer from './VocalHarmonizer';

/**
 * Tools Component
 * Display all available AI tools and features
 */
const Tools = () => {
  const [activeTool, setActiveTool] = useState(null);

  const handleToolClick = useCallback((toolName, isActive = false) => {
    console.log(`🛠️ Tool clicked: ${toolName}`);
    if (isActive) {
      setActiveTool(toolName);
    } else {
      alert(`${toolName} - Coming Soon!`);
    }
  }, []);

  const handleBackToTools = useCallback(() => {
    setActiveTool(null);
  }, []);

  const tools = [
    {
      name: 'Vocal Harmonizer',
      icon: 'fa-users',
      description: 'Generate harmonies from your vocal recordings',
      color: 'rgba(255, 140, 100, 0.2)',
      isActive: true
    },
    {
      name: 'Video to Music',
      icon: 'fa-video',
      description: 'Transform your videos into professional music',
      color: 'rgba(102, 126, 234, 0.2)',
      isActive: false
    },
    {
      name: 'Lyric Edit',
      icon: 'fa-pen-to-square',
      description: 'Edit and generate lyrics with AI assistance',
      color: 'rgba(156, 130, 200, 0.2)',
      isActive: false
    },
    {
      name: 'Voice to Instrument',
      icon: 'fa-microphone-lines',
      description: 'Convert voice recordings to instrumental tracks',
      color: 'rgba(102, 126, 234, 0.2)',
      isActive: false
    },
    {
      name: 'Sample Regenerator',
      icon: 'fa-rotate',
      description: 'Regenerate and enhance audio samples',
      color: 'rgba(186, 156, 255, 0.2)',
      isActive: false
    },
    {
      name: 'Stem Separation',
      icon: 'fa-layer-group',
      description: 'Separate audio into individual stems',
      color: 'rgba(156, 130, 200, 0.2)',
      isActive: false
    },
    {
      name: 'Beat Generator',
      icon: 'fa-drum',
      description: 'Generate custom beats and rhythms',
      color: 'rgba(102, 126, 234, 0.2)',
      isActive: false
    }
  ];

  // Render active tool component
  if (activeTool === 'Vocal Harmonizer') {
    return <VocalHarmonizer onBack={handleBackToTools} />;
  }

  return (
    <div className={styles.toolsContainer}>
      <div className={styles.toolsHeader}>
        <h1 className={styles.toolsTitle}>AI Tools</h1>
        <p className={styles.toolsSubtitle}>Explore our suite of AI-powered creative tools</p>
      </div>

      <div className={styles.toolsGrid}>
        {tools.map((tool, index) => (
          <div
            key={index}
            className={`${styles.toolCard} ${tool.isActive ? styles.toolCardActive : ''}`}
            onClick={() => handleToolClick(tool.name, tool.isActive)}
            style={{ background: `linear-gradient(135deg, ${tool.color}, rgba(0, 0, 0, 0.1))` }}
          >
            <div className={styles.toolIcon}>
              <i className={`fa-solid ${tool.icon}`}></i>
            </div>
            <h3 className={styles.toolName}>{tool.name}</h3>
            <p className={styles.toolDescription}>{tool.description}</p>
            {tool.isActive ? (
              <div className={styles.toolBadgeActive}>Available</div>
            ) : (
              <div className={styles.toolBadge}>Coming Soon</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Tools;
