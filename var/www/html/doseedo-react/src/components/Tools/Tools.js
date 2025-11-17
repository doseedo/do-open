import React from 'react';
import styles from './Tools.module.css';

/**
 * Tools Component
 * Display all available AI tools and features
 */
const Tools = () => {
  const handleToolClick = (toolName) => {
    console.log(`🛠️ Tool clicked: ${toolName}`);
    // TODO: Implement tool functionality
    alert(`${toolName} - Coming Soon!`);
  };

  const tools = [
    {
      name: 'Video to Music',
      icon: 'fa-video',
      description: 'Transform your videos into professional music',
      color: 'rgba(102, 126, 234, 0.2)'
    },
    {
      name: 'Video to SFX',
      icon: 'fa-volume-high',
      description: 'Generate sound effects from video content',
      color: 'rgba(186, 156, 255, 0.2)'
    },
    {
      name: 'AI Voiceover',
      icon: 'fa-microphone',
      description: 'Create AI-powered voiceovers for your projects',
      color: 'rgba(118, 75, 162, 0.2)'
    },
    {
      name: 'Lyric Edit',
      icon: 'fa-pen-to-square',
      description: 'Edit and generate lyrics with AI assistance',
      color: 'rgba(156, 130, 200, 0.2)'
    },
    {
      name: 'Voice to Instrument',
      icon: 'fa-microphone-lines',
      description: 'Convert voice recordings to instrumental tracks',
      color: 'rgba(102, 126, 234, 0.2)'
    },
    {
      name: 'Sample Regenerator',
      icon: 'fa-rotate',
      description: 'Regenerate and enhance audio samples',
      color: 'rgba(186, 156, 255, 0.2)'
    },
    {
      name: 'Audio Mastering',
      icon: 'fa-sliders',
      description: 'Professional AI-powered audio mastering',
      color: 'rgba(118, 75, 162, 0.2)'
    },
    {
      name: 'Stem Separation',
      icon: 'fa-layer-group',
      description: 'Separate audio into individual stems',
      color: 'rgba(156, 130, 200, 0.2)'
    },
    {
      name: 'Beat Generator',
      icon: 'fa-drum',
      description: 'Generate custom beats and rhythms',
      color: 'rgba(102, 126, 234, 0.2)'
    }
  ];

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
            className={styles.toolCard}
            onClick={() => handleToolClick(tool.name)}
            style={{ background: `linear-gradient(135deg, ${tool.color}, rgba(0, 0, 0, 0.1))` }}
          >
            <div className={styles.toolIcon}>
              <i className={`fa-solid ${tool.icon}`}></i>
            </div>
            <h3 className={styles.toolName}>{tool.name}</h3>
            <p className={styles.toolDescription}>{tool.description}</p>
            <div className={styles.toolBadge}>Coming Soon</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Tools;
