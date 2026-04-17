import React, { useState, useRef, useEffect } from 'react';
import VocalHarmonizerTool from './VocalHarmonizerTool';
import VoiceToInstrumentTool from './VoiceToInstrumentTool';
import LyricEditTool from './LyricEditTool';
import StemSeparationTool from './StemSeparationTool';
import VideoToMusicTool from './VideoToMusicTool';
import SampleRegeneratorTool from './SampleRegeneratorTool';
import BeatGeneratorTool from './BeatGeneratorTool';
import styles from './Tools.module.css';

/**
 * Tools Component
 * Display all available AI tools with scrollable selection bar
 * When a tool is selected, shows the tool-specific generator UI
 */
const Tools = () => {
  const [selectedTool, setSelectedTool] = useState(null);
  const toolBarRef = useRef(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  const tools = [
    {
      id: 'vocal-harmonizer',
      name: 'Vocal Harmonizer',
      icon: 'fa-music',
      description: 'Generate beautiful harmony tracks from your vocals',
      color: 'rgba(76, 175, 80, 0.2)',
      component: VocalHarmonizerTool,
      available: true
    },
    {
      id: 'video-to-music',
      name: 'Video to Music',
      icon: 'fa-video',
      description: 'Transform your videos into professional music',
      color: 'rgba(102, 126, 234, 0.2)',
      component: VideoToMusicTool
    },
    {
      id: 'lyric-edit',
      name: 'Lyric Edit',
      icon: 'fa-pen-to-square',
      description: 'Edit and generate lyrics with AI assistance',
      color: 'rgba(156, 130, 200, 0.2)',
      component: LyricEditTool
    },
    {
      id: 'voice-to-instrument',
      name: 'Voice to Instrument',
      icon: 'fa-microphone-lines',
      description: 'Convert voice recordings to instrumental tracks',
      color: 'rgba(102, 126, 234, 0.2)',
      component: VoiceToInstrumentTool
    },
    {
      id: 'sample-regenerator',
      name: 'Sample Regenerator',
      icon: 'fa-rotate',
      description: 'Regenerate and enhance audio samples',
      color: 'rgba(186, 156, 255, 0.2)',
      component: SampleRegeneratorTool
    },
    {
      id: 'stem-separation',
      name: 'Stem Separation',
      icon: 'fa-layer-group',
      description: 'Separate audio into individual stems',
      color: 'rgba(156, 130, 200, 0.2)',
      component: StemSeparationTool
    },
    {
      id: 'beat-generator',
      name: 'Beat Generator',
      icon: 'fa-drum',
      description: 'Generate custom beats and rhythms',
      color: 'rgba(102, 126, 234, 0.2)',
      component: BeatGeneratorTool
    }
  ];

  // Check scroll state
  const updateScrollState = () => {
    if (toolBarRef.current) {
      const { scrollLeft, scrollWidth, clientWidth } = toolBarRef.current;
      setCanScrollLeft(scrollLeft > 0);
      setCanScrollRight(scrollLeft + clientWidth < scrollWidth - 1);
    }
  };

  // Update scroll state on mount and resize
  useEffect(() => {
    updateScrollState();
    window.addEventListener('resize', updateScrollState);
    return () => window.removeEventListener('resize', updateScrollState);
  }, [selectedTool]);

  // Scroll handlers
  const scrollToolBar = (direction) => {
    if (toolBarRef.current) {
      const scrollAmount = 200;
      toolBarRef.current.scrollBy({
        left: direction === 'left' ? -scrollAmount : scrollAmount,
        behavior: 'smooth'
      });
      setTimeout(updateScrollState, 300);
    }
  };

  const handleToolClick = (tool) => {
    setSelectedTool(tool);
  };

  const handleBackToGrid = () => {
    setSelectedTool(null);
  };

  // Tool selection bar (shown when a tool is selected)
  const renderToolBar = () => (
    <div className={styles.toolBarContainer}>
      {/* Left scroll arrow */}
      <button
        className={`${styles.toolBarArrow} ${styles.toolBarArrowLeft} ${!canScrollLeft ? styles.toolBarArrowHidden : ''}`}
        onClick={() => scrollToolBar('left')}
        disabled={!canScrollLeft}
      >
        <i className="fa-solid fa-chevron-left"></i>
      </button>

      {/* Scrollable tool buttons */}
      <div
        ref={toolBarRef}
        className={styles.toolBarScroll}
        onScroll={updateScrollState}
      >
        {tools.map((tool) => (
          <button
            key={tool.id}
            className={`${styles.toolBarItem} ${selectedTool?.id === tool.id ? styles.toolBarItemActive : ''}`}
            onClick={() => handleToolClick(tool)}
          >
            <i className={`fa-solid ${tool.icon}`}></i>
            <span>{tool.name}</span>
          </button>
        ))}
      </div>

      {/* Right scroll arrow */}
      <button
        className={`${styles.toolBarArrow} ${styles.toolBarArrowRight} ${!canScrollRight ? styles.toolBarArrowHidden : ''}`}
        onClick={() => scrollToolBar('right')}
        disabled={!canScrollRight}
      >
        <i className="fa-solid fa-chevron-right"></i>
      </button>
    </div>
  );

  // Grid view (no tool selected)
  const renderToolGrid = () => (
    <>
      <div className={styles.toolsHeader}>
        <h1 className={styles.toolsTitle}>AI Tools</h1>
        <p className={styles.toolsSubtitle}>Explore our suite of AI-powered creative tools</p>
      </div>

      <div className={styles.toolsGrid}>
        {tools.map((tool) => (
          <div
            key={tool.id}
            className={styles.toolCard}
            onClick={() => handleToolClick(tool)}
            style={{ background: `linear-gradient(135deg, ${tool.color}, rgba(0, 0, 0, 0.1))` }}
          >
            <div className={styles.toolIcon}>
              <i className={`fa-solid ${tool.icon}`}></i>
            </div>
            <h3 className={styles.toolName}>{tool.name}</h3>
            <p className={styles.toolDescription}>{tool.description}</p>
            <div className={styles.toolBadgeRow}>
              {tool.available && (
                <span className={styles.availableBadge}>Available</span>
              )}
              <div className={styles.toolBadge}>
                <i className="fa-solid fa-arrow-right"></i>
                Open Tool
              </div>
            </div>
          </div>
        ))}
      </div>
    </>
  );

  // Render the selected tool's component
  const renderSelectedToolView = () => {
    const ToolComponent = selectedTool?.component;

    return (
      <>
        {/* Tool selection bar at top */}
        {renderToolBar()}

        {/* Tool-specific UI */}
        {ToolComponent && <ToolComponent tool={selectedTool} onBack={handleBackToGrid} />}
      </>
    );
  };

  return (
    <div className={`${styles.toolsContainer} ${selectedTool ? styles.toolsContainerWithSelection : ''}`}>
      {selectedTool ? renderSelectedToolView() : renderToolGrid()}
    </div>
  );
};

export default Tools;
