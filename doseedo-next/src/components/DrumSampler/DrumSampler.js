import React, { useState, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import drumSamplerAPI from '../../services/drumSamplerAPI';
import styles from './DrumSampler.module.css';

/**
 * DrumSampler Component
 * UI for generating drum samples from your HuggingFace Space
 */
const DrumSampler = () => {
  const { state, dispatch } = useApp();
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerate = useCallback(async () => {
    setIsGenerating(true);
    setError(null);

    try {
      const currentBPM = state.bpm || 120;
      console.log(`🎲 Generating random drum pattern at ${currentBPM} BPM...`);

      // Generate random drum sample with current timeline BPM
      const result = await drumSamplerAPI.generateAndDownloadDrumSamples(currentBPM);

      if (!result || !result.blob) {
        throw new Error('No audio generated');
      }

      // Convert blob to audio buffer
      const arrayBuffer = await result.blob.arrayBuffer();
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

      // Automatically add to timeline (no preview step)
      const busId = `drums-${Date.now()}`;
      const trackId = `drum-track-${Date.now()}`;
      const trackName = result.midiFile || 'Drum Pattern';

      // Create a new Drums bus
      dispatch({
        type: 'CREATE_BUS',
        payload: {
          id: busId,
          type: 'Drums',
          name: trackName,
          expanded: true
        }
      });

      // Add the drum track to the bus
      dispatch({
        type: 'ADD_TRACK',
        payload: {
          busId: busId,
          track: {
            id: trackId,
            name: trackName,
            audioBuffer: audioBuffer,
            audioUrl: result.audioUrl,  // Use persistent server URL
            midiData: null,
            samples: [],
            gain: 1.0,
            pan: 0,
            mute: false,
            solo: false,
            color: '#ff6b6b',
            instrumentGroup: 'drums',
            instrumentSubgroup: result.drumKit || 'sampler',
            source: 'drumsampler'
          }
        }
      });

      console.log('✅ Drum sample generated and added to timeline:', result.midiFile);

    } catch (err) {
      console.error('❌ Error generating drum:', err);
      setError(err.message);
    } finally {
      setIsGenerating(false);
    }
  }, [state.bpm, dispatch]);

  return (
    <div className={styles.drumSampler}>
      <div className={styles.header}>
        <h3 className={styles.title}>
          <i className="fa-solid fa-wand-magic-sparkles"></i>
          AI Drum Sampler
        </h3>
        <p className={styles.subtitle}>Generate random drum patterns - automatically added to timeline</p>
      </div>

      {/* Generate Button */}
      <button
        className={styles.generateBtn}
        onClick={handleGenerate}
        disabled={isGenerating}
      >
        {isGenerating ? (
          <>
            <i className="fa-solid fa-spinner fa-spin"></i>
            Generating Random Pattern...
          </>
        ) : (
          <>
            <i className="fa-solid fa-dice"></i>
            Generate Random Drum Pattern
          </>
        )}
      </button>

      {/* Error Display */}
      {error && (
        <div className={styles.error}>
          <i className="fa-solid fa-exclamation-triangle"></i>
          {error}
        </div>
      )}

      {/* Info */}
      <div className={styles.info}>
        <i className="fa-solid fa-info-circle"></i>
        <span>Space: doseedo/DrumSampler</span>
      </div>
    </div>
  );
};

export default DrumSampler;
