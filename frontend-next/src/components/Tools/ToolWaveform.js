import React, { useRef, useEffect, useState } from 'react';
import styles from './Tools.module.css';

/**
 * ToolWaveform - Static waveform display for tool generator UI
 * Similar to DAW waveform but not draggable on timeline
 */
const ToolWaveform = ({ audioUrl, height = 120, color = '#667eea' }) => {
  const canvasRef = useRef(null);
  const [waveformData, setWaveformData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [duration, setDuration] = useState(0);

  // Load and analyze audio when URL changes
  useEffect(() => {
    if (!audioUrl) {
      setWaveformData(null);
      setDuration(0);
      return;
    }

    let cancelled = false;
    setIsLoading(true);

    const loadAudio = async () => {
      try {
        const response = await fetch(audioUrl);
        const arrayBuffer = await response.arrayBuffer();

        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        if (cancelled) return;

        const channelData = audioBuffer.getChannelData(0);
        const numBars = 200;
        const samplesPerBar = Math.floor(channelData.length / numBars);
        const samples = [];

        let maxRms = 0;
        for (let i = 0; i < numBars; i++) {
          const start = i * samplesPerBar;
          const end = Math.min(start + samplesPerBar, channelData.length);

          let sumSquares = 0;
          for (let j = start; j < end; j++) {
            sumSquares += channelData[j] * channelData[j];
          }
          const rms = Math.sqrt(sumSquares / (end - start));
          samples.push(rms);
          maxRms = Math.max(maxRms, rms);
        }

        // Normalize samples
        const normalizedSamples = samples.map(s => maxRms > 0 ? s / maxRms : 0);

        setWaveformData(normalizedSamples);
        setDuration(audioBuffer.duration);
        setIsLoading(false);

      } catch (error) {
        console.error('Error loading audio for waveform:', error);
        setIsLoading(false);
      }
    };

    loadAudio();

    return () => {
      cancelled = true;
    };
  }, [audioUrl]);

  // Draw waveform on canvas
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.offsetWidth;
    const canvasHeight = canvas.offsetHeight;

    canvas.width = width;
    canvas.height = canvasHeight;

    ctx.clearRect(0, 0, width, canvasHeight);

    // Draw center line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, canvasHeight / 2);
    ctx.lineTo(width, canvasHeight / 2);
    ctx.stroke();

    if (!waveformData || waveformData.length === 0) {
      // Draw placeholder bars
      const barSpacing = 3;
      const barWidth = 2;
      const numBars = Math.floor(width / (barWidth + barSpacing));
      const midY = canvasHeight / 2;

      ctx.fillStyle = 'rgba(102, 126, 234, 0.2)';

      for (let i = 0; i < numBars; i++) {
        const x = i * (barWidth + barSpacing);
        const barHeight = Math.random() * 10 + 2;

        ctx.fillRect(x, midY - barHeight, barWidth, barHeight * 2);
      }
      return;
    }

    // Draw actual waveform
    const barSpacing = 3;
    const barWidth = 2;
    const midY = canvasHeight / 2;
    const maxBarHeight = canvasHeight * 0.45;

    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = barWidth;
    ctx.lineCap = 'round';
    ctx.globalAlpha = 0.8;

    const numBars = Math.floor(width / (barWidth + barSpacing));

    for (let i = 0; i < numBars; i++) {
      const sampleIndex = Math.floor(i * waveformData.length / numBars);
      const sample = waveformData[Math.min(sampleIndex, waveformData.length - 1)] || 0;

      const x = i * (barWidth + barSpacing);
      const barHeight = Math.max(2, sample * maxBarHeight);

      ctx.beginPath();
      ctx.moveTo(x + barWidth / 2, midY - barHeight);
      ctx.lineTo(x + barWidth / 2, midY + barHeight);
      ctx.stroke();
    }

    ctx.globalAlpha = 1.0;
  }, [waveformData, color]);

  // Format duration
  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className={styles.toolWaveformContainer}>
      <div className={styles.toolWaveformWrapper}>
        {isLoading ? (
          <div className={styles.waveformLoading}>
            <i className="fa-solid fa-spinner fa-spin"></i>
            <span>Loading audio...</span>
          </div>
        ) : !audioUrl ? (
          <div className={styles.waveformEmpty}>
            <i className="fa-solid fa-waveform-lines"></i>
            <span>Generated audio will appear here</span>
          </div>
        ) : null}
        <canvas ref={canvasRef} className={styles.toolWaveformCanvas} />
      </div>
      {duration > 0 && (
        <div className={styles.waveformDuration}>
          <span>0:00</span>
          <span>{formatDuration(duration)}</span>
        </div>
      )}
    </div>
  );
};

export default ToolWaveform;
