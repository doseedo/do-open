import React, { useRef, useEffect, useState } from 'react';
import { useApp } from '../../context/AppContext';
import styles from './AudioWaveform.module.css';

/**
 * AudioWaveform - Display waveform of selected audio track
 */
const AudioWaveform = () => {
  const { state } = useApp();
  const canvasRef = useRef(null);
  const [waveformData, setWaveformData] = useState(null);
  const [trackName, setTrackName] = useState('');

  // Get selected track
  const selectedTrack = state.selectedTrack;
  // Audio tracks have audioUrl property (not type === 'audio')
  const isAudioTrack = selectedTrack && selectedTrack.audioUrl;

  // Load and analyze audio when track is selected
  useEffect(() => {
    if (!isAudioTrack) {
      setWaveformData(null);
      setTrackName('');
      return;
    }

    setTrackName(selectedTrack.name || 'Audio Track');
    const audioUrl = selectedTrack.audioUrl;

    const loadAudio = async () => {
      try {
        console.log('🎵 Loading audio for waveform:', audioUrl);

        // Fetch audio file
        const response = await fetch(audioUrl);
        const arrayBuffer = await response.arrayBuffer();

        // Decode audio
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        // Get channel data (use first channel)
        const channelData = audioBuffer.getChannelData(0);
        const sampleRate = audioBuffer.sampleRate;
        const duration = audioBuffer.duration;

        // Calculate simplified waveform bars (RMS values for chunks)
        const numBars = 150; // Number of vertical bars
        const samplesPerBar = Math.floor(channelData.length / numBars);
        const samples = [];

        for (let i = 0; i < numBars; i++) {
          const start = i * samplesPerBar;
          const end = Math.min(start + samplesPerBar, channelData.length);

          // Calculate RMS (Root Mean Square) for this chunk
          let sumSquares = 0;
          for (let j = start; j < end; j++) {
            sumSquares += channelData[j] * channelData[j];
          }
          const rms = Math.sqrt(sumSquares / (end - start));

          samples.push({ rms });
        }

        setWaveformData({ samples, duration, sampleRate });
        console.log(`✅ Waveform loaded: ${samples.length} samples, ${duration.toFixed(2)}s`);

      } catch (error) {
        console.error('❌ Error loading audio for waveform:', error);
        setWaveformData(null);
      }
    };

    loadAudio();
  }, [isAudioTrack, selectedTrack]);

  // Draw waveform on canvas
  useEffect(() => {
    if (!waveformData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { samples, duration } = waveformData;

    // Set canvas size
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;

    // Clear canvas with transparent background
    ctx.clearRect(0, 0, width, height);

    // Draw simplified waveform with vertical rounded bars
    const midY = height / 2;
    const maxBarHeight = height * 0.45; // Maximum bar height (90% of half-height)

    // Calculate bar width and spacing
    const totalWidth = width;
    const barSpacing = 3; // Space between bars
    const barWidth = Math.max(2, (totalWidth / samples.length) - barSpacing);

    samples.forEach((sample, i) => {
      const x = i * (barWidth + barSpacing);

      // Calculate bar height from RMS (amplify for visibility)
      const barHeight = Math.max(2, sample.rms * maxBarHeight * 2.5);

      // Draw rounded bar (vertical line with circular caps)
      const barTop = midY - barHeight;
      const barBottom = midY + barHeight;

      // Bright white waveform
      ctx.fillStyle = '#f5f5f5';
      ctx.strokeStyle = '#f5f5f5';
      ctx.lineWidth = barWidth;
      ctx.lineCap = 'round'; // Rounded caps for smooth look

      ctx.beginPath();
      ctx.moveTo(x + barWidth / 2, barTop);
      ctx.lineTo(x + barWidth / 2, barBottom);
      ctx.stroke();
    });

    // Draw playhead position indicator
    const playheadPosition = state.playheadPosition || 0; // In seconds
    if (playheadPosition >= 0 && playheadPosition <= duration) {
      const playheadX = (playheadPosition / duration) * width;

      // Draw playhead line
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.9)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.stroke();

      // Draw playhead triangle at top
      ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX - 6, 10);
      ctx.lineTo(playheadX + 6, 10);
      ctx.closePath();
      ctx.fill();
    }

    // Draw duration markers
    ctx.fillStyle = '#888';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('0:00', 10, height - 10);
    ctx.textAlign = 'right';
    const minutes = Math.floor(duration / 60);
    const seconds = Math.floor(duration % 60);
    ctx.fillText(`${minutes}:${seconds.toString().padStart(2, '0')}`, width - 10, height - 10);

  }, [waveformData, state.playheadPosition]);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <i className="fa-solid fa-waveform-lines"></i>
        <span className={styles.title}>
          {isAudioTrack ? trackName : 'No Audio Track Selected'}
        </span>
      </div>

      <div className={styles.waveformWrapper}>
        {!isAudioTrack ? (
          <div className={styles.emptyState}>
            <i className="fa-solid fa-circle-info"></i>
            <p>Select an audio track to view its waveform</p>
          </div>
        ) : !waveformData ? (
          <div className={styles.loadingState}>
            <i className="fa-solid fa-spinner fa-spin"></i>
            <p>Loading waveform...</p>
          </div>
        ) : (
          <canvas ref={canvasRef} className={styles.canvas}></canvas>
        )}
      </div>

      {waveformData && (
        <div className={styles.info}>
          <div className={styles.infoItem}>
            <i className="fa-solid fa-clock"></i>
            <span>Duration: {waveformData.duration.toFixed(2)}s</span>
          </div>
          <div className={styles.infoItem}>
            <i className="fa-solid fa-chart-simple"></i>
            <span>Sample Rate: {waveformData.sampleRate} Hz</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default AudioWaveform;
