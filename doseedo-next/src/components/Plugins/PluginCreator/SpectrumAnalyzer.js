/**
 * SpectrumAnalyzer - FabFilter Pro-Q style spectrum display
 * Real-time FFT visualization with WebAudio integration
 */

import React, { useRef, useEffect, useCallback, useState, useMemo } from 'react';

const SpectrumAnalyzer = ({
  width = 400,
  height = 200,
  audioContext = null,
  sourceNode = null,
  fftSize = 2048,
  smoothing = 0.8,
  minDb = -90,
  maxDb = -10,
  minFreq = 20,
  maxFreq = 20000,
  color = '#667eea',
  backgroundColor = 'rgba(0,0,0,0.4)',
  gridColor = 'rgba(255,255,255,0.1)',
  showGrid = true,
  showFreqLabels = true,
  showDbLabels = true,
  fillOpacity = 0.3,
  lineWidth = 2,
  borderRadius = 8,
}) => {
  const canvasRef = useRef(null);
  const analyserRef = useRef(null);
  const dataArrayRef = useRef(null);
  const animationRef = useRef(null);
  const [isActive, setIsActive] = useState(false);

  // Frequency labels (logarithmic scale)
  const freqLabels = useMemo(() => [20, 50, 100, 200, 500, '1k', '2k', '5k', '10k', '20k'], []);
  const freqValues = useMemo(() => [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], []);

  // dB labels
  const dbLabels = useMemo(() => {
    const labels = [];
    for (let db = maxDb; db >= minDb; db -= 20) {
      labels.push(db);
    }
    return labels;
  }, [minDb, maxDb]);

  // Map frequency to X position (logarithmic)
  const freqToX = useCallback((freq) => {
    const logMin = Math.log10(minFreq);
    const logMax = Math.log10(maxFreq);
    const logFreq = Math.log10(Math.max(minFreq, Math.min(maxFreq, freq)));
    return ((logFreq - logMin) / (logMax - logMin)) * width;
  }, [minFreq, maxFreq, width]);

  // Map dB to Y position
  const dbToY = useCallback((db) => {
    const clampedDb = Math.max(minDb, Math.min(maxDb, db));
    return height - ((clampedDb - minDb) / (maxDb - minDb)) * height;
  }, [minDb, maxDb, height]);

  // Setup analyser when audio context is available
  useEffect(() => {
    if (!audioContext || !sourceNode) {
      setIsActive(false);
      return;
    }

    try {
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = fftSize;
      analyser.smoothingTimeConstant = smoothing;
      analyser.minDecibels = minDb;
      analyser.maxDecibels = maxDb;

      sourceNode.connect(analyser);
      analyserRef.current = analyser;
      dataArrayRef.current = new Float32Array(analyser.frequencyBinCount);
      setIsActive(true);

      return () => {
        try {
          sourceNode.disconnect(analyser);
        } catch (e) {}
      };
    } catch (e) {
      console.warn('SpectrumAnalyzer setup failed:', e);
      setIsActive(false);
    }
  }, [audioContext, sourceNode, fftSize, smoothing, minDb, maxDb]);

  // Animation loop
  useEffect(() => {
    if (!isActive || !canvasRef.current || !analyserRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const analyser = analyserRef.current;
    const dataArray = dataArrayRef.current;
    const binCount = analyser.frequencyBinCount;
    const nyquist = audioContext.sampleRate / 2;

    const draw = () => {
      analyser.getFloatFrequencyData(dataArray);

      // Clear canvas
      ctx.fillStyle = backgroundColor;
      ctx.fillRect(0, 0, width, height);

      // Draw grid
      if (showGrid) {
        ctx.strokeStyle = gridColor;
        ctx.lineWidth = 1;

        // Frequency grid lines
        freqValues.forEach(freq => {
          const x = freqToX(freq);
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, height);
          ctx.stroke();
        });

        // dB grid lines
        dbLabels.forEach(db => {
          const y = dbToY(db);
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(width, y);
          ctx.stroke();
        });
      }

      // Draw spectrum
      ctx.beginPath();
      ctx.moveTo(0, height);

      for (let i = 0; i < binCount; i++) {
        const freq = (i / binCount) * nyquist;
        if (freq < minFreq) continue;
        if (freq > maxFreq) break;

        const x = freqToX(freq);
        const db = dataArray[i];
        const y = dbToY(db);

        if (i === 0 || freq < minFreq + 1) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      // Fill under curve
      ctx.lineTo(width, height);
      ctx.lineTo(0, height);
      ctx.closePath();
      ctx.fillStyle = color.replace(')', `, ${fillOpacity})`).replace('rgb', 'rgba');
      ctx.fill();

      // Draw line on top
      ctx.beginPath();
      for (let i = 0; i < binCount; i++) {
        const freq = (i / binCount) * nyquist;
        if (freq < minFreq) continue;
        if (freq > maxFreq) break;

        const x = freqToX(freq);
        const db = dataArray[i];
        const y = dbToY(db);

        if (i === 0 || freq < minFreq + 1) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.stroke();

      animationRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive, width, height, color, backgroundColor, gridColor, showGrid, 
      freqToX, dbToY, freqValues, dbLabels, fillOpacity, lineWidth, minFreq, maxFreq]);

  // Draw static placeholder when inactive
  useEffect(() => {
    if (isActive || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    if (showGrid) {
      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 1;

      freqValues.forEach(freq => {
        const x = freqToX(freq);
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      });

      dbLabels.forEach(db => {
        const y = dbToY(db);
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      });
    }

    // Placeholder text
    ctx.fillStyle = `${color}66`;
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No audio input', width / 2, height / 2);
  }, [isActive, width, height, backgroundColor, gridColor, showGrid, 
      freqToX, dbToY, freqValues, dbLabels, color]);

  return (
    <div style={{ position: 'relative', borderRadius, overflow: 'hidden' }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ display: 'block' }}
      />
      
      {/* Frequency labels */}
      {showFreqLabels && (
        <div style={{
          position: 'absolute',
          bottom: 2,
          left: 0,
          right: 0,
          display: 'flex',
          justifyContent: 'space-between',
          padding: '0 4px',
          fontSize: 9,
          color: `${color}aa`,
          pointerEvents: 'none',
        }}>
          {freqLabels.map((label, i) => (
            <span key={i} style={{ transform: `translateX(${freqToX(freqValues[i]) - width/2}px)` }}>
              {label}
            </span>
          ))}
        </div>
      )}

      {/* dB labels */}
      {showDbLabels && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 4,
          bottom: 0,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          fontSize: 9,
          color: `${color}aa`,
          pointerEvents: 'none',
        }}>
          {dbLabels.map((db, i) => (
            <span key={i}>{db}</span>
          ))}
        </div>
      )}
    </div>
  );
};

export default SpectrumAnalyzer;
