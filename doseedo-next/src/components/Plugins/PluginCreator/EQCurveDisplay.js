/**
 * EQCurveDisplay - FabFilter Pro-Q style EQ visualization
 * Draggable bands on frequency spectrum
 */

import React, { useRef, useEffect, useCallback, useState } from 'react';

const EQCurveDisplay = ({
  width = 500,
  height = 200,
  bands = [
    { id: 'band1', freq: 100, gain: 0, q: 1, type: 'lowshelf', enabled: true },
    { id: 'band2', freq: 1000, gain: 0, q: 1, type: 'peaking', enabled: true },
    { id: 'band3', freq: 8000, gain: 0, q: 1, type: 'highshelf', enabled: true },
  ],
  onChange,
  color = '#3498db',
  backgroundColor = 'rgba(0,0,0,0.4)',
  gridColor = 'rgba(255,255,255,0.1)',
  minFreq = 20,
  maxFreq = 20000,
  minGain = -18,
  maxGain = 18,
  borderRadius = 8,
}) => {
  const canvasRef = useRef(null);
  const [dragging, setDragging] = useState(null);
  const [hoveredBand, setHoveredBand] = useState(null);

  // Log frequency to X
  const freqToX = useCallback((freq) => {
    const logMin = Math.log10(minFreq);
    const logMax = Math.log10(maxFreq);
    return ((Math.log10(freq) - logMin) / (logMax - logMin)) * width;
  }, [minFreq, maxFreq, width]);

  // X to log frequency
  const xToFreq = useCallback((x) => {
    const logMin = Math.log10(minFreq);
    const logMax = Math.log10(maxFreq);
    const logFreq = logMin + (x / width) * (logMax - logMin);
    return Math.pow(10, logFreq);
  }, [minFreq, maxFreq, width]);

  // Gain to Y
  const gainToY = useCallback((gain) => {
    return ((maxGain - gain) / (maxGain - minGain)) * height;
  }, [minGain, maxGain, height]);

  // Y to gain
  const yToGain = useCallback((y) => {
    return maxGain - (y / height) * (maxGain - minGain);
  }, [minGain, maxGain, height]);

  // Calculate EQ curve response
  const calculateResponse = useCallback((freq) => {
    let totalGain = 0;
    bands.forEach(band => {
      if (!band.enabled) return;
      const f0 = band.freq;
      const Q = band.q || 1;
      const gain = band.gain;
      const ratio = freq / f0;
      
      switch (band.type) {
        case 'lowshelf':
          if (freq < f0) totalGain += gain * (1 - Math.pow(ratio, 2));
          break;
        case 'highshelf':
          if (freq > f0) totalGain += gain * (1 - Math.pow(1/ratio, 2));
          break;
        case 'peaking':
        default:
          const bw = f0 / Q;
          const response = 1 / (1 + Math.pow((freq - f0) / (bw/2), 2));
          totalGain += gain * response;
      }
    });
    return Math.max(minGain, Math.min(maxGain, totalGain));
  }, [bands, minGain, maxGain]);

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Clear
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    // Grid
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    
    // Frequency lines
    [100, 1000, 10000].forEach(f => {
      const x = freqToX(f);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    });
    
    // Gain lines
    [-12, -6, 0, 6, 12].forEach(g => {
      const y = gainToY(g);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    });

    // Zero line
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    const zeroY = gainToY(0);
    ctx.beginPath();
    ctx.moveTo(0, zeroY);
    ctx.lineTo(width, zeroY);
    ctx.stroke();

    // Draw curve
    ctx.beginPath();
    ctx.moveTo(0, gainToY(calculateResponse(minFreq)));
    for (let x = 0; x <= width; x += 2) {
      const freq = xToFreq(x);
      const gain = calculateResponse(freq);
      ctx.lineTo(x, gainToY(gain));
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Fill under curve
    ctx.lineTo(width, zeroY);
    ctx.lineTo(0, zeroY);
    ctx.closePath();
    ctx.fillStyle = `${color}22`;
    ctx.fill();

    // Draw band points
    bands.forEach((band, i) => {
      if (!band.enabled) return;
      const x = freqToX(band.freq);
      const y = gainToY(band.gain);
      const isHovered = hoveredBand === i;
      const isDragging = dragging === i;

      // Point
      ctx.beginPath();
      ctx.arc(x, y, isHovered || isDragging ? 10 : 8, 0, Math.PI * 2);
      ctx.fillStyle = isDragging ? '#fff' : color;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      ctx.fillStyle = '#fff';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`${i + 1}`, x, y + 4);
    });

    // Labels
    ctx.fillStyle = `${color}88`;
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('100', freqToX(100), height - 5);
    ctx.fillText('1k', freqToX(1000), height - 5);
    ctx.fillText('10k', freqToX(10000), height - 5);

  }, [bands, width, height, backgroundColor, gridColor, color, freqToX, gainToY, calculateResponse, xToFreq, hoveredBand, dragging, minFreq]);

  // Mouse handlers
  const findBandNear = useCallback((x, y) => {
    for (let i = 0; i < bands.length; i++) {
      const bx = freqToX(bands[i].freq);
      const by = gainToY(bands[i].gain);
      if (Math.sqrt((x - bx) ** 2 + (y - by) ** 2) < 15) return i;
    }
    return null;
  }, [bands, freqToX, gainToY]);

  const handleMouseDown = useCallback((e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const band = findBandNear(x, y);
    if (band !== null) setDragging(band);
  }, [findBandNear]);

  const handleMouseMove = useCallback((e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (dragging !== null) {
      const newBands = [...bands];
      newBands[dragging] = {
        ...newBands[dragging],
        freq: Math.max(minFreq, Math.min(maxFreq, xToFreq(x))),
        gain: Math.max(minGain, Math.min(maxGain, yToGain(y))),
      };
      onChange?.(newBands);
    } else {
      setHoveredBand(findBandNear(x, y));
    }
  }, [dragging, bands, minFreq, maxFreq, minGain, maxGain, xToFreq, yToGain, onChange, findBandNear]);

  const handleMouseUp = () => setDragging(null);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ borderRadius, cursor: dragging !== null ? 'grabbing' : hoveredBand !== null ? 'grab' : 'crosshair' }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    />
  );
};

export default EQCurveDisplay;
