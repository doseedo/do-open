import React, { useRef, useEffect, useMemo } from 'react';
import { useThemeColor } from '../../hooks/useThemeColor';

/**
 * MIDITrackVisualization - Displays MIDI notes as a piano roll or F0 contour
 * @param {Object} midiData - MIDI data with notes array and optional voiceColors
 * @param {Array} midiData.voiceColors - Optional array of colors for multi-voice visualization
 * @param {Array} f0Contour - Optional F0 contour data for pitch visualization
 */
const MIDITrackVisualization = ({ midiData, width, height, pixelsPerSecond, startTime, endTime, timelineBpm = 120, f0Contour }) => {
  const canvasRef = useRef(null);

  // Get theme colors
  const bgDark = useThemeColor('--color-bg-dark', '#0f0f1a');
  const bgMedium = useThemeColor('--color-bg-medium', '#12121f');
  const borderLight = useThemeColor('--color-border-light', '#1a1a2e');
  const borderMedium = useThemeColor('--color-bg-lighter', '#2a2a4e');
  const primaryBlue = useThemeColor('--color-primary-blue', '#667eea');

  // Convert hex to HSL for note coloring
  const hexToHsl = (hex) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!result) return { h: 260, s: 70, l: 50 };

    let r = parseInt(result[1], 16) / 255;
    let g = parseInt(result[2], 16) / 255;
    let b = parseInt(result[3], 16) / 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h, s, l = (max + min) / 2;

    if (max === min) {
      h = s = 0;
    } else {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      switch (max) {
        case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
        case g: h = ((b - r) / d + 2) / 6; break;
        case b: h = ((r - g) / d + 4) / 6; break;
        default: h = 0;
      }
    }

    return { h: Math.round(h * 360), s: Math.round(s * 100), l: Math.round(l * 100) };
  };

  const themeHsl = useMemo(() => hexToHsl(primaryBlue), [primaryBlue]);

  // No tempo scaling needed - notes are already converted to timeline BPM
  const visibleNotes = useMemo(() => {
    if (!midiData?.notes) return [];

    return midiData.notes.filter(note => {
      const noteEnd = note.time + note.duration;
      return noteEnd >= startTime && note.time <= endTime;
    });
  }, [midiData, startTime, endTime]);

  // Find note range for display
  const noteRange = useMemo(() => {
    if (visibleNotes.length === 0) return { min: 60, max: 72 }; // Default octave

    let min = 127;
    let max = 0;

    visibleNotes.forEach(note => {
      if (note.note < min) min = note.note;
      if (note.note > max) max = note.note;
    });

    // Add padding
    min = Math.max(0, min - 2);
    max = Math.min(127, max + 2);

    return { min, max };
  }, [visibleNotes]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const noteHeightPixels = height / (noteRange.max - noteRange.min + 1);

    // Clear canvas with theme background
    ctx.fillStyle = bgDark;
    ctx.fillRect(0, 0, width, height);

    // Draw F0 contour if available
    if (f0Contour && f0Contour.length > 0) {
      ctx.strokeStyle = primaryBlue;
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.beginPath();
      let firstPoint = true;

      f0Contour.forEach((point, idx) => {
        // Skip points outside visible range
        if (point.time < startTime || point.time > endTime) return;

        const x = (point.time - startTime) * pixelsPerSecond;
        const y = height - ((point.note - noteRange.min + 1) * noteHeightPixels);

        if (firstPoint) {
          ctx.moveTo(x, y);
          firstPoint = false;
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();

      // Draw control points
      ctx.fillStyle = primaryBlue;
      f0Contour.forEach((point) => {
        if (point.time < startTime || point.time > endTime) return;

        const x = (point.time - startTime) * pixelsPerSecond;
        const y = height - ((point.note - noteRange.min + 1) * noteHeightPixels);

        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Draw MIDI notes (no scaling needed - already at timeline BPM)
    visibleNotes.forEach((note, idx) => {
      const x = (note.time - startTime) * pixelsPerSecond;
      const noteWidth = note.duration * pixelsPerSecond;
      const y = height - ((note.note - noteRange.min + 1) * noteHeightPixels);

      // Determine color - use voice color if available, otherwise use velocity-based color with theme
      let hue, saturation, lightness;
      const velocityRatio = note.velocity / 127;

      if (midiData.voiceColors && typeof note.voiceIndex !== 'undefined') {
        // Multi-voice mode: use distinct color for each voice
        const voiceColor = midiData.voiceColors[note.voiceIndex];
        if (voiceColor) {
          hue = voiceColor.hue;
          saturation = voiceColor.saturation;
          lightness = voiceColor.lightness + (velocityRatio * 10); // Slight variation by velocity
        } else {
          // Fallback to theme color
          hue = themeHsl.h;
          saturation = themeHsl.s;
          lightness = 40 + (velocityRatio * 20);
        }
      } else {
        // Single voice mode: use theme color with velocity variation
        hue = themeHsl.h;
        saturation = Math.min(100, themeHsl.s + (velocityRatio * 30));
        lightness = 40 + (velocityRatio * 20);
      }

      ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
      ctx.fillRect(x, y, Math.max(noteWidth, 2), noteHeightPixels - 1);

      // Add border for clarity
      ctx.strokeStyle = `hsl(${hue}, ${saturation}%, ${lightness + 10}%)`;
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, Math.max(noteWidth, 2), noteHeightPixels - 1);
    });

  }, [visibleNotes, width, height, pixelsPerSecond, startTime, endTime, noteRange, midiData, bgDark, bgMedium, borderLight, borderMedium, themeHsl, f0Contour, primaryBlue]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        display: 'block',
        width: `${width}px`,
        height: `${height}px`,
        imageRendering: 'crisp-edges'
      }}
    />
  );
};

export default MIDITrackVisualization;
