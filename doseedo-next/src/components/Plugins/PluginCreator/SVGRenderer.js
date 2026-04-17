/**
 * SVG Renderer components — render LLM-generated SVGs with interactive state.
 *
 * Convention:
 * - Knobs: <g id="rotate"> is rotated by value (0-1 → -225° to +45°)
 * - Sliders: <g id="thumb"> is translated vertically by value
 * - Buttons: <g id="normal"> shown by default, <g id="pressed"> shown on press
 * - Fallback: if no well-known id found, whole SVG is wrapped in transform
 */

import React, { useMemo, useRef, useCallback, useState } from 'react';
import { sanitizeSVG } from './svgSanitizer';

// Map 0-1 value to knob angle (-225° to +45°, 270° range)
const valueToAngle = (v) => -225 + (v ?? 0.65) * 270;

/**
 * Apply rotation to #rotate group in an SVG string.
 * Falls back to rotating a wrapper if no #rotate group exists.
 */
function applyKnobRotation(svgStr, angle, cx, cy) {
  // Try to inject transform on the #rotate group
  const rotateGroupRegex = /(<g\s[^>]*id=["']rotate["'][^>]*)(>)/i;
  const match = svgStr.match(rotateGroupRegex);
  if (match) {
    // Remove any existing transform, then add our rotation
    const tag = match[1].replace(/\s*transform=["'][^"']*["']/gi, '');
    return svgStr.replace(rotateGroupRegex, `${tag} transform="rotate(${angle}, ${cx}, ${cy})">`);
  }
  // Fallback: no #rotate group found — wrap content in a rotation
  return svgStr;
}

/**
 * Apply vertical translation to #thumb group.
 */
function applySliderThumb(svgStr, offset) {
  const thumbRegex = /(<g\s[^>]*id=["']thumb["'][^>]*)(>)/i;
  const match = svgStr.match(thumbRegex);
  if (match) {
    const tag = match[1].replace(/\s*transform=["'][^"']*["']/gi, '');
    return svgStr.replace(thumbRegex, `${tag} transform="translate(0, ${offset})">`);
  }
  return svgStr;
}

/**
 * Toggle visibility of #normal and #pressed groups.
 */
function applyButtonState(svgStr, pressed) {
  let result = svgStr;
  // Set #normal visibility
  result = result.replace(
    /(<g\s[^>]*id=["']normal["'][^>]*)(>)/i,
    (_, tag, gt) => {
      const cleaned = tag.replace(/\s*(?:display|visibility)=["'][^"']*["']/gi, '');
      return `${cleaned} display="${pressed ? 'none' : 'inline'}"${gt}`;
    }
  );
  // Set #pressed visibility
  result = result.replace(
    /(<g\s[^>]*id=["']pressed["'][^>]*)(>)/i,
    (_, tag, gt) => {
      const cleaned = tag.replace(/\s*(?:display|visibility)=["'][^"']*["']/gi, '');
      return `${cleaned} display="${pressed ? 'inline' : 'none'}"${gt}`;
    }
  );
  return result;
}

/**
 * Extract viewBox center from SVG string.
 */
function getCenter(svgStr, fallbackSize) {
  const vbMatch = svgStr.match(/viewBox=["']([^"']+)["']/);
  if (vbMatch) {
    const parts = vbMatch[1].trim().split(/\s+/).map(Number);
    if (parts.length === 4) return { cx: parts[0] + parts[2] / 2, cy: parts[1] + parts[3] / 2 };
  }
  return { cx: fallbackSize / 2, cy: fallbackSize / 2 };
}

// ═══════════════════════════════════════════════════════════════════════
// SVG KNOB RENDERER
// ═══════════════════════════════════════════════════════════════════════

export const SVGKnobRenderer = React.memo(({ svgString, size, value, color, isTestMode, onChange, knobImages }) => {
  const sanitized = useMemo(() => sanitizeSVG(svgString), [svgString]);

  const angle = valueToAngle(value);
  const { cx, cy } = useMemo(() => getCenter(sanitized || '', size), [sanitized, size]);

  const renderedSvg = useMemo(() => {
    if (!sanitized) return null;
    return applyKnobRotation(sanitized, angle, cx, cy);
  }, [sanitized, angle, cx, cy]);

  const handleMouseDown = useCallback((e) => {
    if (!isTestMode || !onChange) return;
    e.preventDefault();
    const startY = e.clientY;
    const startVal = value ?? 0.65;

    const handleMove = (ev) => {
      const dy = startY - ev.clientY;
      const newVal = Math.max(0, Math.min(1, startVal + dy / 120));
      onChange(newVal);
    };
    const handleUp = () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
  }, [isTestMode, onChange, value]);

  if (!renderedSvg) return null;

  // CSS-layer mode: knob image rotated via CSS transform, specular overlay stays fixed
  if (knobImages) {
    return (
      <div
        style={{ width: size, height: size, position: 'relative', cursor: isTestMode ? 'ns-resize' : 'default' }}
        onMouseDown={handleMouseDown}
      >
        {/* Rotating knob image */}
        <img
          src={knobImages.full}
          alt=""
          draggable={false}
          style={{
            position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
            transform: `rotate(${angle}deg)`,
            transformOrigin: 'center center',
            pointerEvents: 'none',
          }}
        />
        {/* Fixed specular highlight overlay */}
        {knobImages.specular && (
          <img
            src={knobImages.specular}
            alt=""
            draggable={false}
            style={{
              position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
              pointerEvents: 'none',
              mixBlendMode: 'screen',
            }}
          />
        )}
      </div>
    );
  }

  // SVG fallback mode (non-moog knobs)
  return (
    <div
      style={{ width: size, height: size, overflow: 'hidden', cursor: isTestMode ? 'ns-resize' : 'default' }}
      onMouseDown={handleMouseDown}
      dangerouslySetInnerHTML={{ __html: renderedSvg }}
    />
  );
});

// ═══════════════════════════════════════════════════════════════════════
// SVG SLIDER RENDERER
// ═══════════════════════════════════════════════════════════════════════

export const SVGSliderRenderer = React.memo(({ svgString, width, height, value, color, isTestMode, onChange }) => {
  const sanitized = useMemo(() => sanitizeSVG(svgString), [svgString]);
  const containerRef = useRef(null);

  // Map value 0-1 to thumb offset (1=top, 0=bottom)
  const maxTravel = height * 0.7;
  const offset = (1 - (value ?? 0.3)) * maxTravel;

  const renderedSvg = useMemo(() => {
    if (!sanitized) return null;
    return applySliderThumb(sanitized, offset);
  }, [sanitized, offset]);

  const handleMouseDown = useCallback((e) => {
    if (!isTestMode || !onChange) return;
    e.preventDefault();
    const startY = e.clientY;
    const startVal = value ?? 0.3;

    const handleMove = (ev) => {
      const dy = startY - ev.clientY;
      const newVal = Math.max(0, Math.min(1, startVal + dy / (height - 20)));
      onChange(newVal);
    };
    const handleUp = () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
  }, [isTestMode, onChange, value, height]);

  if (!renderedSvg) return null;

  return (
    <div
      ref={containerRef}
      style={{ width, height, overflow: 'hidden', cursor: isTestMode ? 'ns-resize' : 'default' }}
      onMouseDown={handleMouseDown}
      dangerouslySetInnerHTML={{ __html: renderedSvg }}
    />
  );
});

// ═══════════════════════════════════════════════════════════════════════
// SVG BUTTON RENDERER
// ═══════════════════════════════════════════════════════════════════════

export const SVGButtonRenderer = React.memo(({ svgString, width, height, pressed, color }) => {
  const sanitized = useMemo(() => sanitizeSVG(svgString), [svgString]);

  const renderedSvg = useMemo(() => {
    if (!sanitized) return null;
    return applyButtonState(sanitized, pressed);
  }, [sanitized, pressed]);

  if (!renderedSvg) return null;

  return (
    <div
      style={{ width, height, overflow: 'hidden' }}
      dangerouslySetInnerHTML={{ __html: renderedSvg }}
    />
  );
});

// ═══════════════════════════════════════════════════════════════════════
// SVG STATIC RENDERER (meters, waveforms, panels, labels, etc.)
// ═══════════════════════════════════════════════════════════════════════

export const SVGStaticRenderer = React.memo(({ svgString, width, height }) => {
  const sanitized = useMemo(() => sanitizeSVG(svgString), [svgString]);

  if (!sanitized) return null;

  return (
    <div
      style={{ width, height, overflow: 'hidden' }}
      dangerouslySetInnerHTML={{ __html: sanitized }}
    />
  );
});
