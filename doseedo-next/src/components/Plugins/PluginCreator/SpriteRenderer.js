/**
 * Sprite Renderer components — render AI-generated raster images
 * with professional audio-plugin-style indicators.
 *
 * Knob rendering: FabFilter / Vital inspired
 * - Configurable layer rotation (inner cap + outer ring)
 * - SVG arc indicator shows value (accent color + glow)
 * - Thin radial line for precise position reading
 * - Subtle track arc shows full range
 */

import React, { useRef, useCallback, useState } from 'react';

// Map 0-1 value to knob angle in math coords (works with cos/sin in SVG's y-down system)
// v=0 → 7:30 position, v=0.5 → 12 o'clock, v=1 → 4:30 position
const valueToAngle = (v) => -225 + (v ?? 0.65) * 270;

const RANGE_DEG = 270;
// SVG rotation offset: rotates the stroke-dasharray start from 3 o'clock to 7:30
const ARC_ROTATE = 135;

// ═══════════════════════════════════════════════════════════════════════
// SPRITE KNOB RENDERER — Layered body + arc indicator
// ═══════════════════════════════════════════════════════════════════════

export const SpriteKnobRenderer = React.memo(({
  spriteUrl, flatRingUrl, size, value, isTestMode, onChange, indicatorColor,
  innerRotates = false, outerRotates = true,
  innerRotateSpeed = 1, outerRotateSpeed = 1,
}) => {
  const [hovered, setHovered] = useState(false);
  const v = value ?? 0.65;

  // -- Proportions --
  const arcStroke = Math.max(2.5, size * 0.055);
  const gap = Math.max(1.5, size * 0.03);
  const padding = 1;
  const cx = size / 2;
  const cy = size / 2;
  const arcRadius = cx - arcStroke / 2 - padding;
  const bodyRadius = arcRadius - arcStroke / 2 - gap;

  // -- Arc math --
  const circumference = 2 * Math.PI * arcRadius;
  const arcLength = circumference * (RANGE_DEG / 360);
  const valueArcLength = v * arcLength;

  // -- Indicator line (always tracks actual value) --
  const angle = valueToAngle(v);
  const angleRad = (angle * Math.PI) / 180;
  const lineInner = bodyRadius * 0.55;
  const lineOuter = bodyRadius * 0.92;
  const lx1 = cx + lineInner * Math.cos(angleRad);
  const ly1 = cy + lineInner * Math.sin(angleRad);
  const lx2 = cx + lineOuter * Math.cos(angleRad);
  const ly2 = cy + lineOuter * Math.sin(angleRad);

  // -- Colors --
  const accent = indicatorColor || '#53d769';
  const glowPx = Math.max(2, size * 0.06);
  const glowAlpha = hovered ? 0.6 : 0.35;

  const handleMouseDown = useCallback((e) => {
    if (!isTestMode || !onChange) return;
    e.preventDefault();
    const startY = e.clientY;
    const startVal = v;

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
  }, [isTestMode, onChange, v]);

  const hexToRgba = (hex, a) => {
    const h = hex.replace('#', '');
    const r = parseInt(h.substring(0, 2), 16) || 83;
    const g = parseInt(h.substring(2, 4), 16) || 215;
    const b = parseInt(h.substring(4, 6), 16) || 105;
    return `rgba(${r},${g},${b},${a})`;
  };

  // If both layers rotate the same way, render single image (no mask split needed)
  const bothRotate = innerRotates && outerRotates;
  const neitherRotate = !innerRotates && !outerRotates;
  const useLayerSplit = !bothRotate && !neitherRotate;

  const rotateStyle = (rotates, speed = 1) => {
    if (!rotates) return {};
    const layerAngle = angle * speed;
    return { transform: `rotate(${layerAngle}deg)`, transition: isTestMode ? 'none' : 'transform 0.15s ease' };
  };

  return (
    <div
      style={{ width: size, height: size, position: 'relative', cursor: isTestMode ? 'ns-resize' : 'default' }}
      onMouseDown={handleMouseDown}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {useLayerSplit ? (
        <>
          {/* Layer A: Outer ring */}
          <div style={{
            position: 'absolute',
            top: cx - bodyRadius, left: cy - bodyRadius,
            width: bodyRadius * 2, height: bodyRadius * 2,
            borderRadius: '50%', overflow: 'hidden',
          }}>
            <div style={{
              width: '100%', height: '100%',
              WebkitMaskImage: 'radial-gradient(circle, transparent 54%, black 58%)',
              maskImage: 'radial-gradient(circle, transparent 54%, black 58%)',
              ...rotateStyle(outerRotates, outerRotateSpeed),
            }}>
              <img src={outerRotates ? (flatRingUrl || spriteUrl) : spriteUrl} alt="" draggable={false} style={{
                width: '100%', height: '100%', objectFit: 'cover', pointerEvents: 'none',
              }} />
            </div>
          </div>

          {/* Static highlight overlay on rotating ring */}
          {outerRotates && flatRingUrl && (
            <div style={{
              position: 'absolute',
              top: cx - bodyRadius, left: cy - bodyRadius,
              width: bodyRadius * 2, height: bodyRadius * 2,
              borderRadius: '50%', overflow: 'hidden', pointerEvents: 'none',
              WebkitMaskImage: 'radial-gradient(circle, transparent 54%, black 58%)',
              maskImage: 'radial-gradient(circle, transparent 54%, black 58%)',
              background: 'radial-gradient(circle at 38% 35%, rgba(255,255,255,0.18) 0%, transparent 55%), radial-gradient(circle at 65% 70%, rgba(0,0,0,0.12) 0%, transparent 50%)',
            }} />
          )}

          {/* Layer B: Inner cap */}
          <div style={{
            position: 'absolute',
            top: cx - bodyRadius, left: cy - bodyRadius,
            width: bodyRadius * 2, height: bodyRadius * 2,
            borderRadius: '50%', overflow: 'hidden',
            boxShadow: '0 1px 3px rgba(0,0,0,0.4)',
          }}>
            <div style={{
              width: '100%', height: '100%',
              WebkitMaskImage: 'radial-gradient(circle, black 54%, transparent 58%)',
              maskImage: 'radial-gradient(circle, black 54%, transparent 58%)',
              ...rotateStyle(innerRotates, innerRotateSpeed),
            }}>
              <img src={spriteUrl} alt="" draggable={false} style={{
                width: '100%', height: '100%', objectFit: 'cover', pointerEvents: 'none',
              }} />
            </div>
          </div>
        </>
      ) : (
        /* Both same rotation — single layer, no mask split */
        <div style={{
          position: 'absolute',
          top: cx - bodyRadius, left: cy - bodyRadius,
          width: bodyRadius * 2, height: bodyRadius * 2,
          borderRadius: '50%', overflow: 'hidden',
          boxShadow: '0 1px 3px rgba(0,0,0,0.4)',
          ...rotateStyle(bothRotate, outerRotateSpeed),
        }}>
          <img src={spriteUrl} alt="" draggable={false} style={{
            width: '100%', height: '100%', objectFit: 'cover', pointerEvents: 'none',
          }} />
        </div>
      )}

      {/* SVG overlay: track arc, value arc, indicator line */}
      <svg
        width={size} height={size}
        viewBox={`0 0 ${size} ${size}`}
        style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
      >
        <circle
          cx={cx} cy={cy} r={arcRadius}
          fill="none" stroke="rgba(255,255,255,0.08)"
          strokeWidth={arcStroke}
          strokeDasharray={`${arcLength} ${circumference}`}
          strokeLinecap="round"
          transform={`rotate(${ARC_ROTATE}, ${cx}, ${cy})`}
        />
        {v > 0.005 && (
          <circle
            cx={cx} cy={cy} r={arcRadius}
            fill="none" stroke={accent}
            strokeWidth={arcStroke}
            strokeDasharray={`${valueArcLength} ${circumference}`}
            strokeLinecap="round"
            transform={`rotate(${ARC_ROTATE}, ${cx}, ${cy})`}
            style={{
              filter: `drop-shadow(0 0 ${glowPx}px ${hexToRgba(accent, glowAlpha)})`,
              transition: isTestMode ? 'none' : 'stroke-dasharray 0.15s ease',
            }}
          />
        )}
        <line
          x1={lx1} y1={ly1} x2={lx2} y2={ly2}
          stroke="rgba(255,255,255,0.85)"
          strokeWidth={Math.max(1.5, size * 0.03)}
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
});

// ═══════════════════════════════════════════════════════════════════════
// SPRITE SLIDER RENDERER
// ═══════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════
// SPRITE BUTTON RENDERER
// ═══════════════════════════════════════════════════════════════════════

export const SpriteButtonRenderer = React.memo(({ spriteUrl, pressedSpriteUrl, width, height, pressed }) => {
  const hasTwoStates = pressedSpriteUrl && pressedSpriteUrl !== spriteUrl;
  const currentSrc = pressed && hasTwoStates ? pressedSpriteUrl : spriteUrl;

  return (
    <div style={{
      width, height, position: 'relative', overflow: 'hidden',
      borderRadius: 4,
      transform: pressed && !hasTwoStates ? 'scale(0.97)' : 'scale(1)',
      transition: 'transform 0.1s ease',
    }}>
      <img src={currentSrc} alt="" draggable={false} style={{
        width: '100%', height: '100%', objectFit: 'contain', pointerEvents: 'none',
      }} />
      {pressed && !hasTwoStates && (
        <div style={{
          position: 'absolute', inset: 0,
          background: 'rgba(0,0,0,0.2)',
          boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.4)',
        }} />
      )}
    </div>
  );
});

// ═══════════════════════════════════════════════════════════════════════
// SPRITE SLIDER RENDERER
// ═══════════════════════════════════════════════════════════════════════

export const SpriteSliderRenderer = React.memo(({ spriteUrl, width, height, value, isTestMode, onChange }) => {
  const containerRef = useRef(null);
  const thumbSize = Math.max(16, width * 0.8);
  const trackHeight = height - thumbSize;
  const thumbTop = (1 - (value ?? 0.3)) * trackHeight;

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

  return (
    <div
      ref={containerRef}
      style={{
        width, height, position: 'relative',
        cursor: isTestMode ? 'ns-resize' : 'default',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
      onMouseDown={handleMouseDown}
    >
      <div style={{
        position: 'absolute', width: 4, height: trackHeight, top: thumbSize / 2,
        background: 'rgba(255,255,255,0.1)', borderRadius: 2,
      }} />
      <div style={{
        position: 'absolute', top: thumbTop, width: thumbSize, height: thumbSize,
        borderRadius: '50%', overflow: 'hidden',
      }}>
        <img src={spriteUrl} alt="" draggable={false} style={{
          width: '100%', height: '100%', objectFit: 'cover', pointerEvents: 'none',
        }} />
      </div>
    </div>
  );
});
