import React, { useEffect, useRef } from 'react';
import { fetchAudioWithCache } from '../services/audioCacheService';

// Global in-memory cache for decoded audio buffers (faster than re-decoding).
// EXPORTED so other services (e.g. latent-based waveform previews) can
// pre-populate it before a track's real audio is ready.
export const audioBufferCache = new Map();
const MAX_CACHE_SIZE = 50; // Limit in-memory cache size

/**
 * Continuously animate random noise bars on the canvas (RAF loop).
 * Each bar smoothly lerps toward a new random target each frame.
 * The loop runs until transitionFrameRef is cancelled (e.g. by
 * paintBarAnimationFromBuffer or paintBarAnimation when audio arrives).
 */
function startNoiseAnimation(canvas, width, height, color, transitionFrameRef) {
  if (!canvas) return;
  if (transitionFrameRef.current) cancelAnimationFrame(transitionFrameRef.current);

  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const barSpacing = 2;
  const barWidth = 2;
  const numBars = Math.floor(width / (barWidth + barSpacing));
  const midY = height / 2;
  const maxBarHeight = height * 0.45;

  // Initialise bar heights randomly
  const heights = new Float32Array(numBars);
  const targets = new Float32Array(numBars);
  for (let i = 0; i < numBars; i++) {
    heights[i] = (0.15 + Math.random() * 0.85) * maxBarHeight;
    targets[i] = (0.15 + Math.random() * 0.85) * maxBarHeight;
  }

  let frameCount = 0;
  const tick = () => {
    // Pick new random targets every ~20 frames so the noise feels alive
    frameCount++;
    if (frameCount % 20 === 0) {
      for (let i = 0; i < numBars; i++) {
        targets[i] = (0.15 + Math.random() * 0.85) * maxBarHeight;
      }
    }

    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = color;
    ctx.lineWidth = barWidth;
    ctx.lineCap = 'round';
    ctx.globalAlpha = 0.7;

    for (let i = 0; i < numBars; i++) {
      heights[i] += (targets[i] - heights[i]) * 0.08;
      const x = i * (barWidth + barSpacing) + barWidth / 2;
      ctx.beginPath();
      ctx.moveTo(x, midY - heights[i]);
      ctx.lineTo(x, midY + heights[i]);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    transitionFrameRef.current = requestAnimationFrame(tick);
  };
  transitionFrameRef.current = requestAnimationFrame(tick);
}

/**
 * Animate every bar from a random noise height to its correct waveform height
 * computed from a decoded AudioBuffer. Duration ≈550 ms.
 */
function paintBarAnimationFromBuffer(canvas, audioBuffer, width, height, color, gain, cropStart, cropEnd, transitionFrameRef) {
  if (!canvas || !audioBuffer) return;
  if (transitionFrameRef.current) cancelAnimationFrame(transitionFrameRef.current);

  const ctx = canvas.getContext('2d');
  const barSpacing = 2;
  const barWidth = 2;
  const numBars = Math.floor(width / (barWidth + barSpacing));
  const midY = height / 2;
  const maxBarHeight = height * 0.45;
  const g = Math.max(0, gain);

  const data = audioBuffer.getChannelData(0);
  const duration = audioBuffer.duration;
  const sampleRate = audioBuffer.sampleRate;
  const startSample = Math.floor(cropStart * sampleRate);
  const endSample = cropEnd > 0 ? Math.floor((duration - cropEnd) * sampleRate) : data.length;
  const visibleLength = endSample - startSample;
  const samplesPerBar = Math.floor(visibleLength / numBars);

  const ptpValues = new Float32Array(numBars);
  let maxPtp = 0;
  for (let i = 0; i < numBars; i++) {
    const s = startSample + i * samplesPerBar;
    const e = Math.min(s + samplesPerBar, endSample);
    let mx = -Infinity, mn = Infinity;
    for (let j = s; j < e && j < data.length; j++) {
      if (data[j] > mx) mx = data[j];
      if (data[j] < mn) mn = data[j];
    }
    ptpValues[i] = mx === -Infinity ? 0 : (mx - mn);
    if (ptpValues[i] > maxPtp) maxPtp = ptpValues[i];
  }
  const normFactor = maxPtp > 0.001 ? maxBarHeight / maxPtp : 1.0;
  const noiseFloor = maxPtp * 0.02;
  const targetHeights = new Float32Array(numBars);
  for (let i = 0; i < numBars; i++) {
    targetHeights[i] = (ptpValues[i] < noiseFloor || g === 0)
      ? 0.5
      : Math.min(ptpValues[i] * normFactor * g, maxBarHeight);
  }

  const startHeights = new Float32Array(numBars);
  for (let i = 0; i < numBars; i++) {
    startHeights[i] = (0.15 + Math.random() * 0.85) * maxBarHeight;
  }

  canvas.width = width;
  canvas.height = height;

  const DURATION = 550;
  const animStart = performance.now();

  const tick = (now) => {
    const t = Math.min((now - animStart) / DURATION, 1);
    const ease = 1 - Math.pow(1 - t, 2.5);

    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = color;
    ctx.lineWidth = barWidth;
    ctx.lineCap = 'round';
    ctx.globalAlpha = 0.7;

    for (let i = 0; i < numBars; i++) {
      const barH = startHeights[i] + (targetHeights[i] - startHeights[i]) * ease;
      const x = i * (barWidth + barSpacing) + barWidth / 2;
      ctx.beginPath();
      ctx.moveTo(x, midY - barH);
      ctx.lineTo(x, midY + barH);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    if (t < 1) {
      transitionFrameRef.current = requestAnimationFrame(tick);
    } else {
      transitionFrameRef.current = null;
      renderWaveform(ctx, audioBuffer, width, height, color, cropStart, cropEnd, gain);
    }
  };
  transitionFrameRef.current = requestAnimationFrame(tick);
}

/**
 * Animate every bar from a random noise height to its correct envelope height.
 * Each bar stays in the same horizontal slot — only height changes. Duration ≈550 ms.
 * Fires on first envelope paint (stem mount) and again when WAV audio decodes
 * (signalling the track is ready for playback).
 */
function paintBarAnimation(canvas, envelopeData, T, width, height, color, gain, transitionFrameRef) {
  if (!canvas) return;
  if (transitionFrameRef.current) cancelAnimationFrame(transitionFrameRef.current);

  const ctx = canvas.getContext('2d');
  const barSpacing = 2;
  const barWidth = 2;
  const numBars = Math.floor(width / (barWidth + barSpacing));
  const midY = height / 2;
  const maxBarHeight = height * 0.45;
  const g = Math.max(0, gain);

  // Pre-compute target bar heights (same peak-to-peak logic as renderEnvelope)
  const mins = envelopeData.subarray(0, T);
  const maxs = envelopeData.subarray(T, 2 * T);
  const ptpValues = new Float32Array(numBars);
  let maxPtp = 0;
  for (let i = 0; i < numBars; i++) {
    const s = Math.floor((i * T) / numBars);
    const e = Math.max(s + 1, Math.floor(((i + 1) * T) / numBars));
    let mx = -Infinity, mn = Infinity;
    for (let j = s; j < e && j < T; j++) {
      if (maxs[j] > mx) mx = maxs[j];
      if (mins[j] < mn) mn = mins[j];
    }
    ptpValues[i] = mx - mn;
    if (ptpValues[i] > maxPtp) maxPtp = ptpValues[i];
  }
  const normFactor = maxPtp > 0.001 ? maxBarHeight / maxPtp : 1.0;
  const noiseFloor = maxPtp * 0.02;
  const targetHeights = new Float32Array(numBars);
  for (let i = 0; i < numBars; i++) {
    targetHeights[i] = (ptpValues[i] < noiseFloor || g === 0)
      ? 0.5
      : Math.min(ptpValues[i] * normFactor * g, maxBarHeight);
  }

  // Start heights — random, spanning the full bar area (the "noise" state)
  const startHeights = new Float32Array(numBars);
  for (let i = 0; i < numBars; i++) {
    startHeights[i] = (0.15 + Math.random() * 0.85) * maxBarHeight;
  }

  canvas.width = width;
  canvas.height = height;

  const DURATION = 550;
  const animStart = performance.now();

  const tick = (now) => {
    const t = Math.min((now - animStart) / DURATION, 1);
    const ease = 1 - Math.pow(1 - t, 2.5); // ease-out

    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = color;
    ctx.lineWidth = barWidth;
    ctx.lineCap = 'round';
    ctx.globalAlpha = 0.7;

    for (let i = 0; i < numBars; i++) {
      const barH = startHeights[i] + (targetHeights[i] - startHeights[i]) * ease;
      const x = i * (barWidth + barSpacing) + barWidth / 2;
      ctx.beginPath();
      ctx.moveTo(x, midY - barH);
      ctx.lineTo(x, midY + barH);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    if (t < 1) {
      transitionFrameRef.current = requestAnimationFrame(tick);
    } else {
      transitionFrameRef.current = null;
      // Final pixel-perfect render at steady state
      renderEnvelope(ctx, envelopeData, T, width, height, color, gain);
    }
  };
  transitionFrameRef.current = requestAnimationFrame(tick);
}

/**
 * Custom hook for rendering audio waveforms on canvas
 * Uses IndexedDB for persistent blob caching + in-memory cache for decoded buffers
 * @param {string} audioUrl - URL of the audio file
 * @param {number} width - Canvas width
 * @param {number} height - Canvas height
 * @param {string} color - Waveform color
 * @param {number} cropStart - Crop start time in seconds
 * @param {number} cropEnd - Crop end time in seconds
 * @returns {Object} - Canvas ref and loading state
 */
/**
 * @param {string} audioUrl - URL of the audio file (or placeholder blob)
 * @param {number} width
 * @param {number} height
 * @param {string} color
 * @param {number} cropStart
 * @param {number} cropEnd
 * @param {Float32Array|null} envelopeData - optional pre-computed [2*T] envelope
 *   from latent_visual. First T values are min, next T are max. If provided,
 *   the waveform renders immediately from this without waiting for audio decode.
 * @param {number} envelopeFps - envelope frame rate (default 25)
 * @param {number} gain - visual gain multiplier (0..N, default 1.0). Scales
 *   bar heights so volume changes are reflected in the waveform in real time.
 *   Pass 0 for muted tracks / soloed-out stems.
 */
export function useWaveform(audioUrl, width = 800, height = 60, color = '#f5f5f5', cropStart = 0, cropEnd = 0, envelopeData = null, envelopeFps = 25, gain = 1.0, showNoise = false) {
  const canvasRef = useRef(null);
  const audioBufferRef = useRef(null);
  const loadedUrlRef = useRef(null);
  const envelopePaintedRef = useRef(false);
  // Set to true once any audio-based waveform is painted. Prevents subsequent
  // audio-decode events from repainting (only envelopeData changes can do so).
  const audioPaintedRef = useRef(false);
  const lastEnvelopeRef = useRef(null);
  const transitionFrameRef = useRef(null);
  const [isLoaded, setIsLoaded] = React.useState(false);
  const [duration, setDuration] = React.useState(null);
  // Increments each time a real audio decode completes (NOT on envelope-only paint).
  // OptimizedTrack watches this to know when to end the noise overlay on WAV arrival.
  const [decodedAudioUrl, setDecodedAudioUrl] = React.useState(null);

  // paintWithTransition — every time the canvas needs a new image, pass the
  // draw through this helper instead of calling `canvas.width = width` +
  // `renderXxx(...)` directly. It snapshots the current pixels, renders the
  // new content, then runs a short RAF loop that crossfades old→new with a
  // purple noise overlay (same palette as PlaceholderWaveform). Net effect:
  // the waveform never "disappears" during latent_visual handoff or final-
  // decode swap — it morphs through noise, matching the studio's visual
  // language. First paint (no prior content OR dimensions changed) skips
  // the animation since there's nothing sensible to fade FROM.
  const paintWithTransition = React.useCallback((drawFn, { animate = true, duration = 320 } = {}) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dimsMatch = canvas.width === width && canvas.height === height;
    const hadContent = dimsMatch && (envelopePaintedRef.current || audioBufferRef.current);

    if (!animate || !hadContent) {
      canvas.width = width;
      canvas.height = height;
      drawFn(ctx);
      return;
    }

    // Snapshot the current pixels before destroying them.
    const old = document.createElement('canvas');
    old.width = width;
    old.height = height;
    old.getContext('2d').drawImage(canvas, 0, 0);

    // Draw the new content freshly so we can blend it over.
    canvas.width = width;
    canvas.height = height;
    drawFn(ctx);
    const fresh = document.createElement('canvas');
    fresh.width = width;
    fresh.height = height;
    fresh.getContext('2d').drawImage(canvas, 0, 0);

    if (transitionFrameRef.current) cancelAnimationFrame(transitionFrameRef.current);
    const start = performance.now();
    const centerY = height / 2;
    // Noise bars use the same purple the placeholder animation uses
    // (components/DAW/PlaceholderWaveform.js) so the whole generation
    // pipeline shares a single "stems in flight" visual motif.
    const NOISE_COLOR = 'rgba(139, 92, 246, 0.7)';
    const tick = (now) => {
      const t = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - t, 3);            // ease-out cubic
      const noiseAlpha = Math.sin(Math.PI * t) * 0.45; // peaks at t=0.5

      ctx.clearRect(0, 0, width, height);
      ctx.globalAlpha = ease;
      ctx.drawImage(fresh, 0, 0);
      ctx.globalAlpha = 1 - ease;
      ctx.drawImage(old, 0, 0);

      if (noiseAlpha > 0.02) {
        ctx.globalAlpha = noiseAlpha;
        ctx.strokeStyle = NOISE_COLOR;
        ctx.lineWidth = 1;
        for (let x = 0; x < width; x += 3) {
          const n = (Math.random() - 0.5) * height * 0.5;
          ctx.beginPath();
          ctx.moveTo(x + 0.5, centerY - n);
          ctx.lineTo(x + 0.5, centerY + n);
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;

      if (t < 1) {
        transitionFrameRef.current = requestAnimationFrame(tick);
      } else {
        transitionFrameRef.current = null;
      }
    };
    transitionFrameRef.current = requestAnimationFrame(tick);
  }, [width, height]);

  // Cleanup: cancel any in-flight transition on unmount / dep change.
  useEffect(() => () => {
    if (transitionFrameRef.current) cancelAnimationFrame(transitionFrameRef.current);
  }, []);

  // Instant envelope rendering — draws the waveform from latent_visual
  // peaks WITHOUT waiting for audio decode. Fires whenever envelopeData
  // changes (v4 → latent_visual refinement handoff, for example).
  useEffect(() => {
    if (!envelopeData || !canvasRef.current) return;
    const T = envelopeData.length / 2;
    const dur = T / envelopeFps;
    const isNewEnvelope = envelopeData !== lastEnvelopeRef.current;
    if (isNewEnvelope) {
      // New envelope: bar-height animation (noise → shape).
      paintBarAnimation(canvasRef.current, envelopeData, T, width, height, color, gain, transitionFrameRef);
    } else {
      // Same envelope, just a dimension / gain repaint — instant, no animation.
      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = width;
        canvas.height = height;
        renderEnvelope(canvas.getContext('2d'), envelopeData, T, width, height, color, gain);
      }
    }
    lastEnvelopeRef.current = envelopeData;
    envelopePaintedRef.current = true;
    if (!isLoaded) {
      setDuration(dur);
      setIsLoaded(true);
    }
  }, [envelopeData, width, height, color, envelopeFps, isLoaded, gain]);

  // When showNoise is true (e.g. MIDI track mid-generation with no audioUrl yet),
  // paint static noise bars immediately so the canvas shows something right away.
  useEffect(() => {
    if (!showNoise || envelopePaintedRef.current || audioPaintedRef.current) return;
    if (canvasRef.current) {
      startNoiseAnimation(canvasRef.current, width, height, color, transitionFrameRef);
    }
  }, [showNoise, width, height, color]);

  // Load audio only when URL changes
  useEffect(() => {
    if (!audioUrl || loadedUrlRef.current === audioUrl) return;

    let cancelled = false;

    // New URL = new waveform; reset audio-painted guard so it re-animates.
    audioPaintedRef.current = false;

    // Immediately show noise on the canvas while we wait for audio to download.
    // This ensures there's never a blank/grey gap between URL change and decode.
    if (!envelopePaintedRef.current && canvasRef.current) {
      startNoiseAnimation(canvasRef.current, width, height, color, transitionFrameRef);
    }

    if (!envelopePaintedRef.current && !audioBufferRef.current) {
      setIsLoaded(false);
    }

    const loadAudio = async () => {
      try {
        // Check in-memory cache first (fastest)
        if (audioBufferCache.has(audioUrl)) {
          const cachedBuffer = audioBufferCache.get(audioUrl);

          if (cancelled) return;

          audioBufferRef.current = cachedBuffer;
          loadedUrlRef.current = audioUrl;
          setDuration(cachedBuffer.duration);
          setIsLoaded(true);
          setDecodedAudioUrl(audioUrl);

          if (canvasRef.current) {
            if (envelopeData) {
              // Envelope wins — noise → envelope shape
              const envT = envelopeData.length / 2;
              paintBarAnimation(canvasRef.current, envelopeData, envT, width, height, color, gain, transitionFrameRef);
              envelopePaintedRef.current = true;
            } else if (!audioPaintedRef.current) {
              // First audio paint: noise → waveform bar animation
              paintBarAnimationFromBuffer(canvasRef.current, cachedBuffer, width, height, color, gain, cropStart, cropEnd, transitionFrameRef);
              audioPaintedRef.current = true;
            }
          }
          return;
        }

        // Try IndexedDB cache (persists across refreshes) or fetch from network
        console.log(`📥 Loading audio: ${audioUrl.substring(0, 50)}...`);
        const { blob, fromCache } = await fetchAudioWithCache(audioUrl);

        if (cancelled) return;

        if (fromCache) {
          console.log(`💾 Loaded from IndexedDB cache`);
        }

        // Decode audio blob to AudioBuffer
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const arrayBuffer = await blob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        if (cancelled) return;

        // Store in in-memory cache for fast re-access
        audioBufferCache.set(audioUrl, audioBuffer);

        if (audioBufferCache.size > MAX_CACHE_SIZE) {
          const firstKey = audioBufferCache.keys().next().value;
          audioBufferCache.delete(firstKey);
        }

        audioBufferRef.current = audioBuffer;
        loadedUrlRef.current = audioUrl;
        setDuration(audioBuffer.duration);
        setIsLoaded(true);
        setDecodedAudioUrl(audioUrl);

        if (canvasRef.current) {
          if (envelopeData) {
            // Envelope wins — noise → envelope shape
            const envT = envelopeData.length / 2;
            paintBarAnimation(canvasRef.current, envelopeData, envT, width, height, color, gain, transitionFrameRef);
            envelopePaintedRef.current = true;
          } else if (!audioPaintedRef.current) {
            // First audio paint: noise → waveform bar animation
            paintBarAnimationFromBuffer(canvasRef.current, audioBuffer, width, height, color, gain, cropStart, cropEnd, transitionFrameRef);
            audioPaintedRef.current = true;
          }
        }
      } catch (error) {
        console.error('Error loading audio:', error);
        if (!envelopePaintedRef.current && !audioBufferRef.current) {
          setIsLoaded(false);
          if (canvasRef.current) {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            canvas.width = width;
            canvas.height = height;
            drawPlaceholder(ctx, width, height);
          }
        }
      }
    };

    loadAudio();

    return () => {
      cancelled = true;
    };
  }, [audioUrl, width, height, color, cropStart, cropEnd]);

  // Re-render waveform when dimensions, crop, envelope, OR gain change
  // (without reloading audio). Gain repaints are how real-time volume ↔
  // waveform size tracking works — slider moves → state → this effect.
  //
  // This effect runs on every render where deps changed. Most of those
  // runs (gain tweaks, crop drags, width resizes) should be INSTANT — no
  // noise transition, just a fresh paint. The noise transition lives in
  // the envelope-change and final-decode paths above, where the content
  // genuinely differs.
  useEffect(() => {
    if (!canvasRef.current) return;

    // Skip instant repaints when an envelope-swap transition is running —
    // otherwise this effect would clobber the animation mid-frame.
    if (transitionFrameRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = width;
    canvas.height = height;

    // Prefer envelope paint for repaints while envelope is the live view.
    if (envelopeData && envelopePaintedRef.current) {
      const envT = envelopeData.length / 2;
      renderEnvelope(ctx, envelopeData, envT, width, height, color, gain);
      return;
    }
    if (!audioBufferRef.current) {
      // No audio decoded yet, but we may still have envelope — paint it.
      if (envelopeData) {
        const envT = envelopeData.length / 2;
        renderEnvelope(ctx, envelopeData, envT, width, height, color, gain);
        envelopePaintedRef.current = true;
      }
      return;
    }

    // If envelopeData exists it is the permanent visual — always render it.
    if (envelopeData) {
      const envT = envelopeData.length / 2;
      renderEnvelope(ctx, envelopeData, envT, width, height, color, gain);
    } else if (audioPaintedRef.current) {
      // Only repaint from audio buffer once the initial noise→waveform
      // animation has completed. Skips while noise is still showing.
      renderWaveform(ctx, audioBufferRef.current, width, height, color, cropStart, cropEnd, gain);
    }
    // If neither condition: noise is currently displayed — leave it alone.
  }, [width, height, color, cropStart, cropEnd, envelopeData, gain, envelopeFps]);

  return {
    canvasRef,
    audioBuffer: audioBufferRef.current,
    duration,
    isLoaded,
    decodedAudioUrl,
  };
}

/**
 * Render waveform on canvas from audio buffer.
 * Uses peak-to-peak per bar (max − min of samples) so the visual
 * matches renderEnvelope and CompositeBusWaveform exactly: when a
 * single uploaded track stem-separates, the master waveform swaps
 * from this path to the composite without changing size or shape.
 * @param {number} gain - visual gain multiplier (0..N).
 */
function renderWaveform(ctx, audioBuffer, width, height, color, cropStart = 0, cropEnd = 0, gain = 1.0) {
  const data = audioBuffer.getChannelData(0);
  const duration = audioBuffer.duration;
  const sampleRate = audioBuffer.sampleRate;

  const startSample = Math.floor(cropStart * sampleRate);
  const endSample = cropEnd > 0 ? Math.floor((duration - cropEnd) * sampleRate) : data.length;
  const visibleLength = endSample - startSample;

  ctx.clearRect(0, 0, width, height);

  // Same bar geometry as renderEnvelope — 2px bars, 2px spacing, rounded caps.
  const barSpacing = 2;
  const barWidth = 2;
  const numBars = Math.floor(width / (barWidth + barSpacing));
  const samplesPerBar = Math.floor(visibleLength / numBars);

  const midY = height / 2;
  const maxBarHeight = height * 0.45;

  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  ctx.lineWidth = barWidth;
  ctx.lineCap = 'round';
  ctx.globalAlpha = 0.7;

  // Peak-to-peak per bar — identical metric to what latent_visual's
  // envelope produces for stems (max − min over the bar's sample/frame
  // window). The composite bus waveform sums this same quantity across
  // stems, so single-track and composite views are visually aligned.
  const ptpValues = [];
  let maxPtp = 0;

  for (let i = 0; i < numBars; i++) {
    const barStartSample = startSample + (i * samplesPerBar);
    const barEndSample = Math.min(barStartSample + samplesPerBar, endSample);

    let mx = -Infinity, mn = Infinity;
    for (let j = barStartSample; j < barEndSample; j++) {
      if (j >= data.length) break;
      const v = data[j];
      if (v > mx) mx = v;
      if (v < mn) mn = v;
    }
    const ptp = mx === -Infinity ? 0 : (mx - mn);
    ptpValues.push(ptp);
    if (ptp > maxPtp) maxPtp = ptp;
  }

  // Same normalization as renderEnvelope: peaks fill the container at
  // gain=1, no headroom divisor. Overshoot is clamped below.
  const normalizationFactor = maxPtp > 0.001 ? maxBarHeight / maxPtp : 1.0;
  const noiseFloor = maxPtp * 0.02;
  const g = Math.max(0, gain);

  for (let i = 0; i < numBars; i++) {
    const ptp = ptpValues[i];
    const x = i * (barWidth + barSpacing);

    if (ptp < noiseFloor || g === 0) {
      ctx.beginPath();
      ctx.moveTo(x + barWidth / 2, midY - 0.5);
      ctx.lineTo(x + barWidth / 2, midY + 0.5);
      ctx.stroke();
      continue;
    }

    const barHeight = Math.min(ptp * normalizationFactor * g, maxBarHeight);
    ctx.beginPath();
    ctx.moveTo(x + barWidth / 2, midY - barHeight);
    ctx.lineTo(x + barWidth / 2, midY + barHeight);
    ctx.stroke();
  }

  ctx.globalAlpha = 1.0;
}

/**
 * Render waveform bars from a pre-computed [2*T] envelope (min/max per frame).
 * Called INSTANTLY from latent_visual output — no audio decoding needed.
 * @param {number} gain - visual gain multiplier (0..N). Bar heights scale
 *   linearly with gain so volume slider tracks waveform size in real time.
 */
function renderEnvelope(ctx, envFlat, T, width, height, color, gain = 1.0) {
  const mins = envFlat.subarray(0, T);
  const maxs = envFlat.subarray(T, 2 * T);

  ctx.clearRect(0, 0, width, height);

  const barSpacing = 2;
  const barWidth = 2;
  const numBars = Math.floor(width / (barWidth + barSpacing));
  const midY = height / 2;
  const maxBarHeight = height * 0.45;

  // Compute per-bar peak-to-peak from envelope
  const ptpValues = [];
  let maxPtp = 0;
  for (let i = 0; i < numBars; i++) {
    const start = Math.floor((i * T) / numBars);
    const end = Math.max(start + 1, Math.floor(((i + 1) * T) / numBars));
    let mx = -Infinity, mn = Infinity;
    for (let j = start; j < end && j < T; j++) {
      if (maxs[j] > mx) mx = maxs[j];
      if (mins[j] < mn) mn = mins[j];
    }
    const ptp = mx - mn;
    ptpValues.push(ptp);
    if (ptp > maxPtp) maxPtp = ptp;
  }

  // Normalize so peak bars fill the full track height — matches
  // renderWaveform (RMS, 100% fill at max) which is used by the
  // pre-separation single-track view. No `× 1.2` headroom; we clamp
  // below anyway if anything overshoots.
  const normFactor = maxPtp > 0.001 ? maxBarHeight / maxPtp : 1.0;
  const noiseFloor = maxPtp * 0.02;
  const g = Math.max(0, gain);

  ctx.strokeStyle = color;
  ctx.lineWidth = barWidth;
  ctx.lineCap = 'round';
  ctx.globalAlpha = 0.7;

  for (let i = 0; i < numBars; i++) {
    const x = i * (barWidth + barSpacing);
    const ptp = ptpValues[i];
    if (ptp < noiseFloor || g === 0) {
      ctx.beginPath();
      ctx.moveTo(x + barWidth / 2, midY - 0.5);
      ctx.lineTo(x + barWidth / 2, midY + 0.5);
      ctx.stroke();
      continue;
    }
    const barH = Math.min(ptp * normFactor * g, maxBarHeight);
    ctx.beginPath();
    ctx.moveTo(x + barWidth / 2, midY - barH);
    ctx.lineTo(x + barWidth / 2, midY + barH);
    ctx.stroke();
  }
  ctx.globalAlpha = 1.0;
}

/**
 * Draw placeholder when no audio is loaded
 */
function drawPlaceholder(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#333';
  ctx.fillRect(0, height / 2 - 1, width, 2);

  ctx.fillStyle = '#666';
  ctx.font = '12px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('No audio loaded', width / 2, height / 2 - 10);
}

/**
 * Utility function to clear the audio buffer cache
 * Can be called from console or for memory management
 */
export function clearWaveformCache() {
  const cacheSize = audioBufferCache.size;
  audioBufferCache.clear();
  console.log(`🗑️ Cleared ${cacheSize} cached audio buffers`);
  return cacheSize;
}

/**
 * Get cache statistics
 */
export function getWaveformCacheStats() {
  return {
    size: audioBufferCache.size,
    maxSize: MAX_CACHE_SIZE,
    urls: Array.from(audioBufferCache.keys())
  };
}
