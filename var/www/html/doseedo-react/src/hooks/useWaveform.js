import React, { useEffect, useRef } from 'react';
import { fetchAudioWithCache } from '../services/audioCacheService';

// Global in-memory cache for decoded audio buffers (faster than re-decoding).
// EXPORTED so other services (e.g. latent-based waveform previews) can
// pre-populate it before a track's real audio is ready.
export const audioBufferCache = new Map();
const MAX_CACHE_SIZE = 50; // Limit in-memory cache size

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
 */
export function useWaveform(audioUrl, width = 800, height = 60, color = '#f5f5f5', cropStart = 0, cropEnd = 0, envelopeData = null, envelopeFps = 25) {
  const canvasRef = useRef(null);
  const audioBufferRef = useRef(null);
  const loadedUrlRef = useRef(null);
  const envelopePaintedRef = useRef(false);
  const [isLoaded, setIsLoaded] = React.useState(false);
  const [duration, setDuration] = React.useState(null);

  // Instant envelope rendering — draws the waveform from latent_visual
  // peaks WITHOUT waiting for audio decode. Fires whenever envelopeData
  // changes (typically once, right after latent separation).
  useEffect(() => {
    if (!envelopeData || !canvasRef.current) return;
    const T = envelopeData.length / 2;
    const dur = T / envelopeFps;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = width;
    canvas.height = height;
    renderEnvelope(ctx, envelopeData, T, width, height, color);
    envelopePaintedRef.current = true;
    if (!isLoaded) {
      setDuration(dur);
      setIsLoaded(true);
    }
  }, [envelopeData, width, height, color, envelopeFps, isLoaded]);

  // Load audio only when URL changes
  useEffect(() => {
    if (!audioUrl || loadedUrlRef.current === audioUrl) return;

    let cancelled = false;
    setIsLoaded(false);

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

          // If envelope was already painted by the envelope effect, don't
          // repaint from cache — the envelope is the correct visual until
          // the final full-length decode arrives via the fetch path.
          if (envelopePaintedRef.current) return;
          if (canvasRef.current) {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            canvas.width = width;
            canvas.height = height;
            if (envelopeData) {
              const envT = envelopeData.length / 2;
              renderEnvelope(ctx, envelopeData, envT, width, height, color);
              envelopePaintedRef.current = true;
            } else {
              renderWaveform(ctx, cachedBuffer, width, height, color, cropStart, cropEnd);
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

        // Limit in-memory cache size (LRU-style: remove oldest entries)
        if (audioBufferCache.size > MAX_CACHE_SIZE) {
          const firstKey = audioBufferCache.keys().next().value;
          audioBufferCache.delete(firstKey);
          console.log(`🗑️ Removed oldest in-memory cached buffer`);
        }

        audioBufferRef.current = audioBuffer;
        loadedUrlRef.current = audioUrl;
        setDuration(audioBuffer.duration);
        setIsLoaded(true);

        // Only repaint if this is the final full-length decode or no
        // envelope was painted. Intermediate chunks and tiny placeholders
        // must NOT overwrite the latent visual envelope.
        if (canvasRef.current) {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          if (envelopeData && envelopePaintedRef.current) {
            const envT = envelopeData.length / 2;
            const expectedDur = envT / envelopeFps;
            const isFinalDecode = audioBuffer.duration >= expectedDur * 0.9;
            if (isFinalDecode) {
              // Final decode ready — switch from envelope to real waveform
              canvas.width = width;
              canvas.height = height;
              renderWaveform(ctx, audioBuffer, width, height, color, cropStart, cropEnd);
              envelopePaintedRef.current = false;
            }
            // else: envelope stays, don't touch canvas
          } else {
            canvas.width = width;
            canvas.height = height;
            if (envelopeData) {
              const envT = envelopeData.length / 2;
              renderEnvelope(ctx, envelopeData, envT, width, height, color);
              envelopePaintedRef.current = true;
            } else {
              renderWaveform(ctx, audioBuffer, width, height, color, cropStart, cropEnd);
            }
          }
        }
      } catch (error) {
        console.error('Error loading audio:', error);
        setIsLoaded(false);
        if (canvasRef.current) {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');

          // Set canvas dimensions for error state too
          canvas.width = width;
          canvas.height = height;

          drawPlaceholder(ctx, width, height);
        }
      }
    };

    loadAudio();

    return () => {
      cancelled = true;
    };
  }, [audioUrl, width, height, color, cropStart, cropEnd]);

  // Re-render waveform when dimensions or crop values change (without reloading audio)
  useEffect(() => {
    if (!canvasRef.current || !audioBufferRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = width;
    canvas.height = height;

    const envT = envelopeData ? envelopeData.length / 2 : 0;
    const expectedDur = envT / envelopeFps;
    const isPartial = envelopeData && audioBufferRef.current.duration < expectedDur * 0.9;
    if (isPartial) {
      renderEnvelope(ctx, envelopeData, envT, width, height, color);
    } else {
      renderWaveform(ctx, audioBufferRef.current, width, height, color, cropStart, cropEnd);
    }
  }, [width, height, color, cropStart, cropEnd, envelopeData]);

  return {
    canvasRef,
    audioBuffer: audioBufferRef.current,
    duration,
    isLoaded
  };
}

/**
 * Render waveform on canvas from audio buffer
 * Uses simplified vertical bar visualization (RMS-based)
 */
function renderWaveform(ctx, audioBuffer, width, height, color, cropStart = 0, cropEnd = 0) {
  const data = audioBuffer.getChannelData(0); // Get first channel
  const duration = audioBuffer.duration;
  const sampleRate = audioBuffer.sampleRate;

  // Calculate which portion of the audio to display
  const startSample = Math.floor(cropStart * sampleRate);
  const endSample = cropEnd > 0 ? Math.floor((duration - cropEnd) * sampleRate) : data.length;
  const visibleLength = endSample - startSample;

  ctx.clearRect(0, 0, width, height);

  // Calculate number of bars based on width
  const barSpacing = 2; // Space between bars
  const barWidth = 2; // Width of each bar
  const numBars = Math.floor(width / (barWidth + barSpacing));
  const samplesPerBar = Math.floor(visibleLength / numBars);

  const midY = height / 2;
  const maxBarHeight = height * 0.45; // Max bar extends 45% of height from center

  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  ctx.lineWidth = barWidth;
  ctx.lineCap = 'round'; // Rounded caps
  ctx.globalAlpha = 0.7;

  // First pass: calculate all RMS values and find max for normalization
  const rmsValues = [];
  let maxRms = 0;

  for (let i = 0; i < numBars; i++) {
    const barStartSample = startSample + (i * samplesPerBar);
    const barEndSample = Math.min(barStartSample + samplesPerBar, endSample);

    // Calculate RMS (Root Mean Square) for this bar's samples
    let sumSquares = 0;
    let count = 0;
    for (let j = barStartSample; j < barEndSample; j++) {
      if (j >= data.length) break;
      sumSquares += data[j] * data[j];
      count++;
    }

    const rms = count > 0 ? Math.sqrt(sumSquares / count) : 0;
    rmsValues.push(rms);
    maxRms = Math.max(maxRms, rms);
  }

  // Calculate normalization factor to ensure waveform fits in container
  // If maxRms is very small, use a minimum amplification
  const normalizationFactor = maxRms > 0.001 ? maxBarHeight / (maxRms * 2.5) : 1.0;

  // Noise floor threshold - values below this are considered silent
  const noiseFloor = maxRms * 0.02; // 2% of max is considered noise/silence

  // Second pass: draw normalized bars
  for (let i = 0; i < numBars; i++) {
    const rms = rmsValues[i];
    const x = i * (barWidth + barSpacing);

    // For silent/near-silent sections, draw a flat center line
    if (rms < noiseFloor) {
      ctx.beginPath();
      ctx.moveTo(x + barWidth / 2, midY - 0.5);
      ctx.lineTo(x + barWidth / 2, midY + 0.5);
      ctx.stroke();
      continue;
    }

    // Normalize - no minimum height enforcement to avoid dots
    const barHeight = rms * normalizationFactor * 2.5;

    // Clamp to maxBarHeight to ensure it never exceeds container
    const clampedHeight = Math.min(barHeight, maxBarHeight);

    // Draw vertical bar with rounded caps
    const barTop = midY - clampedHeight;
    const barBottom = midY + clampedHeight;

    ctx.beginPath();
    ctx.moveTo(x + barWidth / 2, barTop);
    ctx.lineTo(x + barWidth / 2, barBottom);
    ctx.stroke();
  }

  ctx.globalAlpha = 1.0;
}

/**
 * Render waveform bars from a pre-computed [2*T] envelope (min/max per frame).
 * Called INSTANTLY from latent_visual output — no audio decoding needed.
 */
function renderEnvelope(ctx, envFlat, T, width, height, color) {
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

  const normFactor = maxPtp > 0.001 ? maxBarHeight / (maxPtp * 1.2) : 1.0;
  const noiseFloor = maxPtp * 0.02;

  ctx.strokeStyle = color;
  ctx.lineWidth = barWidth;
  ctx.lineCap = 'round';
  ctx.globalAlpha = 0.7;

  for (let i = 0; i < numBars; i++) {
    const x = i * (barWidth + barSpacing);
    const ptp = ptpValues[i];
    if (ptp < noiseFloor) {
      ctx.beginPath();
      ctx.moveTo(x + barWidth / 2, midY - 0.5);
      ctx.lineTo(x + barWidth / 2, midY + 0.5);
      ctx.stroke();
      continue;
    }
    const barH = Math.min(ptp * normFactor, maxBarHeight);
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
