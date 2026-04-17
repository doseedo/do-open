import { useState, useRef, useCallback } from 'react';
import { convertAudioToMidi } from '../services/generationAPI';
import { useApp } from '../context/AppContext';
import { parseMIDI } from '../utils/midiParser';

/**
 * Convert audio to sine wave using FFT frequency domain analysis (like DEAF app)
 * This analyzes the frequency spectrum and generates a sine wave following the dominant frequency
 * @param {AudioBuffer} audioBuffer - The input audio buffer
 * @param {boolean} octaveUp - Whether to shift the frequency up by one octave (multiply by 2)
 * @param {number} glideTime - Glide/portamento time in seconds (default: 0.05)
 */
const convertAudioToSine = async (audioBuffer, octaveUp = false, glideTime = 0.05) => {
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const sampleRate = audioBuffer.sampleRate;
  const numberOfChannels = 1; // Mono output

  // Create offline context for processing
  const offlineContext = new OfflineAudioContext(
    numberOfChannels,
    audioBuffer.length,
    sampleRate
  );

  // Create analyser for FFT
  const analyser = offlineContext.createAnalyser();
  analyser.fftSize = 2048 * 8; // Large FFT for better frequency resolution (same as DEAF)
  const bufferLength = analyser.frequencyBinCount;
  const freqDomain = new Uint8Array(bufferLength);
  const timeDomain = new Uint8Array(bufferLength);

  // Create source from input audio (for analysis only, NOT output)
  const source = offlineContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(analyser);
  // DO NOT connect source to destination - we only want the sine wave output

  // Create oscillator for sine wave
  const oscillator = offlineContext.createOscillator();
  oscillator.type = 'sine';
  oscillator.frequency.setValueAtTime(120, 0);

  // Create gain for oscillator (for amplitude envelope)
  const oscGain = offlineContext.createGain();
  oscGain.gain.setValueAtTime(0.0001, 0);
  oscillator.connect(oscGain);
  oscGain.connect(offlineContext.destination); // Only sine goes to output

  console.log('🎵 Converting to sine wave using FFT analysis...');

  // Start playback
  source.start(0);
  oscillator.start(0);
  oscillator.stop(audioBuffer.duration);

  // Process in chunks (every 50ms like DEAF)
  const chunkInterval = 0.05; // 50ms
  const totalChunks = Math.ceil(audioBuffer.duration / chunkInterval);

  let previousFreq = 0;
  const frequencyHistory = [];
  const historySize = 3; // Average over 3 frames for smoothing

  // Schedule frequency and amplitude updates
  for (let chunk = 0; chunk < totalChunks; chunk++) {
    const time = chunk * chunkInterval;

    // Use larger window for better frequency resolution
    const windowSize = 4096; // Larger window = better low frequency resolution
    const sampleStart = Math.floor(time * sampleRate);
    const sampleEnd = Math.min(sampleStart + windowSize, audioBuffer.length);
    const chunkData = audioBuffer.getChannelData(0).slice(sampleStart, sampleEnd);

    if (chunkData.length < windowSize / 2) continue;

    // Calculate amplitude from time domain
    let maxAmplitude = 0;
    let rms = 0;
    for (let i = 0; i < chunkData.length; i++) {
      const amp = Math.abs(chunkData[i]);
      if (amp > maxAmplitude) maxAmplitude = amp;
      rms += chunkData[i] * chunkData[i];
    }
    rms = Math.sqrt(rms / chunkData.length);

    // Calculate volume (using RMS for better stability)
    let volume = rms * 10;
    if (volume < 0) volume = 0;

    // Use autocorrelation for more accurate pitch detection
    const detectedFreq = autoCorrelate(chunkData, sampleRate);

    // Add to frequency history for smoothing
    if (detectedFreq > 0) {
      frequencyHistory.push(detectedFreq);
      if (frequencyHistory.length > historySize) {
        frequencyHistory.shift();
      }
    }

    // Calculate median frequency from history (more stable than average)
    let smoothedFreq = 0;
    if (frequencyHistory.length > 0) {
      const sorted = [...frequencyHistory].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      smoothedFreq = sorted.length % 2 === 0
        ? (sorted[mid - 1] + sorted[mid]) / 2
        : sorted[mid];
    }

    // Only update if frequency changed significantly (reduce jitter)
    const freqDiff = Math.abs(smoothedFreq - previousFreq);
    const shouldUpdate = freqDiff > 2 || previousFreq === 0; // 2Hz threshold

    // Set oscillator frequency - use smoothed pitch
    if (smoothedFreq > 0 && volume > 0.02) { // Lower threshold for better sensitivity
      if (shouldUpdate) {
        // Apply octave up if enabled (multiply frequency by 2)
        const finalFreq = octaveUp ? smoothedFreq * 2 : smoothedFreq;

        // Use glide time for frequency transitions (portamento)
        if (glideTime > 0) {
          oscillator.frequency.linearRampToValueAtTime(finalFreq, time + glideTime);
        } else {
          // Instant change if glide is 0
          oscillator.frequency.setValueAtTime(finalFreq, time);
        }
        previousFreq = smoothedFreq;
      }
      // Reduce max gain significantly to prevent clipping (0.3 max instead of 0.8)
      oscGain.gain.exponentialRampToValueAtTime(Math.min(volume * 0.3, 0.3), time + 0.15);
    } else {
      oscGain.gain.exponentialRampToValueAtTime(0.0001, time + 0.15);
      frequencyHistory.length = 0; // Clear history on silence
    }
  }

  // Render the audio
  const renderedBuffer = await offlineContext.startRendering();

  // Normalize the audio to prevent clipping
  const channelData = renderedBuffer.getChannelData(0);
  let maxAmplitude = 0;

  // Find the maximum amplitude
  for (let i = 0; i < channelData.length; i++) {
    const absValue = Math.abs(channelData[i]);
    if (absValue > maxAmplitude) {
      maxAmplitude = absValue;
    }
  }

  // Always normalize to consistent level (with headroom to prevent clipping)
  if (maxAmplitude > 0.01) { // Only if there's actual audio
    const targetLevel = 0.5; // Target 50% to prevent clipping
    const normalizeGain = targetLevel / maxAmplitude;
    console.log(`🎚️ Normalizing sine wave: peak ${maxAmplitude.toFixed(3)} → ${(maxAmplitude * normalizeGain).toFixed(3)}`);

    for (let i = 0; i < channelData.length; i++) {
      channelData[i] *= normalizeGain;
    }
  }

  console.log('✅ Sine conversion complete');
  return renderedBuffer;
};

/**
 * Accurate pitch detection using autocorrelation (YIN-like algorithm)
 * This is much more stable than simple FFT for monophonic pitch tracking
 */
const autoCorrelate = (buffer, sampleRate) => {
  // Minimum and maximum frequencies to detect (human voice range)
  const minFreq = 80;  // ~E2
  const maxFreq = 1000; // ~C6

  const minPeriod = Math.floor(sampleRate / maxFreq);
  const maxPeriod = Math.ceil(sampleRate / minFreq);

  const size = buffer.length;

  // Step 1: Calculate difference function (autocorrelation)
  const diff = new Float32Array(maxPeriod);

  for (let tau = minPeriod; tau < maxPeriod; tau++) {
    let sum = 0;
    for (let i = 0; i < size - tau; i++) {
      const delta = buffer[i] - buffer[i + tau];
      sum += delta * delta;
    }
    diff[tau] = sum;
  }

  // Step 2: Cumulative mean normalized difference (YIN algorithm)
  const cmndf = new Float32Array(maxPeriod);
  cmndf[0] = 1;

  let runningSum = 0;
  for (let tau = 1; tau < maxPeriod; tau++) {
    runningSum += diff[tau];
    cmndf[tau] = diff[tau] / (runningSum / tau);
  }

  // Step 3: Find the best period (absolute threshold method)
  const threshold = 0.1; // Lower = more strict
  let bestPeriod = 0;

  for (let tau = minPeriod; tau < maxPeriod; tau++) {
    if (cmndf[tau] < threshold) {
      // Look for local minimum
      while (tau + 1 < maxPeriod && cmndf[tau + 1] < cmndf[tau]) {
        tau++;
      }
      bestPeriod = tau;
      break;
    }
  }

  // If no clear pitch found, return 0
  if (bestPeriod === 0) {
    return 0;
  }

  // Step 4: Parabolic interpolation for sub-sample accuracy
  let betterPeriod = bestPeriod;
  if (bestPeriod > 0 && bestPeriod < maxPeriod - 1) {
    const s0 = cmndf[bestPeriod - 1];
    const s1 = cmndf[bestPeriod];
    const s2 = cmndf[bestPeriod + 1];
    betterPeriod = bestPeriod + (s2 - s0) / (2 * (2 * s1 - s2 - s0));
  }

  // Convert period to frequency
  const frequency = sampleRate / betterPeriod;

  // Sanity check
  if (frequency < minFreq || frequency > maxFreq) {
    return 0;
  }

  return frequency;
};

/**
 * Convert AudioBuffer to WAV blob
 */
const audioBufferToWav = (audioBuffer) => {
  const numberOfChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const length = audioBuffer.length * numberOfChannels * 2;

  // Create WAV file buffer
  const buffer = new ArrayBuffer(44 + length);
  const view = new DataView(buffer);

  // Write WAV header
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + length, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, numberOfChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numberOfChannels * 2, true); // byte rate
  view.setUint16(32, numberOfChannels * 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, 'data');
  view.setUint32(40, length, true);

  // Write audio data
  const channels = [];
  for (let i = 0; i < numberOfChannels; i++) {
    channels.push(audioBuffer.getChannelData(i));
  }

  let offset = 44;
  for (let i = 0; i < audioBuffer.length; i++) {
    for (let channel = 0; channel < numberOfChannels; channel++) {
      const sample = Math.max(-1, Math.min(1, channels[channel][i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      offset += 2;
    }
  }

  return new Blob([buffer], { type: 'audio/wav' });
};

/**
 * Convert audio blob to WAV format using Web Audio API
 */
const convertToWav = async (blob) => {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  return audioBufferToWav(audioBuffer);
};

/**
 * Hook for recording audio from browser microphone
 * Returns recording controls and state
 * @param {Object} options - Configuration options
 * @param {boolean} options.extractMidi - Whether to extract MIDI from recording (default: false)
 * @param {boolean} options.convertToSine - Whether to convert recording to sine wave (default: false)
 * @param {boolean} options.octaveUp - Whether to shift sine wave up one octave (default: false)
 * @param {number} options.glideTime - Glide/portamento time in seconds (default: 0.05)
 */
export const useAudioRecorder = ({ extractMidi = false, convertToSine = false, octaveUp = false, glideTime = 0.05 } = {}) => {
  const { state, dispatch } = useApp();
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);
  const timerRef = useRef(null);

  const startRecording = useCallback(async () => {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 2,
          sampleRate: 44100,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      streamRef.current = stream;
      chunksRef.current = [];

      // Create MediaRecorder with appropriate MIME type
      let mimeType = 'audio/webm;codecs=opus';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/webm';
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = 'audio/mp4';
        }
      }

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingDuration(0);

      // Start timer
      const startTime = Date.now();
      timerRef.current = setInterval(() => {
        setRecordingDuration(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);

      console.log('🎤 Recording started');

    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Could not access microphone. Please check permissions.');
    }
  }, []);

  const stopRecording = useCallback(() => {
    return new Promise((resolve) => {
      if (!mediaRecorderRef.current || !isRecording) {
        resolve(null);
        return;
      }

      mediaRecorderRef.current.onstop = async () => {
        // Stop all tracks
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
        }

        // Clear timer
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }

        // Create blob from chunks
        const blob = new Blob(chunksRef.current, { type: mediaRecorderRef.current.mimeType });

        console.log('🎤 Converting recording to WAV format...');

        try {
          // Convert to WAV format
          const wavBlob = await convertToWav(blob);

          // Create File object from WAV blob
          const fileName = `recording-${Date.now()}.wav`;
          const file = new File([wavBlob], fileName, {
            type: 'audio/wav',
            lastModified: Date.now()
          });

          console.log(`🎤 Recording stopped - ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);

          // Reset recording state
          setIsRecording(false);
          chunksRef.current = [];
          mediaRecorderRef.current = null;
          streamRef.current = null;

          // Extract MIDI if enabled (using backend Basic Pitch)
          if (extractMidi) {
            try {
              setIsProcessing(true);
              const currentBPM = state.bpm || 120;
              console.log(`🎼 Extracting MIDI from recording at ${currentBPM} BPM (using backend Basic Pitch)...`);

              // Use backend Basic Pitch extraction with current timeline BPM
              const midiResult = await convertAudioToMidi(file, currentBPM);

              // Convert base64 MIDI to Blob
              const midiBytes = atob(midiResult.midi_data);
              const midiArray = new Uint8Array(midiBytes.length);
              for (let i = 0; i < midiBytes.length; i++) {
                midiArray[i] = midiBytes.charCodeAt(i);
              }
              const midiBlob = new Blob([midiArray], { type: 'audio/midi' });
              const metadata = midiResult.metadata;

              // CRITICAL: Parse MIDI to get notes for display!
              console.log('🎹 Parsing MIDI blob to extract notes...');
              const parsedMidi = parseMIDI(midiArray.buffer);
              console.log(`   Parsed ${parsedMidi.notes?.length || 0} notes from MIDI`);
              console.log(`   MIDI duration: ${parsedMidi.duration.toFixed(2)}s`);

              // Create MIDI track in DAW
              const busId = `midi-${Date.now()}`;
              dispatch({
                type: 'CREATE_BUS',
                payload: {
                  id: busId,
                  type: 'MIDI',
                  name: 'MIDI Recording'
                }
              });

              // Create MIDI track
              const track = {
                id: `track-${Date.now()}`,
                type: 'midi', // CRITICAL: Must be 'midi' for MIDI window to recognize it!
                name: `Recording (${metadata.total_notes} notes)`,
                midiData: {
                  notes: parsedMidi.notes || [], // Parsed notes from MIDI file!
                  tempo: parsedMidi.tempo || metadata.tempo,
                  duration: parsedMidi.duration || metadata.duration
                },
                duration: metadata.duration,
                startPosition: 0,
                gain: 1.0,
                isMuted: false,
                isSolo: false,
                fx: {
                  reverb: 0,
                  fadeIn: 0,
                  fadeOut: 0
                },
                metadata: {
                  type: 'recorded-midi',
                  midiBlob: midiBlob,
                  midiFilename: `recording-${Date.now()}.mid`,
                  totalNotes: metadata.total_notes,
                  minPitch: metadata.min_pitch,
                  maxPitch: metadata.max_pitch,
                  tempo: metadata.tempo
                }
              };

              dispatch({
                type: 'ADD_TRACK',
                payload: { busId, track }
              });

              console.log('✅ MIDI track created:', track.name);

              // Auto-load as conditioning
              console.log('🎹 Auto-loading MIDI track as conditioning input...');
              dispatch({
                type: 'SET_UPLOADED_FILE',
                payload: {
                  file: null,
                  fileType: 'midi',
                  previewUrl: null,
                  sourceTrack: {
                    trackId: track.id, // Use trackId for replacement logic
                    busId: busId,
                    name: track.name,
                    midiData: track.midiData
                  }
                }
              });
              console.log('✅ MIDI track loaded as conditioning input');

              setIsProcessing(false);

            } catch (error) {
              console.error('❌ MIDI extraction failed:', error);
              alert(`Failed to extract MIDI: ${error.message}`);
              setIsProcessing(false);
            }
          }

          // Convert to sine wave if enabled
          if (convertToSine) {
            try {
              setIsProcessing(true);
              console.log('🎵 Converting recording to sine wave...');

              // Decode audio to buffer
              const arrayBuffer = await file.arrayBuffer();
              const audioContext = new (window.AudioContext || window.webkitAudioContext)();
              const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

              // Convert to sine wave
              const sineBuffer = await convertAudioToSine(audioBuffer, octaveUp, glideTime);

              // Convert sine buffer to WAV blob
              const sineWavBlob = audioBufferToWav(sineBuffer);

              // Create File object from sine WAV blob
              const sineFileName = `sine-recording-${Date.now()}.wav`;
              const sineFile = new File([sineWavBlob], sineFileName, {
                type: 'audio/wav',
                lastModified: Date.now()
              });

              console.log(`✅ Sine conversion complete - ${sineFile.name} (${(sineFile.size / 1024 / 1024).toFixed(2)} MB)`);

              // Create Blob URL for audio playback
              const sineAudioUrl = URL.createObjectURL(sineWavBlob);

              // Create audio track in DAW with sine audio
              const busId = `audio-${Date.now()}`;
              dispatch({
                type: 'CREATE_BUS',
                payload: {
                  id: busId,
                  type: 'AUDIO',
                  name: 'Sine Wave'
                }
              });

              // Create audio track
              const track = {
                id: `track-${Date.now()}`,
                type: 'audio',
                name: 'Sine Recording',
                audioUrl: sineAudioUrl, // Blob URL for playback
                audioBuffer: sineBuffer, // Keep buffer for waveform display
                duration: sineBuffer.duration,
                startPosition: 0,
                gain: 1.0,
                isMuted: false,
                isSolo: false,
                fx: {
                  reverb: 0,
                  fadeIn: 0,
                  fadeOut: 0
                },
                metadata: {
                  type: 'sine-audio',
                  filename: sineFileName,
                  sampleRate: sineBuffer.sampleRate,
                  numberOfChannels: sineBuffer.numberOfChannels
                }
              };

              dispatch({
                type: 'ADD_TRACK',
                payload: { busId, track }
              });

              console.log('✅ Sine audio track created:', track.name);

              setIsProcessing(false);

            } catch (error) {
              console.error('❌ Sine conversion failed:', error);
              alert(`Failed to convert to sine wave: ${error.message}`);
              setIsProcessing(false);
            }
          }

          resolve(file);
        } catch (error) {
          console.error('Error converting to WAV:', error);
          alert('Failed to convert recording to WAV format');

          // Reset state
          setIsRecording(false);
          chunksRef.current = [];
          mediaRecorderRef.current = null;
          streamRef.current = null;

          resolve(null);
        }
      };

      mediaRecorderRef.current.stop();
    });
  }, [isRecording]);

  const cancelRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      // Stop recording
      mediaRecorderRef.current.stop();

      // Stop all tracks
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      // Clear timer
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      // Reset state
      setIsRecording(false);
      setRecordingDuration(0);
      chunksRef.current = [];
      mediaRecorderRef.current = null;
      streamRef.current = null;

      console.log('🎤 Recording cancelled');
    }
  }, [isRecording]);

  return {
    isRecording,
    recordingDuration,
    isProcessing,
    startRecording,
    stopRecording,
    cancelRecording
  };
};
