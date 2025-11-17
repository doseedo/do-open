import { useEffect, useRef, useCallback } from 'react';

/**
 * useMetronome - Hook for metronome playback synchronized with timeline
 *
 * Features:
 * - Generates click sounds using Web Audio API
 * - Syncs with playback position and tempo
 * - Handles tempo changes across scenes
 * - Downbeat emphasis (higher pitch for bar start)
 */
export function useMetronome(isPlaying, playheadPosition, isMetronomeOn, bpm, sceneTempos, sceneChanges) {
  const audioContextRef = useRef(null);
  const schedulerTimeoutRef = useRef(null);
  const lastScheduledBeatRef = useRef(-1);

  // Initialize audio context
  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioContextRef.current;
  }, []);

  // Generate metronome click sound
  const playMetronomeClick = useCallback((isDownbeat = false) => {
    const ctx = getAudioContext();
    if (!ctx) return;

    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();

    // Downbeat (bar start) is higher pitched (1200Hz), regular beats are lower (800Hz)
    oscillator.frequency.value = isDownbeat ? 1200 : 800;
    oscillator.type = 'sine';

    // Quick attack/decay for click sound
    const now = ctx.currentTime;
    gainNode.gain.setValueAtTime(0, now);
    gainNode.gain.linearRampToValueAtTime(0.3, now + 0.001);
    gainNode.gain.exponentialRampToValueAtTime(0.01, now + 0.05);

    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);

    oscillator.start(now);
    oscillator.stop(now + 0.05);
  }, [getAudioContext]);

  // Calculate total beats elapsed from start, accounting for tempo changes
  const calculateBeatsElapsed = useCallback((time) => {
    if (!sceneTempos || !sceneChanges || sceneTempos.length === 0 || sceneChanges.length <= 1) {
      // Constant BPM mode
      const secondsPerBeat = 60 / bpm;
      return time / secondsPerBeat;
    }

    let totalBeats = 0;

    // Accumulate beats from each scene up to current time
    for (let i = 0; i < sceneTempos.length; i++) {
      const sceneStart = sceneChanges[i];
      const sceneEnd = sceneChanges[i + 1] || time;
      const sceneBPM = sceneTempos[i];
      const secondsPerBeat = 60 / sceneBPM;

      if (time <= sceneStart) {
        break;
      } else if (time >= sceneEnd) {
        // Fully passed this scene - add all beats from this scene
        const sceneDuration = sceneEnd - sceneStart;
        totalBeats += sceneDuration / secondsPerBeat;
      } else {
        // Currently in this scene - add partial beats
        const timeInScene = time - sceneStart;
        totalBeats += timeInScene / secondsPerBeat;
        break;
      }
    }

    return totalBeats;
  }, [bpm, sceneTempos, sceneChanges]);

  // Calculate time of next beat after current position
  const calculateNextBeatTime = useCallback((currentTime) => {
    const beatsElapsed = calculateBeatsElapsed(currentTime);
    const nextBeatNumber = Math.floor(beatsElapsed) + 1;

    // Work backwards to find the time of this beat number
    if (!sceneTempos || !sceneChanges || sceneTempos.length === 0 || sceneChanges.length <= 1) {
      // Constant BPM mode
      const secondsPerBeat = 60 / bpm;
      return nextBeatNumber * secondsPerBeat;
    }

    let accumulatedBeats = 0;

    for (let i = 0; i < sceneTempos.length; i++) {
      const sceneStart = sceneChanges[i];
      const sceneEnd = sceneChanges[i + 1] || currentTime + 100; // Fallback
      const sceneBPM = sceneTempos[i];
      const secondsPerBeat = 60 / sceneBPM;
      const sceneDuration = sceneEnd - sceneStart;
      const beatsInScene = sceneDuration / secondsPerBeat;

      if (nextBeatNumber <= accumulatedBeats + beatsInScene) {
        // The next beat is in this scene
        const beatsIntoScene = nextBeatNumber - accumulatedBeats;
        return sceneStart + (beatsIntoScene * secondsPerBeat);
      }

      accumulatedBeats += beatsInScene;
    }

    // Fallback
    return currentTime + (60 / bpm);
  }, [bpm, sceneTempos, sceneChanges, calculateBeatsElapsed]);

  // Schedule metronome clicks based on current playback position
  const scheduleMetronomeBeats = useCallback(() => {
    if (!isMetronomeOn || !isPlaying) return;

    const currentTime = playheadPosition;

    // Calculate total beats elapsed and determine beat in bar
    const beatsElapsed = calculateBeatsElapsed(currentTime);
    const currentBeatNumber = Math.floor(beatsElapsed);
    const beatInBar = currentBeatNumber % 4; // 0, 1, 2, 3 (0 is downbeat)

    // Calculate next beat time
    const nextBeatTime = calculateNextBeatTime(currentTime);
    const nextBeatNumber = currentBeatNumber + 1;

    // Only schedule if we haven't already scheduled this beat
    if (nextBeatNumber > lastScheduledBeatRef.current) {
      const timeUntilNextBeat = nextBeatTime - currentTime;

      // If next beat is within scheduling window (150ms), schedule it
      if (timeUntilNextBeat > 0 && timeUntilNextBeat <= 0.15) {
        const nextBeatInBar = (beatInBar + 1) % 4;
        const isDownbeat = nextBeatInBar === 0;

        // Schedule the click
        setTimeout(() => {
          if (isMetronomeOn && isPlaying) {
            playMetronomeClick(isDownbeat);
            console.log(`🥁 Metronome: ${isDownbeat ? 'DOWNBEAT' : 'beat'} at ${nextBeatTime.toFixed(2)}s`);
          }
        }, timeUntilNextBeat * 1000);

        lastScheduledBeatRef.current = nextBeatNumber;
      }
    }

    // Schedule next check (look ahead by 50ms)
    if (isMetronomeOn && isPlaying) {
      schedulerTimeoutRef.current = setTimeout(scheduleMetronomeBeats, 50);
    }
  }, [isMetronomeOn, isPlaying, playheadPosition, calculateBeatsElapsed, calculateNextBeatTime, playMetronomeClick]);

  // Start/stop metronome scheduler
  useEffect(() => {
    if (isMetronomeOn && isPlaying) {
      // Reset last scheduled beat when starting
      lastScheduledBeatRef.current = -1;
      scheduleMetronomeBeats();
    } else {
      // Clear scheduler when stopping
      if (schedulerTimeoutRef.current) {
        clearTimeout(schedulerTimeoutRef.current);
        schedulerTimeoutRef.current = null;
      }
      lastScheduledBeatRef.current = -1;
    }

    return () => {
      if (schedulerTimeoutRef.current) {
        clearTimeout(schedulerTimeoutRef.current);
      }
    };
  }, [isMetronomeOn, isPlaying, scheduleMetronomeBeats]);

  // Cleanup audio context on unmount
  useEffect(() => {
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  return {
    // Expose for external use if needed
    playMetronomeClick
  };
}
