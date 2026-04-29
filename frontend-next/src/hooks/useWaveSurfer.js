import { useEffect, useRef, useState } from 'react';

/**
 * Custom hook for WaveSurfer.js integration
 * Handles WaveSurfer instance creation and management
 */
export function useWaveSurfer(containerRef, options = {}) {
  const wavesurferRef = useRef(null);
  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    if (!containerRef.current) return;

    // Dynamically import WaveSurfer
    import('wavesurfer.js').then((WaveSurfer) => {
      wavesurferRef.current = WaveSurfer.default.create({
        container: containerRef.current,
        waveColor: '#667eea',
        progressColor: '#764ba2',
        cursorColor: '#ffffff',
        barWidth: 2,
        barRadius: 3,
        cursorWidth: 1,
        height: 150,
        barGap: 2,
        ...options
      });

      // Event listeners
      wavesurferRef.current.on('ready', () => {
        setIsReady(true);
        setDuration(wavesurferRef.current.getDuration());
      });

      wavesurferRef.current.on('play', () => {
        setIsPlaying(true);
      });

      wavesurferRef.current.on('pause', () => {
        setIsPlaying(false);
      });

      wavesurferRef.current.on('audioprocess', () => {
        setCurrentTime(wavesurferRef.current.getCurrentTime());
      });
    });

    return () => {
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
      }
    };
  }, [containerRef]);

  const playPause = () => {
    if (wavesurferRef.current) {
      wavesurferRef.current.playPause();
    }
  };

  const load = (url) => {
    if (wavesurferRef.current) {
      wavesurferRef.current.load(url);
    }
  };

  const seekTo = (progress) => {
    if (wavesurferRef.current) {
      wavesurferRef.current.seekTo(progress);
    }
  };

  return {
    wavesurfer: wavesurferRef.current,
    isReady,
    isPlaying,
    duration,
    currentTime,
    playPause,
    load,
    seekTo
  };
}
