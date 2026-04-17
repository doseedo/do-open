import { useEffect } from 'react';

/**
 * Custom hook for keyboard controls
 * Handles spacebar for play/pause, arrow keys for seeking, etc.
 */
export function useKeyboardControls(dispatch, isPlaying) {
  useEffect(() => {
    const handleKeyPress = (e) => {
      // Ignore if user is typing in an input field
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
        return;
      }

      switch (e.code) {
        case 'Space':
          e.preventDefault();
          dispatch({ type: 'TOGGLE_PLAY' });
          break;

        case 'ArrowLeft':
          // TODO: Seek backward
          e.preventDefault();
          break;

        case 'ArrowRight':
          // TODO: Seek forward
          e.preventDefault();
          break;

        case 'Escape':
          // Stop playback
          if (isPlaying) {
            dispatch({ type: 'SET_PLAYING', payload: false });
          }
          break;

        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);

    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [dispatch, isPlaying]);
}
