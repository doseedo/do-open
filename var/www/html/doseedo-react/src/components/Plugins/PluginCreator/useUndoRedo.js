import { useState, useCallback, useRef } from 'react';

const MAX_HISTORY = 50;

export default function useUndoRedo(initialState) {
  const [state, setState] = useState(initialState);
  const historyRef = useRef([JSON.stringify(initialState)]);
  const indexRef = useRef(0);

  const pushState = useCallback((newState) => {
    const serialized = JSON.stringify(newState);
    // Trim future states if we undid
    historyRef.current = historyRef.current.slice(0, indexRef.current + 1);
    historyRef.current.push(serialized);
    if (historyRef.current.length > MAX_HISTORY) {
      historyRef.current.shift();
    } else {
      indexRef.current++;
    }
    setState(newState);
  }, []);

  const undo = useCallback(() => {
    if (indexRef.current > 0) {
      indexRef.current--;
      const prev = JSON.parse(historyRef.current[indexRef.current]);
      setState(prev);
      return prev;
    }
    return null;
  }, []);

  const redo = useCallback(() => {
    if (indexRef.current < historyRef.current.length - 1) {
      indexRef.current++;
      const next = JSON.parse(historyRef.current[indexRef.current]);
      setState(next);
      return next;
    }
    return null;
  }, []);

  const canUndo = indexRef.current > 0;
  const canRedo = indexRef.current < historyRef.current.length - 1;

  return { state, pushState, undo, redo, canUndo, canRedo, setState };
}
