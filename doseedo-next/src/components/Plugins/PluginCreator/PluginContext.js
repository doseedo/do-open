import React, { createContext, useContext, useRef, useMemo } from 'react';

/**
 * PluginContext — single source of truth for DSP engine + param controller.
 * Wrap PluginCreator children in <PluginProvider> to give all orphaned
 * components (ModMatrix, MSEGEditor, EQCurveDisplay, ADSRDisplay) access
 * without prop drilling.
 */

export const PluginContext = createContext(null);

export function usePluginContext() {
  const ctx = useContext(PluginContext);
  if (!ctx) throw new Error('usePluginContext must be used inside <PluginProvider>');
  return ctx;
}

/**
 * PluginParamController — thin bridge from UI events to WebAudioDSPEngine.
 * Wraps engine.setParameter() and fans out to subscribers.
 */
export class PluginParamController {
  constructor(engine) {
    this._engine = engine;
    this._listeners = new Map();
    this._modConnections = [];
  }

  setParameter(paramId, value) {
    if (this._engine) this._engine.setParameter(paramId, value);
    (this._listeners.get(paramId) || []).forEach(fn => fn(value));
  }

  setModConnections(connections) {
    this._modConnections = connections;
    // Apply mod depth to each target param
    connections.forEach(({ targetParam, depth, sourceValue = 0 }) => {
      const modulated = Math.max(0, Math.min(1, sourceValue * depth));
      if (this._engine) this._engine.setParameter(targetParam, modulated);
    });
  }

  getModConnections() { return this._modConnections; }

  subscribe(paramId, fn) {
    if (!this._listeners.has(paramId)) this._listeners.set(paramId, new Set());
    this._listeners.get(paramId).add(fn);
    return () => this._listeners.get(paramId).delete(fn);
  }
}

/**
 * PluginProvider — instantiates PluginParamController and provides it to tree.
 * Usage in PluginCreator.js:
 *
 *   <PluginProvider engine={engineRef.current}>
 *     <ModMatrix />
 *     <MSEGEditor />
 *     ...
 *   </PluginProvider>
 */
export function PluginProvider({ children, engine }) {
  const paramController = useMemo(
    () => new PluginParamController(engine),
    [engine]
  );

  return (
    <PluginContext.Provider value={{ paramController, engine }}>
      {children}
    </PluginContext.Provider>
  );
}
