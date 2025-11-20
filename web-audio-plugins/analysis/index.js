/**
 * Analysis Plugins
 * Audio analysis and visualization tools
 *
 * @module analysis
 * @author Agent 8 (Analyzer Plugins)
 * @version 1.0.0
 */

export { default as MeterPlugin } from './MeterPlugin.js';
export { default as SpectrumAnalyzerPlugin } from './SpectrumAnalyzerPlugin.js';
export { default as OscilloscopePlugin } from './OscilloscopePlugin.js';

// Also export with simpler names
export { default as Meter } from './MeterPlugin.js';
export { default as SpectrumAnalyzer } from './SpectrumAnalyzerPlugin.js';
export { default as Oscilloscope } from './OscilloscopePlugin.js';
