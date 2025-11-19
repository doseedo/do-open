/**
 * Web Audio Plugins - Core Module
 *
 * Entry point for the core integration and routing system.
 * Provides base plugin class, router, preset management, automation,
 * and performance monitoring.
 *
 * @module web-audio-plugins/core
 * @author Agent 10 - Integration & Routing System
 */

import BasePlugin from './BasePlugin.js';
import Router from './Router.js';
import PresetManager from './PresetManager.js';
import ParamAutomation from './ParamAutomation.js';
import PerformanceMonitor from './PerformanceMonitor.js';
import PluginFactory from './PluginFactory.js';

export {
  BasePlugin,
  Router,
  PresetManager,
  ParamAutomation,
  PerformanceMonitor,
  PluginFactory
};

export default {
  BasePlugin,
  Router,
  PresetManager,
  ParamAutomation,
  PerformanceMonitor,
  PluginFactory
};
