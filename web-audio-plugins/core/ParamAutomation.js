/**
 * ParamAutomation - Timeline-based parameter automation system
 *
 * Records, plays back, and manages parameter automation with
 * support for multiple curve types, timeline sync, and MIDI CC mapping.
 *
 * @class ParamAutomation
 * @author Agent 10 - Integration & Routing System
 */

class ParamAutomation {
  /**
   * Create a new ParamAutomation instance
   * @param {AudioContext} audioContext - The Web Audio API context
   * @param {Object} options - Configuration options
   * @param {number} options.bpm - Tempo in BPM (default: 120)
   * @param {number} options.timeSignature - Time signature (default: 4)
   */
  constructor(audioContext, options = {}) {
    if (!audioContext) {
      throw new Error('AudioContext is required');
    }

    this.context = audioContext;
    this.options = {
      bpm: options.bpm || 120,
      timeSignature: options.timeSignature || 4
    };

    // Automation storage: Map<pluginId:paramName, AutomationTrack>
    this.automations = new Map();

    // Playback state
    this.isPlaying = false;
    this.isRecording = false;
    this.startTime = 0;
    this.currentTime = 0;
    this.loopStart = 0;
    this.loopEnd = 0;
    this.looping = false;

    // Recording state
    this.recordingTrack = null;
    this.recordingInterval = null;
    this.recordingResolution = 50; // ms between samples

    // MIDI CC mapping
    this.midiCCMap = new Map(); // CC number -> { pluginId, paramName }

    // Event listeners
    this.eventListeners = {
      playbackStarted: [],
      playbackStopped: [],
      recordingStarted: [],
      recordingStopped: [],
      automationAdded: [],
      automationRemoved: []
    };
  }

  /**
   * Create automation track key
   * @param {string} pluginId - Plugin ID
   * @param {string} paramName - Parameter name
   * @returns {string} Track key
   * @private
   */
  getTrackKey(pluginId, paramName) {
    return `${pluginId}:${paramName}`;
  }

  /**
   * Record automation for a parameter
   * @param {string} pluginId - Plugin ID
   * @param {string} paramName - Parameter name
   * @param {number} value - Parameter value
   * @param {number} time - Time in seconds (relative to start)
   * @param {string} curveType - Curve type: 'linear', 'exponential', 'step' (default: 'linear')
   */
  recordAutomation(pluginId, paramName, value, time, curveType = 'linear') {
    const key = this.getTrackKey(pluginId, paramName);

    if (!this.automations.has(key)) {
      this.automations.set(key, {
        pluginId,
        paramName,
        events: []
      });
      this.emit('automationAdded', { pluginId, paramName });
    }

    const track = this.automations.get(key);

    // Add event
    track.events.push({
      time: time,
      value: value,
      curveType: curveType
    });

    // Sort by time
    track.events.sort((a, b) => a.time - b.time);
  }

  /**
   * Start recording automation for a parameter
   * @param {Router} router - Router instance
   * @param {string} pluginId - Plugin ID
   * @param {string} paramName - Parameter name
   * @returns {boolean} Success status
   */
  startRecording(router, pluginId, paramName) {
    const plugin = router.getPlugin(pluginId);
    if (!plugin || !plugin.params[paramName]) {
      console.error(`Plugin or parameter not found: ${pluginId}.${paramName}`);
      return false;
    }

    if (this.isRecording) {
      console.warn('Already recording');
      return false;
    }

    this.isRecording = true;
    this.recordingTrack = { pluginId, paramName };
    this.startTime = this.context.currentTime;

    // Clear existing automation for this parameter
    const key = this.getTrackKey(pluginId, paramName);
    this.automations.delete(key);

    // Poll parameter value at regular intervals
    const param = plugin.params[paramName];
    this.recordingInterval = setInterval(() => {
      const currentTime = this.context.currentTime - this.startTime;
      const value = param.value;
      this.recordAutomation(pluginId, paramName, value, currentTime, 'linear');
    }, this.recordingResolution);

    this.emit('recordingStarted', { pluginId, paramName });

    return true;
  }

  /**
   * Stop recording automation
   */
  stopRecording() {
    if (!this.isRecording) {
      return;
    }

    if (this.recordingInterval) {
      clearInterval(this.recordingInterval);
      this.recordingInterval = null;
    }

    this.emit('recordingStopped', this.recordingTrack);

    this.isRecording = false;
    this.recordingTrack = null;
  }

  /**
   * Play automation for all parameters
   * @param {Router} router - Router instance
   * @param {number} startTime - Start time in seconds (default: 0)
   */
  playAutomation(router, startTime = 0) {
    if (this.isPlaying) {
      console.warn('Already playing');
      return;
    }

    this.isPlaying = true;
    this.startTime = this.context.currentTime;
    this.currentTime = startTime;

    this.automations.forEach((track, key) => {
      const plugin = router.getPlugin(track.pluginId);

      if (!plugin) {
        console.warn(`Plugin ${track.pluginId} not found`);
        return;
      }

      const param = plugin.params[track.paramName];
      if (!param) {
        console.warn(`Parameter ${track.paramName} not found on ${track.pluginId}`);
        return;
      }

      // Cancel any existing automation
      param.cancelScheduledValues(this.context.currentTime);

      // Schedule all events
      track.events.forEach((event, index) => {
        if (event.time < startTime) {
          return; // Skip events before start time
        }

        const scheduleTime = this.startTime + (event.time - startTime);

        try {
          switch (event.curveType) {
            case 'step':
              param.setValueAtTime(event.value, scheduleTime);
              break;

            case 'exponential':
              if (event.value > 0) {
                if (index === 0 || track.events[index - 1].time < startTime) {
                  param.setValueAtTime(event.value, scheduleTime);
                } else {
                  param.exponentialRampToValueAtTime(event.value, scheduleTime);
                }
              } else {
                // Exponential ramp doesn't work with 0, use linear
                param.linearRampToValueAtTime(event.value, scheduleTime);
              }
              break;

            case 'linear':
            default:
              if (index === 0 || track.events[index - 1].time < startTime) {
                param.setValueAtTime(event.value, scheduleTime);
              } else {
                param.linearRampToValueAtTime(event.value, scheduleTime);
              }
              break;
          }
        } catch (error) {
          console.error(`Error scheduling automation event:`, error);
        }
      });
    });

    this.emit('playbackStarted', { startTime });
  }

  /**
   * Stop automation playback
   */
  stopAutomation() {
    if (!this.isPlaying) {
      return;
    }

    this.isPlaying = false;

    this.emit('playbackStopped', {});
  }

  /**
   * Clear automation for a specific parameter or plugin
   * @param {string} pluginId - Plugin ID (null to clear all)
   * @param {string} paramName - Parameter name (null to clear all for plugin)
   */
  clearAutomation(pluginId = null, paramName = null) {
    if (!pluginId) {
      // Clear all automation
      this.automations.forEach((track, key) => {
        this.emit('automationRemoved', { pluginId: track.pluginId, paramName: track.paramName });
      });
      this.automations.clear();
      return;
    }

    if (!paramName) {
      // Clear all automation for plugin
      const keys = Array.from(this.automations.keys())
        .filter(key => key.startsWith(pluginId + ':'));

      keys.forEach(key => {
        const track = this.automations.get(key);
        this.emit('automationRemoved', { pluginId: track.pluginId, paramName: track.paramName });
        this.automations.delete(key);
      });
    } else {
      // Clear specific parameter
      const key = this.getTrackKey(pluginId, paramName);
      const track = this.automations.get(key);
      if (track) {
        this.emit('automationRemoved', { pluginId, paramName });
        this.automations.delete(key);
      }
    }
  }

  /**
   * Get automation track for a parameter
   * @param {string} pluginId - Plugin ID
   * @param {string} paramName - Parameter name
   * @returns {Object|null} Automation track or null
   */
  getAutomation(pluginId, paramName) {
    const key = this.getTrackKey(pluginId, paramName);
    const track = this.automations.get(key);
    return track ? { ...track, events: [...track.events] } : null;
  }

  /**
   * Get all automation tracks
   * @returns {Array} Array of automation tracks
   */
  getAllAutomations() {
    const tracks = [];
    this.automations.forEach((track, key) => {
      tracks.push({
        ...track,
        events: [...track.events]
      });
    });
    return tracks;
  }

  /**
   * Set automation curve type for specific events
   * @param {string} pluginId - Plugin ID
   * @param {string} paramName - Parameter name
   * @param {number} startTime - Start time
   * @param {number} endTime - End time
   * @param {string} curveType - New curve type
   */
  setAutomationCurve(pluginId, paramName, startTime, endTime, curveType) {
    const key = this.getTrackKey(pluginId, paramName);
    const track = this.automations.get(key);

    if (!track) {
      console.warn(`No automation found for ${pluginId}.${paramName}`);
      return;
    }

    track.events.forEach(event => {
      if (event.time >= startTime && event.time <= endTime) {
        event.curveType = curveType;
      }
    });
  }

  /**
   * Add automation point at specific time
   * @param {string} pluginId - Plugin ID
   * @param {string} paramName - Parameter name
   * @param {number} time - Time in seconds
   * @param {number} value - Parameter value
   * @param {string} curveType - Curve type (default: 'linear')
   */
  addAutomationPoint(pluginId, paramName, time, value, curveType = 'linear') {
    this.recordAutomation(pluginId, paramName, value, time, curveType);
  }

  /**
   * Remove automation point at specific time
   * @param {string} pluginId - Plugin ID
   * @param {string} paramName - Parameter name
   * @param {number} time - Time in seconds
   * @param {number} tolerance - Time tolerance in seconds (default: 0.01)
   * @returns {boolean} Success status
   */
  removeAutomationPoint(pluginId, paramName, time, tolerance = 0.01) {
    const key = this.getTrackKey(pluginId, paramName);
    const track = this.automations.get(key);

    if (!track) {
      return false;
    }

    const initialLength = track.events.length;
    track.events = track.events.filter(event =>
      Math.abs(event.time - time) > tolerance
    );

    return track.events.length < initialLength;
  }

  /**
   * Map MIDI CC to parameter
   * @param {number} ccNumber - MIDI CC number (0-127)
   * @param {string} pluginId - Plugin ID
   * @param {string} paramName - Parameter name
   * @param {Object} options - Mapping options
   * @param {number} options.min - Minimum value (default: param min)
   * @param {number} options.max - Maximum value (default: param max)
   */
  mapMIDICC(ccNumber, pluginId, paramName, options = {}) {
    this.midiCCMap.set(ccNumber, {
      pluginId,
      paramName,
      min: options.min,
      max: options.max
    });
  }

  /**
   * Handle MIDI CC message
   * @param {Router} router - Router instance
   * @param {number} ccNumber - MIDI CC number
   * @param {number} value - MIDI value (0-127)
   */
  handleMIDICC(router, ccNumber, value) {
    const mapping = this.midiCCMap.get(ccNumber);
    if (!mapping) {
      return;
    }

    const plugin = router.getPlugin(mapping.pluginId);
    if (!plugin) {
      return;
    }

    const param = plugin.params[mapping.paramName];
    if (!param) {
      return;
    }

    // Normalize MIDI value (0-127) to parameter range
    const normalized = value / 127;
    const desc = plugin.getParameterDescription(mapping.paramName);

    let min = mapping.min !== undefined ? mapping.min : desc.min;
    let max = mapping.max !== undefined ? mapping.max : desc.max;

    const paramValue = min + (normalized * (max - min));

    // Set parameter
    plugin.setParameter(mapping.paramName, paramValue);

    // Record if recording
    if (this.isRecording &&
        this.recordingTrack &&
        this.recordingTrack.pluginId === mapping.pluginId &&
        this.recordingTrack.paramName === mapping.paramName) {
      const time = this.context.currentTime - this.startTime;
      this.recordAutomation(mapping.pluginId, mapping.paramName, paramValue, time);
    }
  }

  /**
   * Export automation as JSON
   * @param {boolean} pretty - Pretty print (default: true)
   * @returns {string} JSON string
   */
  exportAutomation(pretty = true) {
    const exported = {
      version: '1.0.0',
      bpm: this.options.bpm,
      timeSignature: this.options.timeSignature,
      tracks: []
    };

    this.automations.forEach((track, key) => {
      exported.tracks.push({
        pluginId: track.pluginId,
        paramName: track.paramName,
        events: track.events
      });
    });

    return JSON.stringify(exported, null, pretty ? 2 : 0);
  }

  /**
   * Import automation from JSON
   * @param {string} jsonString - JSON string
   * @param {boolean} merge - Merge with existing automation (default: false)
   * @returns {boolean} Success status
   */
  importAutomation(jsonString, merge = false) {
    try {
      const imported = JSON.parse(jsonString);

      if (!merge) {
        this.clearAutomation();
      }

      // Import settings
      if (imported.bpm) {
        this.options.bpm = imported.bpm;
      }
      if (imported.timeSignature) {
        this.options.timeSignature = imported.timeSignature;
      }

      // Import tracks
      if (imported.tracks) {
        imported.tracks.forEach(track => {
          const key = this.getTrackKey(track.pluginId, track.paramName);

          if (merge && this.automations.has(key)) {
            // Merge events
            const existing = this.automations.get(key);
            existing.events.push(...track.events);
            existing.events.sort((a, b) => a.time - b.time);
          } else {
            // Replace
            this.automations.set(key, {
              pluginId: track.pluginId,
              paramName: track.paramName,
              events: track.events
            });
          }
        });
      }

      return true;
    } catch (error) {
      console.error('Failed to import automation:', error);
      return false;
    }
  }

  /**
   * Convert beats to seconds
   * @param {number} beats - Number of beats
   * @returns {number} Time in seconds
   */
  beatsToSeconds(beats) {
    const beatsPerSecond = this.options.bpm / 60;
    return beats / beatsPerSecond;
  }

  /**
   * Convert seconds to beats
   * @param {number} seconds - Time in seconds
   * @returns {number} Number of beats
   */
  secondsToBeats(seconds) {
    const beatsPerSecond = this.options.bpm / 60;
    return seconds * beatsPerSecond;
  }

  /**
   * Set tempo (BPM)
   * @param {number} bpm - Tempo in beats per minute
   */
  setBPM(bpm) {
    this.options.bpm = bpm;
  }

  /**
   * Get tempo (BPM)
   * @returns {number} Current BPM
   */
  getBPM() {
    return this.options.bpm;
  }

  /**
   * Add event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   * @returns {Function} Unsubscribe function
   */
  on(event, callback) {
    if (!this.eventListeners[event]) {
      console.warn(`Unknown event: ${event}`);
      return () => {};
    }

    this.eventListeners[event].push(callback);

    return () => {
      const index = this.eventListeners[event].indexOf(callback);
      if (index > -1) {
        this.eventListeners[event].splice(index, 1);
      }
    };
  }

  /**
   * Emit event
   * @param {string} event - Event name
   * @param {*} data - Event data
   * @private
   */
  emit(event, data) {
    const listeners = this.eventListeners[event];
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in event listener:`, error);
        }
      });
    }
  }

  /**
   * Get statistics
   * @returns {Object} Statistics
   */
  getStats() {
    let totalEvents = 0;
    let minTime = Infinity;
    let maxTime = -Infinity;

    this.automations.forEach(track => {
      totalEvents += track.events.length;
      track.events.forEach(event => {
        minTime = Math.min(minTime, event.time);
        maxTime = Math.max(maxTime, event.time);
      });
    });

    return {
      trackCount: this.automations.size,
      totalEvents,
      duration: maxTime - minTime,
      isPlaying: this.isPlaying,
      isRecording: this.isRecording,
      bpm: this.options.bpm
    };
  }

  /**
   * String representation
   * @returns {string} String description
   */
  toString() {
    return `ParamAutomation(${this.automations.size} tracks, ${this.isPlaying ? 'playing' : 'stopped'})`;
  }
}

export default ParamAutomation;
