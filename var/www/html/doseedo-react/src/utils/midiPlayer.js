import Soundfont from 'soundfont-player';

/**
 * MIDI Player - Handles MIDI playback using Web Audio API and soundfonts
 */
class MIDIPlayer {
  constructor() {
    this.audioContext = null;
    this.instrument = null;
    this.masterGain = null; // Master gain node for instant mute
    this.scheduledNotes = []; // Track scheduled notes for cleanup
    this.isLoaded = false;
    this.instrumentName = 'acoustic_grand_piano';
  }

  /**
   * Initialize the audio context and load the soundfont
   */
  async initialize() {
    if (this.isLoaded) return;

    try {
      // Create audio context
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

      // Create master gain node for instant mute capability
      this.masterGain = this.audioContext.createGain();
      this.masterGain.gain.value = 1.0;
      this.masterGain.connect(this.audioContext.destination);

      console.log('🎹 Loading piano soundfont...');

      // Load acoustic piano soundfont from CDN
      // Connect it to our master gain node instead of destination
      this.instrument = await Soundfont.instrument(
        this.audioContext,
        this.instrumentName,
        {
          soundfont: 'MusyngKite', // High quality soundfont
          destination: this.masterGain // Route through master gain
        }
      );

      this.isLoaded = true;
      console.log('✅ Piano soundfont loaded');
    } catch (error) {
      console.error('❌ Failed to load soundfont:', error);
      throw error;
    }
  }

  /**
   * Play a single MIDI note
   * @param {number} midiNote - MIDI note number (0-127)
   * @param {number} velocity - Note velocity (0-1)
   * @param {number} duration - Duration in seconds
   * @param {number} time - When to play (audioContext time, 0 = now)
   * @returns {object} note - The scheduled note object
   */
  playNote(midiNote, velocity = 0.8, duration = 1.0, time = 0) {
    if (!this.isLoaded || !this.instrument) {
      console.warn('⚠️ MIDI player not initialized');
      return null;
    }

    try {
      const when = time || this.audioContext.currentTime;

      // Play the note - returns an audio node
      const node = this.instrument.play(midiNote, when, {
        duration: duration,
        gain: velocity
      });

      // Track this note for cleanup
      if (node) {
        this.scheduledNotes.push(node);
      }

      return node;
    } catch (error) {
      console.error('❌ Failed to play note:', error);
      return null;
    }
  }

  /**
   * Stop a specific note
   * @param {object} node - Note node returned from playNote
   */
  stopNote(node) {
    if (node && node.stop) {
      try {
        const now = this.audioContext.currentTime;
        node.stop(now);
      } catch (error) {
        // Note might have already stopped
      }
    }
  }

  /**
   * Stop all currently playing notes immediately using master gain mute
   */
  stopAll() {
    if (!this.audioContext || !this.masterGain) return;

    const nodeCount = this.scheduledNotes.length;

    // Instantly mute all MIDI by setting master gain to 0
    // This stops all sound immediately, even pre-scheduled notes
    this.masterGain.gain.setValueAtTime(0, this.audioContext.currentTime);

    // Stop all scheduled nodes to prevent them from playing later
    const now = this.audioContext.currentTime;
    this.scheduledNotes.forEach((node) => {
      if (node && node.stop) {
        try {
          node.stop(now);
        } catch (error) {
          // Node might have already stopped or not started yet
        }
      }
    });

    // Clear scheduled notes array
    this.scheduledNotes = [];

    console.log(`🔇 Muted all MIDI via master gain and stopped ${nodeCount} nodes`);
  }

  /**
   * Schedule multiple notes to play
   * @param {Array} notes - Array of {note, time, duration, velocity}
   *                        where time and duration are in SECONDS
   * @param {number} bpm - Tempo in beats per minute (not used, for backward compatibility)
   * @param {number} startTime - When to start playing (audioContext time)
   */
  scheduleNotes(notes, bpm = 120, startTime = 0) {
    if (!this.isLoaded) {
      console.warn('⚠️ MIDI player not initialized');
      return;
    }

    const baseTime = startTime || this.audioContext.currentTime;

    console.log(`🎵 Scheduling ${notes.length} notes (times already in seconds)`);

    notes.forEach(({ note, time, duration, velocity = 0.8 }) => {
      // Note: time and duration are already in seconds from MIDI parser!
      // No conversion needed - use them directly
      const playTime = baseTime + time;

      this.playNote(note, velocity, duration, playTime);
    });
  }

  /**
   * Get current playback time
   */
  getCurrentTime() {
    return this.audioContext ? this.audioContext.currentTime : 0;
  }

  /**
   * Resume audio context if suspended and restore master gain
   */
  async resume() {
    if (this.audioContext && this.masterGain) {
      // Resume audio context if suspended
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }
      // Restore master gain to full volume
      this.masterGain.gain.setValueAtTime(1.0, this.audioContext.currentTime);
      console.log('🔊 Restored MIDI master gain');
    }
  }

  /**
   * Cleanup and release resources
   */
  dispose() {
    this.stopAll();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    this.instrument = null;
    this.isLoaded = false;
  }
}

// Create singleton instance
const midiPlayer = new MIDIPlayer();

export default midiPlayer;
