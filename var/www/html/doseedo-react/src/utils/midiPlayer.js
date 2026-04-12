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
   * Play a synthesized drum hit for the given GM percussion note number.
   * Used by MIDIChart when in drum-roll mode so previews don't sound
   * like a piano playing kick/snare. No extra soundfont download — pure
   * Web Audio: noise + sine bursts with envelopes per drum class.
   */
  playDrum(midiNote, velocity = 0.8, time = 0) {
    if (!this.audioContext) {
      try {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.masterGain = this.audioContext.createGain();
        this.masterGain.gain.value = 1.0;
        this.masterGain.connect(this.audioContext.destination);
      } catch (e) {
        return null;
      }
    }
    const ctx = this.audioContext;
    const dest = this.masterGain || ctx.destination;
    const t0 = time || ctx.currentTime;
    const v = Math.max(0, Math.min(1, velocity));

    // Map GM note → drum class (mirrors backend GM_DRUM_NOTE_TO_CHANNEL)
    const cls = (() => {
      if ([35, 36].includes(midiNote)) return 'kick';
      if ([38, 40].includes(midiNote)) return 'snare';
      if ([37, 39].includes(midiNote)) return 'rim';
      if ([42, 44].includes(midiNote)) return 'hh';
      if ([46].includes(midiNote)) return 'hho';
      if ([41, 43].includes(midiNote)) return 'tomL';
      if ([45, 47].includes(midiNote)) return 'tomM';
      if ([48, 50].includes(midiNote)) return 'tomH';
      if ([49, 57].includes(midiNote)) return 'crash';
      if ([51, 59, 53].includes(midiNote)) return 'ride';
      if ([52].includes(midiNote)) return 'china';
      if ([55].includes(midiNote)) return 'splash';
      return 'perc';
    })();

    const playNoise = (dur, hp = 0, lp = 20000, gain = 0.5) => {
      const len = Math.max(1, Math.floor(ctx.sampleRate * dur));
      const buf = ctx.createBuffer(1, len, ctx.sampleRate);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < len; i++) ch[i] = Math.random() * 2 - 1;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      const node = ctx.createGain();
      node.gain.setValueAtTime(gain * v, t0);
      node.gain.exponentialRampToValueAtTime(0.001, t0 + dur);
      let chain = src;
      if (hp > 0) {
        const f = ctx.createBiquadFilter();
        f.type = 'highpass';
        f.frequency.value = hp;
        chain.connect(f); chain = f;
      }
      if (lp < 20000) {
        const f = ctx.createBiquadFilter();
        f.type = 'lowpass';
        f.frequency.value = lp;
        chain.connect(f); chain = f;
      }
      chain.connect(node);
      node.connect(dest);
      src.start(t0);
      src.stop(t0 + dur + 0.01);
    };

    const playTone = (freq, dur, type = 'sine', gain = 0.6, sweepTo = null) => {
      const osc = ctx.createOscillator();
      const node = ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, t0);
      if (sweepTo !== null) {
        osc.frequency.exponentialRampToValueAtTime(Math.max(20, sweepTo), t0 + dur);
      }
      node.gain.setValueAtTime(gain * v, t0);
      node.gain.exponentialRampToValueAtTime(0.001, t0 + dur);
      osc.connect(node);
      node.connect(dest);
      osc.start(t0);
      osc.stop(t0 + dur + 0.01);
    };

    switch (cls) {
      case 'kick':
        playTone(120, 0.18, 'sine', 0.9, 40);
        playNoise(0.03, 0, 8000, 0.2);
        break;
      case 'snare':
        playTone(200, 0.08, 'triangle', 0.4);
        playNoise(0.18, 1500, 12000, 0.6);
        break;
      case 'rim':
        playNoise(0.04, 2000, 9000, 0.4);
        playTone(800, 0.04, 'square', 0.2);
        break;
      case 'hh':
        playNoise(0.05, 6000, 16000, 0.5);
        break;
      case 'hho':
        playNoise(0.25, 5000, 16000, 0.5);
        break;
      case 'tomL':
        playTone(110, 0.32, 'sine', 0.8, 60);
        break;
      case 'tomM':
        playTone(160, 0.28, 'sine', 0.8, 80);
        break;
      case 'tomH':
        playTone(220, 0.24, 'sine', 0.8, 110);
        break;
      case 'crash':
        playNoise(1.2, 3000, 16000, 0.55);
        break;
      case 'ride':
        playNoise(0.6, 4000, 14000, 0.4);
        playTone(2200, 0.4, 'sine', 0.2);
        break;
      case 'china':
        playNoise(0.8, 2500, 14000, 0.55);
        break;
      case 'splash':
        playNoise(0.5, 4000, 16000, 0.55);
        break;
      default:
        playNoise(0.12, 800, 8000, 0.45);
        break;
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
