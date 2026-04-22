/**
 * MIDI Player — Web Audio sine synth.
 *
 * Previously used soundfont-player (acoustic grand piano). Replaced with
 * a pure sine oscillator so any frequency — including microtonal pitches
 * like quarter-tones — plays cleanly without sample artefacts.
 *
 * Axis-ratio support:
 *   yAxisRatio: frequency multiplier per MIDI-note step. Default
 *               2^(1/12) (standard equal temperament). Set to 2^(1/24)
 *               for quarter-tones, 3/2 for just-intonation fifths, etc.
 *   timeScale:  multiplier applied to note.time on scheduleNotes. Lets
 *               the X-axis setting re-interpret the beat unit without
 *               moving notes on the grid. Default 1.
 */
const REF_MIDI = 69;   // A4
const REF_HZ   = 440;

class MIDIPlayer {
  constructor() {
    this.audioContext = null;
    this.masterGain = null;
    this.activeNodes = [];           // { osc, gain } pairs currently scheduled
    this.isLoaded = false;
    this.yAxisRatio = Math.pow(2, 1 / 12);
    this.timeScale = 1;
  }

  /** Create the AudioContext + masterGain. No sample download. */
  async initialize() {
    if (this.isLoaded) return;
    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      this.masterGain = this.audioContext.createGain();
      this.masterGain.gain.value = 1.0;
      this.masterGain.connect(this.audioContext.destination);
      this.isLoaded = true;
      console.log('🎹 MIDI sine synth ready');
    } catch (error) {
      console.error('❌ Failed to init audio context:', error);
      throw error;
    }
  }

  /** Set the frequency ratio between adjacent MIDI pitches. Default is
   *  2^(1/12) (12-EDO semitone). 2^(1/24) = quarter-tones. */
  setYAxisRatio(ratio) {
    if (Number.isFinite(ratio) && ratio > 0) this.yAxisRatio = ratio;
  }

  /** Multiplier applied to all scheduled note times. 1 = no change,
   *  0.5 = double-tempo, 2 = half-tempo. */
  setTimeScale(s) {
    if (Number.isFinite(s) && s > 0) this.timeScale = s;
  }

  /** MIDI pitch → Hz using the current yAxisRatio. */
  _freqFor(midiNote) {
    return REF_HZ * Math.pow(this.yAxisRatio, midiNote - REF_MIDI);
  }

  /**
   * Play a single note as a sine oscillator.
   * @param {number} midiNote  MIDI pitch (may be fractional; 60.5 etc.)
   * @param {number} velocity  0–1 (scales gain)
   * @param {number} duration  seconds
   * @param {number} time      audioContext time (0 = now)
   * @returns {object|null}    { osc, gain } — use .stop() on osc to cancel
   */
  playNote(midiNote, velocity = 0.8, duration = 1.0, time = 0) {
    if (!this.isLoaded || !this.audioContext) {
      console.warn('⚠️ MIDI player not initialized');
      return null;
    }
    try {
      const ctx = this.audioContext;
      const when = time || ctx.currentTime;
      const freq = this._freqFor(midiNote);
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(Math.max(20, freq), when);
      // Short attack + release envelope so each note doesn't click.
      const peak = 0.25 * Math.max(0, Math.min(1, velocity));
      const attack = 0.005;
      const release = Math.min(0.08, duration * 0.5);
      gain.gain.setValueAtTime(0, when);
      gain.gain.linearRampToValueAtTime(peak, when + attack);
      gain.gain.setValueAtTime(peak, when + Math.max(attack, duration - release));
      gain.gain.exponentialRampToValueAtTime(0.0001, when + duration);
      osc.connect(gain).connect(this.masterGain);
      osc.start(when);
      osc.stop(when + duration + 0.05);
      const node = { osc, gain };
      this.activeNodes.push(node);
      osc.onended = () => {
        const idx = this.activeNodes.indexOf(node);
        if (idx >= 0) this.activeNodes.splice(idx, 1);
      };
      return node;
    } catch (error) {
      console.error('❌ Failed to play note:', error);
      return null;
    }
  }

  /**
   * GM drum hit (Web Audio synth — kick/snare/hh/…). Unchanged from the
   * soundfont era; only pitched playback is sine now.
   */
  playDrum(midiNote, velocity = 0.8, time = 0) {
    if (!this.audioContext) {
      try {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.masterGain = this.audioContext.createGain();
        this.masterGain.gain.value = 1.0;
        this.masterGain.connect(this.audioContext.destination);
      } catch (e) { return null; }
    }
    const ctx = this.audioContext;
    const dest = this.masterGain || ctx.destination;
    const t0 = time || ctx.currentTime;
    const v = Math.max(0, Math.min(1, velocity));

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
      if (hp > 0) { const f = ctx.createBiquadFilter(); f.type = 'highpass'; f.frequency.value = hp; chain.connect(f); chain = f; }
      if (lp < 20000) { const f = ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = lp; chain.connect(f); chain = f; }
      chain.connect(node); node.connect(dest);
      src.start(t0); src.stop(t0 + dur + 0.01);
    };
    const playTone = (freq, dur, type = 'sine', gain = 0.6, sweepTo = null) => {
      const osc = ctx.createOscillator();
      const node = ctx.createGain();
      osc.type = type; osc.frequency.setValueAtTime(freq, t0);
      if (sweepTo !== null) osc.frequency.exponentialRampToValueAtTime(Math.max(20, sweepTo), t0 + dur);
      node.gain.setValueAtTime(gain * v, t0);
      node.gain.exponentialRampToValueAtTime(0.001, t0 + dur);
      osc.connect(node); node.connect(dest);
      osc.start(t0); osc.stop(t0 + dur + 0.01);
    };
    switch (cls) {
      case 'kick':   playTone(120, 0.18, 'sine', 0.9, 40); playNoise(0.03, 0, 8000, 0.2); break;
      case 'snare':  playTone(200, 0.08, 'triangle', 0.4); playNoise(0.18, 1500, 12000, 0.6); break;
      case 'rim':    playNoise(0.04, 2000, 9000, 0.4); playTone(800, 0.04, 'square', 0.2); break;
      case 'hh':     playNoise(0.05, 6000, 16000, 0.5); break;
      case 'hho':    playNoise(0.25, 5000, 16000, 0.5); break;
      case 'tomL':   playTone(110, 0.32, 'sine', 0.8, 60); break;
      case 'tomM':   playTone(160, 0.28, 'sine', 0.8, 80); break;
      case 'tomH':   playTone(220, 0.24, 'sine', 0.8, 110); break;
      case 'crash':  playNoise(1.2, 3000, 16000, 0.55); break;
      case 'ride':   playNoise(0.6, 4000, 14000, 0.4); playTone(2200, 0.4, 'sine', 0.2); break;
      case 'china':  playNoise(0.8, 2500, 14000, 0.55); break;
      case 'splash': playNoise(0.5, 4000, 16000, 0.55); break;
      default:       playNoise(0.12, 800, 8000, 0.45); break;
    }
  }

  /** Stop a specific node handle returned from playNote. */
  stopNote(node) {
    if (node?.osc?.stop) {
      try { node.osc.stop(this.audioContext.currentTime); } catch (_) {}
    }
  }

  /** Instantly mute + stop every scheduled note. */
  stopAll() {
    if (!this.audioContext || !this.masterGain) return;
    const ct = this.audioContext.currentTime;
    this.masterGain.gain.setValueAtTime(0, ct);
    const count = this.activeNodes.length;
    for (const n of this.activeNodes) {
      try { n.osc.stop(ct); } catch (_) {}
    }
    this.activeNodes = [];
    console.log(`🔇 Stopped ${count} MIDI sine notes`);
  }

  /**
   * Schedule multiple notes. `times` are in seconds and multiplied by
   * the current timeScale before scheduling, so the X-axis setting can
   * re-interpret the beat unit without touching stored note data.
   */
  scheduleNotes(notes, bpm = 120, startTime = 0) {
    if (!this.isLoaded) {
      console.warn('⚠️ MIDI player not initialized');
      return;
    }
    const baseTime = startTime || this.audioContext.currentTime;
    for (const { note, time, duration, velocity = 0.8 } of notes) {
      const when = baseTime + time * this.timeScale;
      const dur = duration * this.timeScale;
      this.playNote(note, velocity, dur, when);
    }
  }

  getCurrentTime() {
    return this.audioContext ? this.audioContext.currentTime : 0;
  }

  async resume() {
    if (this.audioContext && this.masterGain) {
      if (this.audioContext.state === 'suspended') await this.audioContext.resume();
      this.masterGain.gain.setValueAtTime(1.0, this.audioContext.currentTime);
    }
  }

  dispose() {
    this.stopAll();
    if (this.audioContext) { this.audioContext.close(); this.audioContext = null; }
    this.isLoaded = false;
  }
}

const midiPlayer = new MIDIPlayer();
export default midiPlayer;
