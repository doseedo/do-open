/**
 * MidiInput — Web MIDI API adapter that drives a VoiceManager (and optional
 * modulation router for CC → param mapping).
 *
 * Usage:
 *   const vm = new VoiceManager(ctx, template, paramDefs);
 *   const midi = new MidiInput(vm);
 *   await midi.attachWebMidi();          // optional — only if browser exposes Web MIDI
 *   midi.bindCC(74, 'cutoff');           // map CC #74 → 'cutoff' shared knob
 *   midi.feed({ type: 'noteon', note: 60, velocity: 0.8 });    // programmatic
 *
 * Event format mirrors the existing internal MIDI shape used by midiPlayer.js:
 *   { type: 'noteon' | 'noteoff' | 'cc' | 'pitchbend' | 'sustain',
 *     note?, velocity?, cc?, value?, time? }
 *
 * Web MIDI raw status/data1/data2 bytes are decoded into this format before
 * being fed to the same `feed` pipeline, so programmatic events and hardware
 * events share the exact same routing/transform path.
 */

const STATUS_NOTE_OFF = 0x80;
const STATUS_NOTE_ON = 0x90;
const STATUS_CC = 0xB0;
const STATUS_PITCH_BEND = 0xE0;

export default class MidiInput {
  /**
   * @param {VoiceManager} voiceManager
   * @param {Object} [modRouter] optional { setParam(paramId, value) } router that
   *   sits between CC and the voice manager (e.g. for modulation depth/curves).
   *   When absent, bindCC writes directly to voiceManager.setParam.
   */
  constructor(voiceManager, modRouter = null) {
    this.voiceManager = voiceManager;
    this.modRouter = modRouter;
    this.ccBindings = new Map();   // ccNumber → paramId (or callback)
    this.midiAccess = null;
    this.inputs = [];
    this._listeners = new WeakMap(); // input → handler
    this._pitchBendRange = 2;       // semitones, +/- 2 default
    this._currentBend = 0;          // semitones
  }

  // ── Web MIDI hookup ─────────────────────────────────────────────────────────

  async attachWebMidi() {
    if (typeof navigator === 'undefined' || !navigator.requestMIDIAccess) {
      // Headless / unsupported — programmatic feed() still works.
      return false;
    }
    try {
      this.midiAccess = await navigator.requestMIDIAccess({ sysex: false });
    } catch (e) {
      return false;
    }
    this._refreshInputs();
    this.midiAccess.onstatechange = () => this._refreshInputs();
    return true;
  }

  _refreshInputs() {
    if (!this.midiAccess) return;
    // Detach old
    for (const input of this.inputs) {
      const h = this._listeners.get(input);
      if (h) input.removeEventListener('midimessage', h);
    }
    this.inputs = [];
    // Attach to all current inputs
    for (const input of this.midiAccess.inputs.values()) {
      const handler = (event) => this._onMessage(event.data);
      input.addEventListener('midimessage', handler);
      this._listeners.set(input, handler);
      this.inputs.push(input);
    }
  }

  /** Decode raw MIDI bytes and dispatch through `feed`. */
  _onMessage(data) {
    if (!data || data.length < 1) return;
    const status = data[0] & 0xF0;
    const d1 = data[1] | 0;
    const d2 = data[2] | 0;
    if (status === STATUS_NOTE_ON) {
      // Note-on with velocity 0 = note-off (running-status convention)
      if (d2 === 0) this.feed({ type: 'noteoff', note: d1, velocity: 0 });
      else this.feed({ type: 'noteon', note: d1, velocity: d2 / 127 });
    } else if (status === STATUS_NOTE_OFF) {
      this.feed({ type: 'noteoff', note: d1, velocity: d2 / 127 });
    } else if (status === STATUS_CC) {
      if (d1 === 64) {
        // CC#64 = sustain pedal
        this.feed({ type: 'sustain', value: d2 >= 64 });
      } else {
        this.feed({ type: 'cc', cc: d1, value: d2 / 127 });
      }
    } else if (status === STATUS_PITCH_BEND) {
      const raw = (d2 << 7) | d1;     // 14-bit
      const norm = (raw - 8192) / 8192; // -1 .. +1
      this.feed({ type: 'pitchbend', value: norm });
    }
  }

  // ── Programmatic event injection ────────────────────────────────────────────

  feed(event) {
    if (!event || !event.type) return;
    const t = event.time;       // null/undefined → now
    const vm = this.voiceManager;
    switch (event.type) {
      case 'noteon': {
        if (event.velocity === 0) {
          vm.noteOff(event.note, t);
        } else {
          vm.noteOn(event.note, event.velocity ?? 0.8, t);
        }
        break;
      }
      case 'noteoff': {
        vm.noteOff(event.note, t);
        break;
      }
      case 'cc': {
        const binding = this.ccBindings.get(event.cc);
        if (!binding) return;
        const value = event.value;  // 0..1 already
        if (typeof binding === 'function') {
          try { binding(value); } catch (_) {}
        } else if (binding.paramId) {
          // Optional remap to [min,max] if specified
          let v = value;
          if (binding.min != null && binding.max != null) {
            v = binding.min + (binding.max - binding.min) * value;
          }
          if (this.modRouter && typeof this.modRouter.setParam === 'function') {
            this.modRouter.setParam(binding.paramId, v);
          } else {
            vm.setParam(binding.paramId, v);
          }
        }
        break;
      }
      case 'sustain': {
        vm.setSustain(!!event.value);
        break;
      }
      case 'pitchbend': {
        // Convert to semitones, retune all live voices by delta
        const semis = (event.value || 0) * this._pitchBendRange;
        this._applyPitchBend(semis, t);
        break;
      }
      case 'panic':
      case 'allnotesoff': {
        for (const note of vm.activeNotes()) vm.noteOff(note, t);
        break;
      }
      default:
        break;
    }
  }

  _applyPitchBend(semis, time) {
    // Re-pitch every voice's '@pitch' targets by delta. We use the voice's
    // base note + bend semitones to compute new Hz.
    this._currentBend = semis;
    const ctx = this.voiceManager.ctx;
    const t = time != null ? time : ctx.currentTime;
    for (const voice of this.voiceManager._allVoices) {
      const baseFreq = 440 * Math.pow(2, ((voice.note + semis) - 69) / 12);
      for (const built of Object.values(voice.builtNodes)) {
        const targets = built.perVoice?.['@pitch'];
        if (!targets) continue;
        const list = Array.isArray(targets) ? targets : [targets];
        for (const tgt of list) {
          if (!tgt.audioParam) continue;
          const oct = tgt.octaveOffset || 0;
          const scale = tgt.scale || 1;
          try {
            tgt.audioParam.cancelScheduledValues(t);
            tgt.audioParam.setValueAtTime(baseFreq * Math.pow(2, oct) * scale, t);
          } catch (_) {}
        }
      }
    }
  }

  // ── CC routing ──────────────────────────────────────────────────────────────

  /**
   * @param {number} cc
   * @param {string|Function|Object} target
   *   - string: paramId (writes 0..1 directly via voiceManager.setParam)
   *   - function: called with the 0..1 value
   *   - object: { paramId, min?, max? } — min/max remap from 0..1
   */
  bindCC(cc, target) {
    if (typeof target === 'string') {
      this.ccBindings.set(cc, { paramId: target });
    } else {
      this.ccBindings.set(cc, target);
    }
  }

  unbindCC(cc) {
    this.ccBindings.delete(cc);
  }

  setPitchBendRange(semitones) {
    this._pitchBendRange = Math.max(0, semitones);
  }

  detachWebMidi() {
    for (const input of this.inputs) {
      const h = this._listeners.get(input);
      if (h) input.removeEventListener('midimessage', h);
    }
    this.inputs = [];
    if (this.midiAccess) this.midiAccess.onstatechange = null;
    this.midiAccess = null;
  }

  destroy() {
    this.detachWebMidi();
    this.ccBindings.clear();
  }
}
