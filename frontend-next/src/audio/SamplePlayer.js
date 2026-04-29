/**
 * SamplePlayer — single-sample voice for the Doseedo runtime.
 *
 * Backed by Web Audio's AudioBufferSourceNode. Each `play(note, velocity)`
 * spawns a fresh source (browsers disallow restarting a stopped buffer source),
 * connects it through a per-voice gain (velocity scaling + release ramp), and
 * returns a "voice handle" the caller can use to release / hard-stop.
 *
 * Pitch shifting is done with the source node's `playbackRate` AudioParam,
 * which is what HW samplers and EXS24 do under the hood: rate = 2^((note -
 * rootNote) / 12) * cents-detune. Loop start/end is propagated to the source
 * node when `loop` is enabled. Reverse playback copies + reverses the buffer
 * once at construction so each voice can simply play it forward.
 *
 * This class is the single-zone primitive. The multi-zone resolver lives in
 * Keymap.js — it picks zones by note+velocity and instantiates one
 * SamplePlayer per active zone.
 *
 * No external dependencies — pure browser APIs.
 *
 * Usage:
 *   const sp = new SamplePlayer(ctx, { sampleBuffer, rootNote: 60, loop: false });
 *   sp.output.connect(destination);
 *   const voice = sp.play(72, 0.8, ctx.currentTime);  // 72 = octave-up
 *   voice.release(ctx.currentTime + 0.5, 0.05);       // 50ms decay
 *   voice.stop(ctx.currentTime + 1.0);                // hard stop
 */

const DEFAULT_RELEASE_SEC = 0.05;
const HARD_STOP_PADDING_SEC = 0.02;

/**
 * Reverse a buffer's audio data, returning a NEW AudioBuffer. The source
 * buffer is not mutated. Used at construction when `reverse: true` so each
 * voice plays the reversed buffer in the normal forward direction.
 */
function reverseBuffer(ctx, src) {
  const numChannels = src.numberOfChannels;
  const length = src.length;
  const reversed = ctx.createBuffer(numChannels, length, src.sampleRate);
  for (let ch = 0; ch < numChannels; ch++) {
    const srcData = src.getChannelData(ch);
    const dstData = reversed.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      dstData[i] = srcData[length - 1 - i];
    }
  }
  return reversed;
}

/**
 * Map MIDI velocity (0..1 or 0..127) to gain. Linear map is the EXS24
 * default; we expose a `velocityCurve` option for callers that want
 * exponential or logarithmic curves later.
 */
function velocityToGain(velocity, curve = 'linear') {
  let v = velocity;
  if (v > 1) v = v / 127;        // accept either 0-1 or 0-127
  v = Math.max(0, Math.min(1, v));
  if (curve === 'exp')   return v * v;
  if (curve === 'log')   return Math.sqrt(v);
  return v;
}

class SamplePlayer {
  /**
   * @param {BaseAudioContext} ctx
   * @param {Object} opts
   * @param {AudioBuffer} opts.sampleBuffer    REQUIRED: decoded audio buffer
   * @param {number}      [opts.rootNote=60]   MIDI note at native playback rate
   * @param {boolean}     [opts.loop=false]
   * @param {boolean}     [opts.reverse=false]
   * @param {number}      [opts.tuning=0]      Detune in cents
   * @param {number}      [opts.gain=1]        Static zone gain (0..1+)
   * @param {number}      [opts.pan=0]         -1 (L) .. +1 (R), 0 = centre
   * @param {number}      [opts.sampleStart]   Start offset (samples)
   * @param {number}      [opts.sampleEnd]     End offset (samples, exclusive)
   * @param {number}      [opts.loopStart]     Loop start (samples)
   * @param {number}      [opts.loopEnd]       Loop end   (samples)
   * @param {string}      [opts.velocityCurve] 'linear' | 'exp' | 'log'
   */
  constructor(ctx, opts = {}) {
    if (!ctx) throw new Error('SamplePlayer: AudioContext required');
    this.ctx = ctx;

    this.sampleBuffer = opts.sampleBuffer || null;
    this.rootNote = opts.rootNote ?? 60;
    this.loop = !!opts.loop;
    this.reverse = !!opts.reverse;
    this.tuning = opts.tuning || 0;          // cents
    this.staticGain = opts.gain ?? 1;
    this.pan = Math.max(-1, Math.min(1, opts.pan ?? 0));
    this.sampleStart = opts.sampleStart;
    this.sampleEnd = opts.sampleEnd;
    this.loopStart = opts.loopStart;
    this.loopEnd = opts.loopEnd;
    this.velocityCurve = opts.velocityCurve || 'linear';

    // If reversed, pre-build the reversed buffer once. Each voice then plays
    // the reversed buffer forward — much cheaper than reversing per-voice.
    if (this.reverse && this.sampleBuffer) {
      this._playbackBuffer = reverseBuffer(ctx, this.sampleBuffer);
    } else {
      this._playbackBuffer = this.sampleBuffer;
    }

    // Output bus: every voice routes here, then to the caller's destination.
    // This gives zone-level pan + static gain in one place.
    this.output = ctx.createGain();
    this.output.gain.value = 1;

    if (this.pan !== 0 && typeof ctx.createStereoPanner === 'function') {
      this._panner = ctx.createStereoPanner();
      this._panner.pan.value = this.pan;
      this._panner.connect(this.output);
      this._postNode = this._panner;
    } else {
      this._postNode = this.output;
    }

    // Track every live voice so .stop() can clear them all.
    this._liveVoices = new Set();
  }

  /**
   * Compute the playback rate for a note relative to the zone's rootNote,
   * including the cents tuning offset.
   */
  _rateFor(note) {
    const semis = note - this.rootNote;
    const cents = this.tuning;
    return Math.pow(2, semis / 12 + cents / 1200);
  }

  /**
   * Compute the reversed-mode start offset in seconds. When playing a reversed
   * buffer that has logical [sampleStart..sampleEnd] within the original
   * buffer, we need to jump to (length - sampleEnd) in the reversed buffer.
   */
  _startTimeSec(buf) {
    if (this.sampleStart == null) return 0;
    if (!this.reverse) return this.sampleStart / buf.sampleRate;
    const len = buf.length;
    const endSamples = (this.sampleEnd != null) ? this.sampleEnd : len;
    return (len - endSamples) / buf.sampleRate;
  }

  _durationSec(buf) {
    if (this.sampleStart == null && this.sampleEnd == null) return undefined;
    const start = this.sampleStart || 0;
    const end = (this.sampleEnd != null) ? this.sampleEnd : buf.length;
    return Math.max(0, (end - start) / buf.sampleRate);
  }

  /**
   * Trigger a voice. Returns a handle with `.release(time, decay)` and
   * `.stop(time)`. Velocity is 0..1 (or 0..127, auto-detected); the resulting
   * peak gain = staticGain * velocityCurve(velocity).
   *
   * Caller is responsible for ensuring `time >= ctx.currentTime` for live
   * scheduling — for OfflineAudioContext, time is the absolute render time.
   */
  play(note = 60, velocity = 0.8, time = 0) {
    const buf = this._playbackBuffer;
    if (!buf) {
      // No buffer loaded — return an inert handle so callers don't crash.
      return {
        note, velocity, startTime: time,
        release: () => {}, stop: () => {},
        _voiceGain: null, _src: null, _alive: false,
      };
    }

    const t = (time != null) ? time : this.ctx.currentTime;

    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.playbackRate.value = this._rateFor(note);

    // Looping: prefer explicit start/end (samples) > buffer-wide loop.
    if (this.loop) {
      src.loop = true;
      if (this.loopStart != null) src.loopStart = this.loopStart / buf.sampleRate;
      if (this.loopEnd != null)   src.loopEnd   = this.loopEnd / buf.sampleRate;
    }

    const voiceGain = this.ctx.createGain();
    const peak = velocityToGain(velocity, this.velocityCurve) * this.staticGain;
    voiceGain.gain.value = peak;

    src.connect(voiceGain);
    voiceGain.connect(this._postNode);

    // Schedule playback. For region playback, pass an offset + duration.
    const startOffset = this._startTimeSec(buf);
    const dur = this._durationSec(buf);
    try {
      if (dur != null && !this.loop) {
        src.start(t, startOffset, dur);
      } else if (startOffset > 0) {
        src.start(t, startOffset);
      } else {
        src.start(t);
      }
    } catch (e) {
      // start() throws if the source has already been started or t is invalid
      // — both are caller bugs but we don't want to take down the whole graph.
      // eslint-disable-next-line no-console
      if (typeof console !== 'undefined') console.warn('[SamplePlayer] start failed', e);
    }

    // Voice handle
    const handle = {
      note, velocity, startTime: t, _alive: true,
      _src: src, _voiceGain: voiceGain, _player: this,

      release: (releaseTime, decaySec = DEFAULT_RELEASE_SEC) => {
        if (!handle._alive) return;
        const rt = (releaseTime != null) ? releaseTime : this.ctx.currentTime;
        const dt = Math.max(0.001, decaySec);
        try {
          voiceGain.gain.cancelScheduledValues(rt);
          voiceGain.gain.setValueAtTime(voiceGain.gain.value, rt);
          voiceGain.gain.linearRampToValueAtTime(0, rt + dt);
          src.stop(rt + dt + HARD_STOP_PADDING_SEC);
        } catch (e) { /* already stopped */ }
        handle._alive = false;
      },

      stop: (stopTime) => {
        if (!handle._alive) return;
        const st = (stopTime != null) ? stopTime : this.ctx.currentTime;
        try {
          voiceGain.gain.cancelScheduledValues(st);
          voiceGain.gain.setValueAtTime(0, st);
          src.stop(st + HARD_STOP_PADDING_SEC);
        } catch (e) { /* already stopped */ }
        handle._alive = false;
      },
    };

    src.onended = () => {
      handle._alive = false;
      this._liveVoices.delete(handle);
      try { voiceGain.disconnect(); } catch (_) {}
    };

    this._liveVoices.add(handle);
    return handle;
  }

  /** Hard-stop every live voice immediately. */
  stop() {
    const t = this.ctx.currentTime;
    for (const h of Array.from(this._liveVoices)) {
      h.stop(t);
    }
    this._liveVoices.clear();
  }

  /** Tear down all routing nodes. */
  destroy() {
    this.stop();
    try { this.output.disconnect(); } catch (_) {}
    if (this._panner) try { this._panner.disconnect(); } catch (_) {}
  }
}

export default SamplePlayer;
export { reverseBuffer, velocityToGain };
