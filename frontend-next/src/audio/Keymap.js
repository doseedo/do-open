/**
 * Keymap — multi-zone resolver for sample-based instruments.
 *
 * A "keymap" is a list of zones (regions of the MIDI key + velocity space).
 * Each zone declares which sample to play, the MIDI range it responds to,
 * the velocity range, optional tuning, optional region within the buffer,
 * loop config, and an optional `roundRobin` group ID.
 *
 * Zone selection rules (mirror EXS24 / Kontakt-style behaviour):
 *   1. Filter zones by note ∈ [keyRange.low, keyRange.high].
 *   2. Filter the remaining zones by velocity ∈ [velRange.low, velRange.high].
 *      Velocities are matched in 1..127 space (we accept 0..1 too and scale).
 *   3. Group remaining zones by `roundRobin` ID:
 *      - Zones with the SAME roundRobin ID form a cycle. Each trigger picks
 *        ONE zone from the group, advancing a per-group counter so successive
 *        notes hit successive zones (a → b → c → a → ...).
 *      - Zones with NO roundRobin ID (or unique IDs) are considered standalone:
 *        every match plays.
 *
 * The keymap is the "voice map" layer — actual audio rendering is delegated
 * to SamplePlayer instances created lazily per zone (one SamplePlayer object
 * per zone, reused across triggers, with its own .play() spawning per-voice
 * BufferSourceNodes).
 *
 * This class is independent of VoiceManager — it doesn't know about ADSR,
 * unison, modulation, etc. The engine wires it into the polyphony layer
 * separately (see INTEGRATION_A3.md).
 */

import SamplePlayer from './SamplePlayer.js';

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/** Return true if `note` is inside the inclusive zone keyRange. */
function noteInRange(note, keyRange) {
  if (!keyRange) return true;
  const lo = keyRange[0] ?? keyRange.low ?? 0;
  const hi = keyRange[1] ?? keyRange.high ?? 127;
  return note >= lo && note <= hi;
}

/** Return true if `velocity` (1..127) is inside the inclusive zone velRange. */
function velInRange(velocity1to127, velRange) {
  if (!velRange) return true;
  const lo = velRange[0] ?? velRange.low ?? 1;
  const hi = velRange[1] ?? velRange.high ?? 127;
  return velocity1to127 >= lo && velocity1to127 <= hi;
}

/** Accept 0..1 or 0..127 velocity → 1..127 (used for range comparisons). */
function vel1to127(v) {
  if (v == null) return 64;
  if (v <= 1) return Math.max(1, Math.round(v * 127));
  return Math.max(1, Math.min(127, Math.round(v)));
}

class Keymap {
  /**
   * @param {BaseAudioContext} ctx
   * @param {Array<Object>}    zones    Array of zone descriptors. Each zone:
   *   {
   *     keyRange:    [low, high]       (MIDI 0..127, inclusive)
   *     velRange:    [low, high]       (MIDI 1..127, inclusive)
   *     sampleBuffer:AudioBuffer       REQUIRED
   *     rootNote:    number            (default 60)
   *     tuning:      cents             (optional)
   *     sampleStart: samples           (optional region start)
   *     sampleEnd:   samples           (optional region end)
   *     loop:        { mode, start, end }
   *                  mode ∈ 'off' | 'sustain' | 'all'
   *     gain:        0..1              (default 1)
   *     pan:         -1..1             (default 0)
   *     roundRobin:  number            (group id, optional)
   *     velocityCurve: 'linear'|'exp'|'log'
   *   }
   * @param {Object} options
   *   destination {AudioNode} optional shared output bus; if omitted,
   *     each zone's SamplePlayer.output is exposed via .output (a master gain).
   */
  constructor(ctx, zones = [], options = {}) {
    if (!ctx) throw new Error('Keymap: AudioContext required');
    this.ctx = ctx;
    this.zones = (zones || []).map((z, i) => this._normalizeZone(z, i));

    // Shared output bus — every zone's player connects here.
    this.output = ctx.createGain();
    this.output.gain.value = options.masterGain ?? 1;

    // Round-robin counters: roundRobinId → next index into the matching zones
    // for that group. Reset on .reset().
    this._rrCounters = new Map();

    // Build one SamplePlayer per zone. Reusing across triggers avoids
    // re-allocating the panner / output gain every voice.
    this._players = this.zones.map(z => this._makePlayer(z));
    for (const p of this._players) {
      try { p.output.connect(this.output); } catch (_) {}
    }

    if (options.destination) {
      try { this.output.connect(options.destination); } catch (_) {}
    }
  }

  _normalizeZone(z, idx) {
    return {
      _idx: idx,
      keyRange: z.keyRange || [0, 127],
      velRange: z.velRange || [1, 127],
      sampleBuffer: z.sampleBuffer,
      rootNote: z.rootNote ?? 60,
      tuning: z.tuning || 0,
      sampleStart: z.sampleStart,
      sampleEnd: z.sampleEnd,
      loop: z.loop || { mode: 'off' },
      gain: z.gain ?? 1,
      pan: clamp(z.pan ?? 0, -1, 1),
      roundRobin: z.roundRobin ?? null,
      velocityCurve: z.velocityCurve || 'linear',
      reverse: !!z.reverse,
    };
  }

  _makePlayer(z) {
    const loopMode = (z.loop && z.loop.mode) || 'off';
    return new SamplePlayer(this.ctx, {
      sampleBuffer: z.sampleBuffer,
      rootNote: z.rootNote,
      loop: loopMode === 'all' || loopMode === 'sustain',
      reverse: z.reverse,
      tuning: z.tuning,
      gain: z.gain,
      pan: z.pan,
      sampleStart: z.sampleStart,
      sampleEnd: z.sampleEnd,
      loopStart: z.loop?.start,
      loopEnd:   z.loop?.end,
      velocityCurve: z.velocityCurve,
    });
  }

  /**
   * Pick zones for a note + velocity, advancing round-robin counters as a
   * side effect. Pure resolution (no audio side effects beyond counter advance).
   * Returns an array of zone descriptors.
   */
  selectZones(note, velocity = 0.8) {
    const v127 = vel1to127(velocity);
    const candidates = this.zones.filter(z =>
      noteInRange(note, z.keyRange) && velInRange(v127, z.velRange)
    );
    if (candidates.length === 0) return [];

    // Group by roundRobin ID. Zones with rr=null go in their own bucket
    // (each plays unconditionally).
    const groups = new Map();
    const standalone = [];
    for (const z of candidates) {
      if (z.roundRobin == null) {
        standalone.push(z);
      } else {
        if (!groups.has(z.roundRobin)) groups.set(z.roundRobin, []);
        groups.get(z.roundRobin).push(z);
      }
    }

    const picked = [...standalone];
    for (const [rrId, group] of groups) {
      // Stable order: by zone index so the cycle is deterministic
      group.sort((a, b) => a._idx - b._idx);
      const counter = this._rrCounters.get(rrId) || 0;
      const next = group[counter % group.length];
      this._rrCounters.set(rrId, counter + 1);
      picked.push(next);
    }
    return picked;
  }

  /**
   * Trigger zones matching this note+velocity. Returns an array of voice
   * handles (one per matched zone). Caller is responsible for storing handles
   * and calling .release()/.stop() on them.
   */
  trigger(note, velocity = 0.8, time = 0) {
    const zones = this.selectZones(note, velocity);
    const handles = [];
    for (const z of zones) {
      const player = this._players[z._idx];
      if (!player) continue;
      const h = player.play(note, velocity, time);
      handles.push(h);
    }
    return handles;
  }

  /** Hard-stop every player. */
  stopAll() {
    for (const p of this._players) {
      try { p.stop(); } catch (_) {}
    }
  }

  /** Reset round-robin counters (e.g. on song reset / transport stop). */
  reset() {
    this._rrCounters.clear();
  }

  destroy() {
    this.stopAll();
    for (const p of this._players) {
      try { p.destroy(); } catch (_) {}
    }
    try { this.output.disconnect(); } catch (_) {}
  }

  /** Number of zones in the map. */
  get zoneCount() { return this.zones.length; }
}

export default Keymap;
export { noteInRange, velInRange, vel1to127 };
