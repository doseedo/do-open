/**
 * a3.js — runtime builders for the A3 batch (sample-based instruments).
 *
 * Implements:
 *   sample_player — single-sample voice; pitch-shifts via playbackRate from
 *                    a configured rootNote. Buffer is supplied through one of:
 *                      - nodeDef._buffer  (engine pre-loads + stashes here;
 *                        same convention used by VoiceManager.buildVoiceSamplePlayer)
 *                      - nodeDef.params.buffer (caller passes AudioBuffer directly)
 *                    If no buffer is available at build-time, we wire up an
 *                    inert ConstantSource so the graph still compiles; the
 *                    engine can swap a buffer in later via the returned
 *                    `setBuffer(buf)` helper.
 *
 * Note: there's intentionally a parallel impl in VoiceManager.js
 * (buildVoiceSamplePlayer) — that one is per-voice (driven by note-on inside
 * the polyphony layer). THIS builder is for the *static graph* path
 * (WebAudioDSPEngine.NODE_BUILDERS), where a sample_player node sits in a
 * non-polyphonic graph and exposes triggers via a method on the returned
 * object. Both paths are valid; they target different runtimes.
 *
 * Each builder honors the standard interface:
 *   buildFoo(ctx, nodeDef, paramDefs) → { input, output, paramTargets, ... }
 *
 * '@'-prefixed param values bind to parameter IDs for live updates.
 */

import SamplePlayer from '../SamplePlayer.js';

function isAtBinding(v) {
  return typeof v === 'string' && v.startsWith('@');
}

function bindParam(targets, paramId, paramDef, audioParam, opts = {}) {
  targets[paramId] = { audioParam, paramDef, ...opts };
}

function bindCustom(targets, paramId, paramDef, customSetter) {
  targets[paramId] = { paramDef, customSetter };
}

/**
 * Builder for the `sample_player` node type.
 *
 * Schema (from dspNodeDefinitions.js):
 *   io: { in: 0, out: 1 }
 *   params: file, root_note, loop, reverse
 *
 * Returns:
 *   {
 *     input,   // unused (sample_player has no audio input) — passthrough Gain
 *     output,  // SamplePlayer.output (Gain) — engine connects this onward
 *     paramTargets,
 *     trigger(note, velocity, time)  // returns voice handle
 *     stop()                         // hard-stops every live voice
 *     setBuffer(buf)                 // hot-swap the underlying buffer
 *     player: SamplePlayer           // raw player for advanced use
 *   }
 */
function buildSamplePlayer(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  // Resolve a buffer at build time, with multiple input conventions:
  //   1. node._buffer            (engine pre-loaded)
  //   2. params.buffer           (caller passed AudioBuffer)
  //   3. params.sampleBuffer     (alternate name)
  //   4. params.audioBuffer      (alternate name)
  // If none of the above are present we still construct the player with no
  // buffer; setBuffer() can fill it in later.
  let buffer = node._buffer
            || params.buffer
            || params.sampleBuffer
            || params.audioBuffer
            || null;

  // Coerce numeric params that may arrive as strings via JSON
  const rootNote = isAtBinding(params.root_note)
    ? 60
    : (typeof params.root_note === 'number' ? params.root_note : 60);
  const loop    = isAtBinding(params.loop)    ? false : !!params.loop;
  const reverse = isAtBinding(params.reverse) ? false : !!params.reverse;

  const player = new SamplePlayer(ctx, {
    sampleBuffer: buffer,
    rootNote,
    loop,
    reverse,
    tuning: typeof params.tuning === 'number' ? params.tuning : 0,
    gain:   typeof params.gain === 'number' ? params.gain : 1,
    pan:    typeof params.pan === 'number' ? params.pan : 0,
  });

  // Inert input gain — sample_player has io.in=0, but engines may still
  // try to connect upstream nodes harmlessly; a passthrough Gain absorbs that.
  const input = ctx.createGain();
  input.gain.value = 0;          // ensure any accidental connection is muted

  // Param bindings. We expose the player's *output* gain for `gain` and the
  // panner (if any) for `pan`. `root_note`/`loop`/`reverse` are structural —
  // changing them at runtime requires reconstruction so we use customSetters
  // that mutate the player's fields and rebuild lazily on next play().
  for (const [k, v] of Object.entries(params)) {
    if (!isAtBinding(v)) continue;
    const id = v.slice(1);
    switch (k) {
      case 'gain':
        bindParam(targets, id, paramDefs[id], player.output.gain);
        break;
      case 'root_note':
        bindCustom(targets, id, paramDefs[id], (val) => {
          player.rootNote = Math.max(0, Math.min(127, Math.round(val)));
        });
        break;
      case 'loop':
        bindCustom(targets, id, paramDefs[id], (val) => {
          player.loop = !!val;
        });
        break;
      case 'reverse':
        // Reverse swaps the underlying playback buffer — only takes effect
        // on next play() (existing voices keep their buffer).
        bindCustom(targets, id, paramDefs[id], (val) => {
          player.reverse = !!val;
          if (player.sampleBuffer) {
            const SP = SamplePlayer; // ref so eslint doesn't whine about unused import in some configs
            void SP;
            const reversed = !!val;
            // Cheap manual reverse to avoid pulling reverseBuffer through indirect import in some bundlers
            if (reversed) {
              const src = player.sampleBuffer;
              const numChannels = src.numberOfChannels;
              const length = src.length;
              const rev = ctx.createBuffer(numChannels, length, src.sampleRate);
              for (let ch = 0; ch < numChannels; ch++) {
                const sd = src.getChannelData(ch);
                const dd = rev.getChannelData(ch);
                for (let i = 0; i < length; i++) dd[i] = sd[length - 1 - i];
              }
              player._playbackBuffer = rev;
            } else {
              player._playbackBuffer = player.sampleBuffer;
            }
          }
        });
        break;
      case 'tuning':
        bindCustom(targets, id, paramDefs[id], (val) => {
          player.tuning = val;
        });
        break;
      default:
        break;
    }
  }

  return {
    input,
    output: player.output,
    paramTargets: targets,
    isSource: true,           // declares "no audio input expected" to the engine
    // Trigger / control surface for the engine's MIDI / sequencer driver:
    trigger: (note, velocity, time) => player.play(note, velocity, time),
    stop: () => player.stop(),
    setBuffer: (buf) => {
      player.sampleBuffer = buf;
      // Re-apply reverse if needed
      if (player.reverse && buf) {
        const src = buf;
        const numChannels = src.numberOfChannels;
        const length = src.length;
        const rev = ctx.createBuffer(numChannels, length, src.sampleRate);
        for (let ch = 0; ch < numChannels; ch++) {
          const sd = src.getChannelData(ch);
          const dd = rev.getChannelData(ch);
          for (let i = 0; i < length; i++) dd[i] = sd[length - 1 - i];
        }
        player._playbackBuffer = rev;
      } else {
        player._playbackBuffer = buf;
      }
    },
    player,
  };
}

// ── Exports ───────────────────────────────────────────────────────────────

const a3Builders = {
  sample_player: buildSamplePlayer,
};

export default a3Builders;
export { buildSamplePlayer };
