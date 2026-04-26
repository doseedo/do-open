/**
 * R13 — Vocoder builder
 *
 * Registers the `vocoder` node type. Internally it constructs an
 * AudioWorkletNode running `r13-vocoder-processor` (a classic N-band phase
 * vocoder: analysis filter bank + envelope followers + synthesis filter
 * bank, with optional internal carrier oscillator or external carrier
 * sidechain).
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildVocoder(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets, inputs }
 *
 * The Vocoder node has TWO inputs:
 *   inputs[0] — modulator (voice / drum loop) — MUST be wired
 *   inputs[1] — carrier (external) — wired only if carrier_type === 'external'
 *
 * The default linear `dspChain` engine wiring will use `input` for slot 0
 * (modulator). Engines that explicitly handle multi-input nodes can use
 * the returned `inputs` array.
 *
 * Following the R4/R9 convention: if the worklet processor isn't registered
 * yet on the AudioContext (e.g. addModule hasn't completed), we fall back
 * to a primitive all-WebAudio implementation built from BiquadFilterNodes
 * + AnalyserNode-driven gain control. The fallback isn't audio-faithful
 * (the analyser is k-rate, not sample-accurate), but the graph still binds
 * and audio still flows so calibration passes that hit the worklet on the
 * second build will succeed.
 *
 * Author: Doseedo R13
 */

const R13_VOCODER = 'r13-vocoder-processor';

// Mirrors the worklet's enum
const CARRIER_NAME_TO_INDEX = {
  saw:      0,
  square:   1,
  noise:    2,
  external: 3,
};

const BANDS_TO_IDX = { 8: 0, 16: 1, 24: 2, 32: 3 };
const IDX_TO_BANDS = [8, 16, 24, 32];

function _carrierIndex(value) {
  if (typeof value === 'number') return Math.max(0, Math.min(3, Math.round(value)));
  if (typeof value === 'string') {
    const idx = CARRIER_NAME_TO_INDEX[value.toLowerCase()];
    return idx == null ? 0 : idx;
  }
  return 0;
}

function _bandsIndex(value) {
  if (typeof value === 'number') {
    // Accept either the literal band count (8/16/24/32) or the enum (0..3).
    if (BANDS_TO_IDX[value] != null) return BANDS_TO_IDX[value];
    return Math.max(0, Math.min(3, Math.round(value)));
  }
  return 1; // default 16
}

function _safeWorklet(ctx, name, options = {}) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R13] worklet ${name} unavailable, using fallback:`, e && e.message);
    }
    return null;
  }
}

// Logarithmic band centers between lo..hi
function _logBandCenters(n, lo, hi) {
  const out = [];
  if (n <= 1) return [Math.sqrt(lo * hi)];
  const ratio = Math.log(hi / lo) / (n - 1);
  for (let i = 0; i < n; i++) out.push(lo * Math.exp(i * ratio));
  return out;
}

// ── Fallback: build an N-band primitive vocoder out of BiquadFilterNodes
// and per-band Gain controls. The analysis envelope is approximated with
// AnalyserNode taps; the resulting CV is updated at ~60 Hz, which is fine
// for slow speech but will smear transients. Used only when the worklet
// hasn't loaded yet.
function _buildFallback(ctx, params) {
  const input = ctx.createGain();    // modulator input
  const carrierIn = ctx.createGain(); // external carrier input
  const output = ctx.createGain();
  output.gain.value = 1;

  const bands = IDX_TO_BANDS[_bandsIndex(params.bands)];
  const formantSemis = (typeof params.formant_shift === 'number') ? params.formant_shift : 0;
  const shift = Math.pow(2, formantSemis / 12);
  const anaCenters = _logBandCenters(bands, 100, 8000);
  const synCenters = anaCenters.map((f) => f * shift);
  const Qval = (typeof params.q === 'number') ? params.q : 12;

  // Decide carrier source
  const carrierType = (typeof params.carrier_type === 'string')
    ? params.carrier_type.toLowerCase()
    : 'saw';

  let internalOsc = null;
  let internalNoise = null;
  if (carrierType === 'noise') {
    // White-noise via small random buffer played in a loop
    const dur = 1.0;
    const buf = ctx.createBuffer(1, Math.floor(ctx.sampleRate * dur), ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    internalNoise = ctx.createBufferSource();
    internalNoise.buffer = buf;
    internalNoise.loop = true;
    try { internalNoise.start(); } catch (e) { /* ignore */ }
  } else if (carrierType !== 'external') {
    internalOsc = ctx.createOscillator();
    internalOsc.type = (carrierType === 'square') ? 'square' : 'sawtooth';
    internalOsc.frequency.value = (typeof params.carrier_freq === 'number') ? params.carrier_freq : 110;
    try { internalOsc.start(); } catch (e) { /* ignore */ }
  }

  // For each band: analyser on modulator, BP on carrier, gain controlled by analyser.
  const analyserBands = [];
  const synBands = [];
  const synGains = [];
  const synthBus = ctx.createGain();
  synthBus.gain.value = 1 / Math.sqrt(bands);
  synthBus.connect(output);

  // Carrier source connect target
  const carrierTap = ctx.createGain();
  carrierTap.gain.value = 1;
  if (internalOsc) internalOsc.connect(carrierTap);
  else if (internalNoise) internalNoise.connect(carrierTap);
  else carrierIn.connect(carrierTap);

  for (let i = 0; i < bands; i++) {
    const ana = ctx.createBiquadFilter();
    ana.type = 'bandpass';
    ana.frequency.value = anaCenters[i];
    ana.Q.value = Qval;
    input.connect(ana);
    const an = ctx.createAnalyser();
    an.fftSize = 256;
    an.smoothingTimeConstant = 0.5;
    ana.connect(an);
    analyserBands.push(an);

    const syn = ctx.createBiquadFilter();
    syn.type = 'bandpass';
    syn.frequency.value = synCenters[i];
    syn.Q.value = Qval;
    carrierTap.connect(syn);
    synBands.push(syn);
    const g = ctx.createGain();
    g.gain.value = 0;
    syn.connect(g);
    g.connect(synthBus);
    synGains.push(g);
  }

  // Dry/wet
  const dryG = ctx.createGain();
  const wetG = ctx.createGain();
  const initMix = (typeof params.mix === 'number') ? params.mix : 1;
  dryG.gain.value = 1 - initMix;
  wetG.gain.value = initMix;
  input.connect(dryG);
  dryG.connect(output);
  // synthBus already → output; rewire through wetG instead
  try { synthBus.disconnect(output); } catch (e) { /* may not be connected on early build */ }
  synthBus.connect(wetG);
  wetG.connect(output);

  // CV update loop: read each analyser, set per-band gain (~60 Hz)
  const tmp = new Uint8Array(256);
  let stopped = false;
  const tick = () => {
    if (stopped) return;
    try {
      for (let i = 0; i < analyserBands.length; i++) {
        analyserBands[i].getByteTimeDomainData(tmp);
        let sum = 0;
        for (let k = 0; k < tmp.length; k++) {
          const v = (tmp[k] - 128) / 128;
          sum += v * v;
        }
        const rms = Math.sqrt(sum / tmp.length);
        synGains[i].gain.setTargetAtTime(rms, ctx.currentTime, 0.02);
      }
    } catch (e) { /* ignore */ }
  };
  const intervalId = setInterval(tick, 16);
  output._r13_cleanup = () => {
    stopped = true;
    clearInterval(intervalId);
    try { if (internalOsc) internalOsc.stop(); } catch (e) { /* ignore */ }
    try { if (internalNoise) internalNoise.stop(); } catch (e) { /* ignore */ }
  };

  return {
    input,
    output,
    inputs: [input, carrierIn],
    fallback: {
      analyserBands, synBands, synGains, dryG, wetG, internalOsc, internalNoise,
      carrierIn, carrierTap, synthBus,
    },
  };
}

// ── vocoder builder ───────────────────────────────────────────────────────
//
// Schema:
//   {
//     type: 'vocoder',
//     params: {
//       bands:         8 | 16 | 24 | 32 | '@<id>'   (default 16)
//       attack_ms:     number (0.1..100) | '@<id>'
//       release_ms:    number (1..500)   | '@<id>'
//       formant_shift: number (-12..12)  | '@<id>'  (semitones)
//       carrier_type:  'saw'|'square'|'noise'|'external' | '@<id>'
//       carrier_freq:  number (Hz)       | '@<id>'
//       mix:           number (0..1)     | '@<id>'  (dry/wet)
//       unvoiced_mix:  number (0..1)     | '@<id>'  (HF noise blend)
//       q:             number (1..50)    | '@<id>'  (filter Q)
//     }
//   }
export function buildVocoder(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  // Initial values used at construction
  const initialBandsIdx = _bandsIndex(typeof params.bands === 'string' && params.bands.startsWith('@')
    ? 16 : (params.bands ?? 16));
  const initialCarrier  = _carrierIndex(typeof params.carrier_type === 'string' && params.carrier_type.startsWith('@')
    ? 'saw' : (params.carrier_type ?? 'saw'));
  const initialAttack   = (typeof params.attack_ms === 'number') ? params.attack_ms : 5;
  const initialRelease  = (typeof params.release_ms === 'number') ? params.release_ms : 50;
  const initialFormant  = (typeof params.formant_shift === 'number') ? params.formant_shift : 0;
  const initialFreq     = (typeof params.carrier_freq === 'number') ? params.carrier_freq : 110;
  const initialMix      = (typeof params.mix === 'number') ? params.mix : 1;
  const initialUnvoiced = (typeof params.unvoiced_mix === 'number') ? params.unvoiced_mix : 0.2;
  const initialQ        = (typeof params.q === 'number') ? params.q : 12;

  // ── Try worklet first ─────────────────────────────────────────────────────
  const worklet = _safeWorklet(ctx, R13_VOCODER, {
    numberOfInputs: 2,                 // [0]=modulator, [1]=carrier
    numberOfOutputs: 1,
    outputChannelCount: [2],
    parameterData: {
      bands_idx:     initialBandsIdx,
      attack_ms:     initialAttack,
      release_ms:    initialRelease,
      formant_shift: initialFormant,
      carrier_type:  initialCarrier,
      carrier_freq:  initialFreq,
      mix:           initialMix,
      unvoiced_mix:  initialUnvoiced,
      q:             initialQ,
    },
  });

  if (worklet) {
    // Wrap the worklet inputs so the engine can connect via .connect()
    // and so we present a stable `input` (slot 0) reference.
    const modIn = ctx.createGain();
    const carIn = ctx.createGain();
    // Connect each pre-input gain to the matching worklet input slot.
    try { modIn.connect(worklet, 0, 0); } catch (e) { modIn.connect(worklet); }
    try { carIn.connect(worklet, 0, 1); } catch (e) { /* ignore — single-input path */ }

    const wpar = (name) => (worklet.parameters && worklet.parameters.get(name)) || null;

    const bind = (paramId, key) => {
      const ap = wpar(key);
      if (!ap) return;
      targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
    };
    const bindCustom = (paramId, setter) => {
      targets[paramId] = { paramDef: paramDefs[paramId], customSetter: setter };
    };

    for (const [k, v] of Object.entries(params)) {
      const isAt = (typeof v === 'string' && v.startsWith('@'));
      const id = isAt ? v.slice(1) : null;

      switch (k) {
        case 'bands': {
          if (isAt) {
            bindCustom(id, (val) => {
              const ap = wpar('bands_idx');
              if (ap) ap.value = _bandsIndex(val);
            });
          } else if (v != null) {
            const ap = wpar('bands_idx');
            if (ap) ap.value = _bandsIndex(v);
          }
          break;
        }
        case 'attack_ms':
        case 'attack':
          if (isAt) bind(id, 'attack_ms');
          else if (typeof v === 'number') { const ap = wpar('attack_ms'); if (ap) ap.value = v; }
          break;
        case 'release_ms':
        case 'release':
          if (isAt) bind(id, 'release_ms');
          else if (typeof v === 'number') { const ap = wpar('release_ms'); if (ap) ap.value = v; }
          break;
        case 'formant_shift':
        case 'formant':
          if (isAt) bind(id, 'formant_shift');
          else if (typeof v === 'number') { const ap = wpar('formant_shift'); if (ap) ap.value = v; }
          break;
        case 'carrier_type': {
          if (isAt) {
            bindCustom(id, (val) => {
              const ap = wpar('carrier_type');
              if (ap) ap.value = _carrierIndex(val);
            });
          } else if (v != null) {
            const ap = wpar('carrier_type');
            if (ap) ap.value = _carrierIndex(v);
          }
          break;
        }
        case 'carrier_freq':
        case 'carrier_frequency':
          if (isAt) bind(id, 'carrier_freq');
          else if (typeof v === 'number') { const ap = wpar('carrier_freq'); if (ap) ap.value = v; }
          break;
        case 'mix':
        case 'wet':
          if (isAt) bind(id, 'mix');
          else if (typeof v === 'number') { const ap = wpar('mix'); if (ap) ap.value = v; }
          break;
        case 'unvoiced_mix':
        case 'sibilance':
          if (isAt) bind(id, 'unvoiced_mix');
          else if (typeof v === 'number') { const ap = wpar('unvoiced_mix'); if (ap) ap.value = v; }
          break;
        case 'q':
        case 'filter_q':
          if (isAt) bind(id, 'q');
          else if (typeof v === 'number') { const ap = wpar('q'); if (ap) ap.value = v; }
          break;
        default:
          break;
      }
    }

    return {
      input: modIn,
      output: worklet,
      inputs: [modIn, carIn],
      paramTargets: targets,
    };
  }

  // ── Fallback path (no worklet) ────────────────────────────────────────────
  const fb = _buildFallback(ctx, params);

  // Live updates for the fallback build
  for (const [k, v] of Object.entries(params)) {
    const isAt = (typeof v === 'string' && v.startsWith('@'));
    if (!isAt) continue;
    const id = v.slice(1);

    switch (k) {
      case 'mix':
      case 'wet':
        targets[id] = {
          paramDef: paramDefs[id],
          customSetter: (val) => {
            const m = Math.max(0, Math.min(1, val));
            fb.fallback.wetG.gain.value = m;
            fb.fallback.dryG.gain.value = 1 - m;
          },
        };
        break;
      case 'carrier_freq':
      case 'carrier_frequency':
        targets[id] = {
          paramDef: paramDefs[id],
          customSetter: (val) => {
            if (fb.fallback.internalOsc) {
              fb.fallback.internalOsc.frequency.setTargetAtTime(val, ctx.currentTime, 0.02);
            }
          },
        };
        break;
      case 'formant_shift':
      case 'formant':
        targets[id] = {
          paramDef: paramDefs[id],
          customSetter: (val) => {
            const shift = Math.pow(2, val / 12);
            const synBands = fb.fallback.synBands;
            const n = synBands.length;
            const anaCenters = _logBandCenters(n, 100, 8000);
            for (let i = 0; i < n; i++) {
              synBands[i].frequency.setTargetAtTime(anaCenters[i] * shift, ctx.currentTime, 0.02);
            }
          },
        };
        break;
      // attack / release / unvoiced_mix / q have no clean fallback hook;
      // bind no-ops so the engine doesn't error on missing target.
      case 'attack_ms':
      case 'attack':
      case 'release_ms':
      case 'release':
      case 'unvoiced_mix':
      case 'sibilance':
      case 'q':
      case 'filter_q':
      case 'bands':
      case 'carrier_type':
        targets[id] = { paramDef: paramDefs[id], customSetter: () => {} };
        break;
      default:
        break;
    }
  }

  return {
    input: fb.input,
    output: fb.output,
    inputs: fb.inputs,
    paramTargets: targets,
  };
}

// ── Default export: NODE_BUILDERS map ─────────────────────────────────────
const R13_BUILDERS = {
  vocoder: buildVocoder,
};

export default R13_BUILDERS;
