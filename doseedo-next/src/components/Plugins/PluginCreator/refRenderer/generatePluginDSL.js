/**
 * generatePluginDSL(brief) — single Qwen call that produces a validated
 * PluginDSL from a user brief. Uses the Helix + Strata golden DSLs as
 * few-shot examples.
 *
 * Qwen3-14B is a smaller model than Claude, so:
 *   - Thinking is OFF (saves tokens, speeds JSON output).
 *   - Temperature is low (0.15) for structural compliance.
 *   - Few-shots are the load-bearing mechanism — the prompt is thin.
 *   - validatePluginDSL gates the output; one retry on failure.
 *
 * Cold-start note: the Modal container scales to zero. Call warmup()
 * at the start of a design session so the first generate isn't blocked
 * 50-80s on a container restore.
 */

import {
  qwenChatJSON,
  qwenHealth,
  qwenVisionHealth,
  analyzePluginReferenceImage,
} from '../../../../services/qwenChat';
import { validatePluginDSL, SCHEMA_FOR_PROMPT, BANNED_BRANDS } from './pluginDSL';
import { helixDSL } from './goldens/helix.dsl';
import { strataDSL } from './goldens/strata.dsl';
import { vhsDSL } from './goldens/vhs.dsl';

// ────────────────────────────────────────────────────────────────
// Prompt construction. Keep tight — every token counts against 28K
// (max_model_len was reduced from 32768 to 28672 to leave room for
// Moondream 2 in the GPU budget).

function buildSystemPrompt() {
  return `You are an expert audio-plugin UI designer. You produce a
PluginDSL JSON object describing a plugin interface. A deterministic
renderer consumes the DSL to generate the final pixels; your job is
to emit the DSL, not to describe design choices in prose.

DESIGN PRINCIPLES (the reason this produces good output):
1. Analyze the brief as an ARCHETYPE, not a specific product.
   "wavetable synth like Serum" → archetype = wavetable-synth; invent
   an original brand distinct from Serum.
2. Pick a palette that is perceptually far from any referenced product.
   Use oklch(L C H). Synth chassis = dark (bg L≈0.15..0.33). Rack/pedal
   faceplate = cream/gray (bg L≈0.80..0.92). Accent hue ≥ 90° from any
   referenced product's signature hue.
3. Compose from the row + module vocabulary in the schema. Each row
   kind is load-bearing — use module-strip for synth voice rows,
   mod-matrix for macros/env/lfo/velocity, led-display + button-row +
   slider-bank for rack-digital reverb/delay/compressor chassis.
4. Density matters. A synth voice row typically has 4-7 modules; each
   module 4-9 knobs. A reverb has 8 program buttons, 6-8 sliders.
5. Meta.name MUST be an invented brand. Banned: ${BANNED_BRANDS.join(', ')}.

${SCHEMA_FOR_PROMPT}

OUTPUT FORMAT:
Respond with ONE JSON object that validates against the schema. No
prose, no markdown fences, no comments. Start with { and end with }.`;
}

// Few-shot: golden DSLs. Strip them to JSON strings; Qwen sees them as
// literal example assistant replies.
const HELIX_JSON   = JSON.stringify(helixDSL);
const STRATA_JSON  = JSON.stringify(strataDSL);

const VHS_JSON = JSON.stringify(vhsDSL);

// Ordering matters on Qwen3-14B: later examples have stronger recency
// influence. Put the NARROWEST-archetype example (lo-fi character) FIRST
// and the broader synth example LAST so the latter dominates the default
// output, with lo-fi only triggered when its specific cues match.
function buildFewShots() {
  return [
    {
      user:
        'Design a lo-fi / tape / vinyl character processor — 6-column ' +
        'multi-stage color plugin. Each column is a different character ' +
        'module (TAPE, WARP, TUBE, BITS, ROOM, TAPE/magnetic) with its own ' +
        'tint and small knob cluster. Below the module-strip, a hero ' +
        'character-row with 6 big knobs — one per module — as the main ' +
        '"amount" controls. Footer eq-strip with IN GAIN, EQ CUT band, OUT ' +
        'WIDTH/GAIN. Dark charcoal chassis, ice-blue accent. Use this ONLY ' +
        'for lo-fi/tape/vinyl/color archetypes — NEVER for synths or reverbs.',
      assistant: VHS_JSON,
    },
    {
      user:
        'Design a digital reverb in a classic rack-hardware chassis. 2-digit ' +
        '7-segment readout, stereo LED meter, 8 program presets, 8 ' +
        'parameter buttons, 6 slider bank (bass/mid/crossover/treble-decay/' +
        'depth/pre-delay) with a REVERB TIME bracket. Cream faceplate, red ' +
        'LED accent, slab-serif wordmark.',
      assistant: STRATA_JSON,
    },
    {
      user:
        'Design a wavetable synthesizer. Three oscillators (two wavetable, ' +
        'one granular), a noise source, two filters, macros, AMP envelope, ' +
        'LFO, velocity curve, keyboard. Dark chassis, amber accent, warm ' +
        'cream-colored mono/brand fonts. Hero preset: "Ember Drift".',
      assistant: HELIX_JSON,
    },
  ];
}

// ────────────────────────────────────────────────────────────────

export async function generatePluginDSL({
  brief,
  referenceAnalysis,      // optional prose description of a reference image
  referenceImage,         // optional base64/dataURL — triggers Moondream analysis
  onStage,                // optional callback (stage, info) for UI progress
  temperature = 0.15,
  maxTokens = 8192,
  timeoutMs = 300000,
} = {}) {
  if (!brief || typeof brief !== 'string' || brief.trim().length === 0) {
    if (!referenceImage) {
      throw new Error('generatePluginDSL: brief or referenceImage required');
    }
  }

  const stage = (name, info) => { if (onStage) onStage(name, info); };

  // Stage 1 (optional): vision analysis of the reference image.
  let analysisBlock = referenceAnalysis || '';
  let visionAnswers = null;
  if (referenceImage) {
    stage('vision:start');
    const t0 = Date.now();
    const { text: autoAnalysis, answers } =
      await analyzePluginReferenceImage({ imageBase64: referenceImage });
    visionAnswers = answers;
    stage('vision:done', { latencyMs: Date.now() - t0, answers });
    analysisBlock = analysisBlock
      ? `${analysisBlock}\n\n${autoAnalysis}`
      : autoAnalysis;
  }

  const userMsg = analysisBlock
    ? `${brief || '(no additional brief — use the reference analysis above)'}\n\n${analysisBlock}`
    : brief;

  stage('dsl:start');
  const t0 = Date.now();
  const { data, text, attempts } = await qwenChatJSON({
    system: buildSystemPrompt(),
    user: userMsg,
    fewShots: buildFewShots(),
    temperature,
    maxTokens,
    timeoutMs,
    validate: validatePluginDSL,
  });
  stage('dsl:done', { latencyMs: Date.now() - t0, attempts });

  return { dsl: data, rawText: text, attempts, visionAnswers };
}

/**
 * Fire-and-forget warmup so the first real generate isn't blocked on
 * Modal's ~50-80s snapshot restore. Pings both the text and vision
 * endpoints so Moondream also gets a chance to finish loading.
 * Call at session start.
 */
export async function warmup() {
  await Promise.allSettled([qwenHealth(), qwenVisionHealth()]);
}

// ────────────────────────────────────────────────────────────────
// Also expose the pieces for callers who want to do streaming or
// custom prompt composition.

export { buildSystemPrompt, buildFewShots };
