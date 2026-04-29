#!/usr/bin/env node
/**
 * End-to-end vision + DSL pipeline test.
 *
 *   node scripts/test-vision-pipeline.mjs <path-to-image.png> [brief]
 *
 * Steps:
 *   1. Read the image, base64-encode.
 *   2. POST /v1/vision/analyze for each of our 5 canned vision queries
 *      (archetype, chassis, palette, controls, text) — stitches them
 *      into the same analysis block analyzePluginReferenceImage builds.
 *   3. Call /v1/chat/completions with helix+strata few-shots, the text
 *      brief, and the vision-derived analysis block.
 *   4. Validate the returned DSL and print a scorecard.
 *
 * Skips both Next proxies — hits Modal directly with Bearer auth from
 * .env.local, so no dev server required.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..');

function loadEnv() {
  const text = fs.readFileSync(path.join(repoRoot, '.env.local'), 'utf8');
  const env = {};
  for (const line of text.split('\n')) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)$/);
    if (!m) continue;
    let v = m[2].trim();
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) v = v.slice(1, -1);
    env[m[1]] = v;
  }
  return env;
}
const env = loadEnv();
const KEY = env.VLLM_API_KEY;
const TEXT_ORIGIN   = 'https://arlo--doseedo-chatbot-qwenchatbot-serve.modal.run';
const VISION_ORIGIN = 'https://arlo--doseedo-chatbot-qwenchatbot-vision.modal.run';

// ────────────────────────────────────────────────────────────────
// Import real DSL schema + goldens + extract utility

const refDir = path.join(repoRoot, 'src/components/Plugins/PluginCreator/refRenderer');
const { validatePluginDSL, SCHEMA_FOR_PROMPT, BANNED_BRANDS } = await import(path.join(refDir, 'pluginDSL.js'));
const { helixDSL }  = await import(path.join(refDir, 'goldens/helix.dsl.js'));
const { strataDSL } = await import(path.join(refDir, 'goldens/strata.dsl.js'));

// ────────────────────────────────────────────────────────────────
// Args

const imgPath = process.argv[2];
if (!imgPath || !fs.existsSync(imgPath)) {
  console.error('usage: test-vision-pipeline.mjs <image-path> [brief]');
  process.exit(1);
}
const userBrief = process.argv[3] || '';
const imgBytes = fs.readFileSync(imgPath);
const imgBase64 = imgBytes.toString('base64');
console.log('image         :', imgPath, `(${(imgBytes.length / 1024).toFixed(1)} KB)`);

// ────────────────────────────────────────────────────────────────
// Vision queries

// Mirror src/services/qwenChat.js analyzePluginReferenceImage — only
// descriptive questions; archetype + chassis are inferred by Qwen from
// the text/controls output instead of asked of Moondream directly.
const visionQuestions = [
  { k: 'palette',  q: 'Describe the dominant colors of the interface in plain English. Mention the background tone (dark/light/cream/etc.), the accent / highlight color, and any LED glow color.' },
  { k: 'controls', q: 'List the controls you can see, grouped by region. Prefix each control with KNOB:, SLIDER:, BUTTON:, LED:, DISPLAY:, METER:, or KEYBOARD:. Count as accurately as you can.' },
  { k: 'text',     q: 'List any visible text labels, brand names, product names, model numbers, or preset names. Separate with commas.' },
];

async function visionAsk(prompt) {
  const body = { image_base64: imgBase64, task: 'query', prompt };
  const t0 = Date.now();
  const r = await fetch(`${VISION_ORIGIN}/v1/vision/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${KEY}` },
    body: JSON.stringify(body),
  });
  const latency = Date.now() - t0;
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`vision ${r.status}: ${txt.slice(0, 200)}`);
  }
  const j = await r.json();
  return { answer: (j.answer || '').trim(), latency };
}

console.log('');
console.log('── vision: 5 questions ──');
const answers = {};
const latencies = [];
for (const { k, q } of visionQuestions) {
  const { answer, latency } = await visionAsk(q);
  answers[k] = answer;
  latencies.push(latency);
  console.log(`  ${k.padEnd(10)} ${latency.toString().padStart(5)}ms  ${answer.slice(0, 100).replace(/\n/g, ' ')}${answer.length > 100 ? '…' : ''}`);
}
const visionSum = latencies.reduce((a, b) => a + b, 0);
console.log(`  → vision total: ${visionSum} ms (${(visionSum / 1000).toFixed(1)}s)`);

const analysisText =
  `REFERENCE IMAGE — RAW OBSERVATIONS (from Moondream 2, a small VLM).\n` +
  `Treat these as descriptions, NOT conclusions. You (the DSL model) must\n` +
  `INFER the archetype and chassis yourself from the visible labels and\n` +
  `control inventory below.\n\n` +
  `  Palette:\n    ${answers.palette}\n\n` +
  `  Visible controls:\n    ${answers.controls}\n\n` +
  `  Visible text / labels / product names:\n    ${answers.text}\n\n` +
  `ARCHETYPE INFERENCE CUES — use the "Visible text" and "Visible controls"\n` +
  `fields to decide:\n` +
  `  - OSC / LFO / ENV / FILTER / WAVETABLE / UNISON / GRANULAR / piano\n` +
  `    KEYBOARD → wavetable-synth (use module-strip row).\n` +
  `  - DECAY / PRE-DELAY / PROGRAM / CROSSOVER / TAIL / 7-seg DISPLAY + bank\n` +
  `    of labeled BUTTONs → digital-reverb (led-display + button-row +\n` +
  `    slider-bank).\n` +
  `  - DRIVE / BIAS / TAPE / IPS / CAL / VU METER → tape-saturator.\n` +
  `  - CUTOFF / Q / FREQUENCY / SHELF / PEAK + curve DISPLAY → EQ.\n` +
  `  - THRESHOLD / RATIO / ATTACK / RELEASE + gain-reduction METER →\n` +
  `    compressor.\n\n` +
  `CHASSIS INFERENCE CUES — infer from the archetype + text:\n` +
  `  - Software synths/effects/EQs (tabs like MAIN/MIX/FX, preset bar) →\n` +
  `    plugin-window. Default for plugin-ish UIs.\n` +
  `  - Classic rack digital hardware — 3-digit model numbers (224, 480,\n` +
  `    960, 1176, 2254, LA-2A), PROGRAM button bank, 7-seg DISPLAY +\n` +
  `    faceplate sliders → rack-hardware.\n` +
  `  - Minimal control set (≤6 knobs, 1-2 footswitches, no tabs, no preset\n` +
  `    bar) → pedal. Only when control inventory is genuinely pedal-sparse.\n\n` +
  `IP SAFETY: if "Visible text" names a real product (Serum, Lexicon, Valhalla,\n` +
  `FabFilter, Pro-Q, Waves SSL, etc.), invent an ORIGINAL brand name for\n` +
  `meta.name and pick a palette accent hue ≥90° away from that product's\n` +
  `signature color.`;

// ────────────────────────────────────────────────────────────────
// DSL call

function buildSystemPrompt() {
  return `You are an expert audio-plugin UI designer. You produce a PluginDSL JSON object describing a plugin interface. A deterministic renderer consumes the DSL; your job is to emit the DSL, not to describe design choices in prose.

DESIGN PRINCIPLES:
1. Treat the reference image as ARCHETYPE, not a specific product to clone.
2. Pick a palette perceptually far from any referenced product. Accent hue ≥90° from referenced signature hue.
3. Compose from the row + module vocabulary in the schema.
4. Density matters. A synth voice row has 4-7 modules; a reverb has 8 program buttons, 6-8 sliders.
5. Meta.name MUST be an invented brand. Banned: ${BANNED_BRANDS.join(', ')}.

${SCHEMA_FOR_PROMPT}

OUTPUT FORMAT: Respond with ONE JSON object. No prose, no markdown fences. Start with { end with }.`;
}

const userMsg = userBrief
  ? `${userBrief}\n\n${analysisText}`
  : `(no additional brief — use the reference analysis above)\n\n${analysisText}`;

const messages = [
  { role: 'system', content: buildSystemPrompt() },
  { role: 'user',   content: 'Design a wavetable synthesizer. Three oscillators (two wavetable, one granular), a noise source, two filters, macros, AMP envelope, LFO, velocity curve, keyboard. Dark chassis, amber accent. Hero preset: "Ember Drift".' },
  { role: 'assistant', content: JSON.stringify(helixDSL) },
  { role: 'user',   content: 'Design a digital reverb in a classic rack-hardware chassis. 2-digit 7-segment readout, stereo LED meter, 8 program presets, 8 parameter buttons, 6 slider bank (bass/mid/crossover/treble-decay/depth/pre-delay). Cream faceplate, red LED accent.' },
  { role: 'assistant', content: JSON.stringify(strataDSL) },
  { role: 'user',   content: userMsg },
];

const chatBody = {
  model: 'qwen3-14b',
  messages,
  temperature: 0.15,
  max_tokens: 8192,
  stream: false,
  chat_template_kwargs: { enable_thinking: false },
  response_format: { type: 'json_object' },
};

console.log('');
console.log('── DSL synthesis ──');
const t0 = Date.now();
const chatRes = await fetch(`${TEXT_ORIGIN}/v1/chat/completions`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${KEY}` },
  body: JSON.stringify(chatBody),
});
const dslLatency = Date.now() - t0;
if (!chatRes.ok) {
  console.error('chat failed:', chatRes.status, (await chatRes.text()).slice(0, 200));
  process.exit(2);
}
const chatJson = await chatRes.json();
const content = chatJson?.choices?.[0]?.message?.content || '';
console.log(`  DSL latency: ${dslLatency} ms; prompt=${chatJson.usage?.prompt_tokens} completion=${chatJson.usage?.completion_tokens}`);

function extractJSON(text) {
  const stripped = (text || '').replace(/<think>[\s\S]*?<\/think>\s*/g, '').trim();
  const fenced = stripped.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidate = fenced ? fenced[1] : stripped;
  const i = candidate.indexOf('{');
  if (i < 0) return null;
  let depth = 0, inStr = false, esc = false;
  for (let k = i; k < candidate.length; k++) {
    const c = candidate[k];
    if (inStr) { if (esc) esc = false; else if (c === '\\') esc = true; else if (c === '"') inStr = false; continue; }
    if (c === '"') { inStr = true; continue; }
    if (c === '{') depth++;
    else if (c === '}') { depth--; if (depth === 0) { try { return JSON.parse(candidate.slice(i, k + 1)); } catch { return null; } } }
  }
  return null;
}

const dsl = extractJSON(content);
if (!dsl) {
  console.error('❌ DSL parse failed. first 300 chars:', content.slice(0, 300));
  process.exit(3);
}
const v = validatePluginDSL(dsl);
console.log('');
console.log(v.ok ? '✅ DSL validates' : '⚠️ schema issues:');
if (!v.ok) v.errors.forEach((e) => console.log('   -', e));

console.log('');
console.log('── scorecard ──');
console.log(`  meta.name        ${dsl.meta?.name}`);
console.log(`  meta.productType ${dsl.meta?.productType}`);
console.log(`  meta.chassis     ${dsl.meta?.chassis}`);
console.log(`  canvas           ${JSON.stringify(dsl.meta?.canvas)}`);
console.log(`  accent           ${dsl.palette?.accent}`);
console.log(`  rows             ${dsl.rows?.length} (${(dsl.rows || []).map(r => r.kind).join(', ')})`);
let knobTotal = 0, sliderTotal = 0;
for (const row of dsl.rows || []) {
  for (const m of row.modules || []) knobTotal += (m.knobs || []).length;
  for (const p of row.panels || [])  knobTotal += (p.knobs || []).length;
  sliderTotal += (row.sliders || []).length;
}
console.log(`  knobs / sliders  ${knobTotal} / ${sliderTotal}`);

// Save output
const outName = path.basename(imgPath).replace(/\.[^.]+$/, '') + '.dsl.json';
const outPath = path.join(here, outName);
fs.writeFileSync(outPath, JSON.stringify(dsl, null, 2));
console.log('');
console.log('  full DSL →', outPath);
