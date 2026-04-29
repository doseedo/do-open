#!/usr/bin/env node
/**
 * End-to-end test of the ref-renderer Qwen pipeline.
 *
 *   - Reads VLLM_API_KEY from .env.local
 *   - Reuses buildSystemPrompt + buildFewShots from the real module
 *   - Hits https://arlo--doseedo-chatbot-qwenchatbot-serve.modal.run
 *     directly with Bearer auth (skips the Next.js /api/chat proxy so
 *     we don't need the dev server running)
 *   - Validates the returned DSL with the same validator the renderer
 *     uses, then prints a scorecard
 *
 * Run from doseedo-next/:
 *   node scripts/test-ref-renderer.mjs
 *   node scripts/test-ref-renderer.mjs "your custom brief here"
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..');

// ────────────────────────────────────────────────────────────────
// Load VLLM_API_KEY from .env.local

function loadEnvLocal() {
  const p = path.join(repoRoot, '.env.local');
  if (!fs.existsSync(p)) throw new Error('.env.local not found');
  const text = fs.readFileSync(p, 'utf8');
  const env = {};
  for (const line of text.split('\n')) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)$/);
    if (!m) continue;
    let v = m[2].trim();
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
      v = v.slice(1, -1);
    }
    env[m[1]] = v;
  }
  return env;
}

const env = loadEnvLocal();
const VLLM_API_KEY = env.VLLM_API_KEY;
if (!VLLM_API_KEY) throw new Error('VLLM_API_KEY not set in .env.local');

const CHATBOT_ORIGIN =
  env.CHATBOT_ORIGIN ||
  'https://arlo--doseedo-chatbot-qwenchatbot-serve.modal.run';

// ────────────────────────────────────────────────────────────────
// Import the real prompt builders + validator

const refDir = path.join(
  repoRoot,
  'src/components/Plugins/PluginCreator/refRenderer',
);
// Node ESM requires .js extensions; the app's webpack resolves without
// them. So we import pluginDSL + goldens directly (they have no deep
// deps), and inline the prompt builders to avoid pulling in qwenChat
// (which is not needed here — we call Modal directly).
const { validatePluginDSL, SCHEMA_FOR_PROMPT, BANNED_BRANDS } =
  await import(path.join(refDir, 'pluginDSL.js'));
const { helixDSL }  = await import(path.join(refDir, 'goldens/helix.dsl.js'));
const { strataDSL } = await import(path.join(refDir, 'goldens/strata.dsl.js'));

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

function buildFewShots() {
  return [
    {
      user:
        'Design a wavetable synthesizer. Three oscillators (two wavetable, ' +
        'one granular), a noise source, two filters, macros, AMP envelope, ' +
        'LFO, velocity curve, keyboard. Dark chassis, amber accent, warm ' +
        'cream-colored mono/brand fonts. Hero preset: "Ember Drift".',
      assistant: JSON.stringify(helixDSL),
    },
    {
      user:
        'Design a digital reverb in a classic rack-hardware chassis. 2-digit ' +
        '7-segment readout, stereo LED meter, 8 program presets, 8 ' +
        'parameter buttons, 6 slider bank (bass/mid/crossover/treble-decay/' +
        'depth/pre-delay) with a REVERB TIME bracket. Cream faceplate, red ' +
        'LED accent, slab-serif wordmark.',
      assistant: JSON.stringify(strataDSL),
    },
  ];
}

function extractJSON(text) {
  if (!text) return null;
  const stripped = text.replace(/<think>[\s\S]*?<\/think>\s*/g, '').trim();
  const fenced = stripped.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidate = fenced ? fenced[1] : stripped;
  const firstObj = candidate.indexOf('{');
  const firstArr = candidate.indexOf('[');
  const starts = [firstObj, firstArr].filter((i) => i >= 0);
  if (starts.length === 0) return null;
  const start = Math.min(...starts);
  const openCh = candidate[start];
  const closeCh = openCh === '{' ? '}' : ']';
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < candidate.length; i++) {
    const ch = candidate[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') { inStr = true; continue; }
    if (ch === openCh) depth++;
    else if (ch === closeCh) {
      depth--;
      if (depth === 0) {
        try { return JSON.parse(candidate.slice(start, i + 1)); } catch { return null; }
      }
    }
  }
  return null;
}

// ────────────────────────────────────────────────────────────────
// One direct call to Modal (non-streaming), with the same payload the
// Next.js proxy would have built.

async function qwenCall({ messages, temperature = 0.15, maxTokens = 8192, timeoutMs = 300000 }) {
  const body = {
    model: 'qwen3-14b',
    messages,
    temperature,
    max_tokens: maxTokens,
    stream: false,
    chat_template_kwargs: { enable_thinking: false },
    response_format: { type: 'json_object' },
  };
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  const t0 = Date.now();
  let res;
  try {
    res = await fetch(`${CHATBOT_ORIGIN}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${VLLM_API_KEY}`,
      },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
  } finally {
    clearTimeout(timer);
  }
  const latency = Date.now() - t0;
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Modal ${res.status}: ${txt.slice(0, 300)} (${latency}ms)`);
  }
  const raw = await res.json();
  const content = raw?.choices?.[0]?.message?.content || '';
  const usage = raw?.usage;
  return { content, usage, latency };
}

function stripThink(s) {
  return (s || '').replace(/<think>[\s\S]*?<\/think>\s*/g, '');
}

// ────────────────────────────────────────────────────────────────
// Scorecard

function scorecard(dsl) {
  const lines = [];
  const push = (k, v) => lines.push(`    ${k.padEnd(28)} ${v}`);

  push('meta.name', dsl?.meta?.name);
  push('meta.productType', dsl?.meta?.productType);
  push('meta.canvas', JSON.stringify(dsl?.meta?.canvas));
  push('meta.chassis', dsl?.meta?.chassis);
  push('palette.accent', dsl?.palette?.accent);
  push('palette.bg[0..n]', `[${(dsl?.palette?.bg || []).length}]`);
  push('type.brand / ui / mono',
    `${dsl?.type?.brand} / ${dsl?.type?.ui} / ${dsl?.type?.mono}`);
  push('header.tabs', JSON.stringify(dsl?.header?.tabs));
  push('rows.length', (dsl?.rows || []).length);

  const rowKindCount = {};
  const modKindCount = {};
  let knobTotal = 0;
  for (const row of dsl?.rows || []) {
    rowKindCount[row.kind] = (rowKindCount[row.kind] || 0) + 1;
    for (const m of row.modules || []) {
      modKindCount[m.kind] = (modKindCount[m.kind] || 0) + 1;
      knobTotal += (m.knobs || []).length;
    }
    for (const p of row.panels || []) {
      modKindCount[p.kind] = (modKindCount[p.kind] || 0) + 1;
      knobTotal += (p.knobs || []).length;
    }
    for (const _s of row.sliders || []) knobTotal += 0; // sliders tracked separately
  }
  push('rowKinds', JSON.stringify(rowKindCount));
  push('moduleKinds', JSON.stringify(modKindCount));
  push('total knobs', knobTotal);
  return lines.join('\n');
}

// ────────────────────────────────────────────────────────────────
// Run

const userBrief = process.argv[2] || `
A vintage tape saturation stereo bus processor. Single large DRIVE knob is
the hero. Smaller INPUT and OUTPUT trim knobs flanking it. BIAS, TONE,
LOW-CUT knobs. A tape-speed rocker (7.5 / 15 IPS) and tape-type switch
(LP / SM / CL). Stereo LINK button. VU meters for L/R level. Overall feel:
warm, cream faceplate, orange-amber VU illumination, slab-serif wordmark.
`.trim();

const system = buildSystemPrompt();
const fewShots = buildFewShots();
const messages = [
  { role: 'system', content: system },
  ...fewShots.flatMap(({ user: u, assistant: a }) => [
    { role: 'user', content: u },
    { role: 'assistant', content: a },
  ]),
  { role: 'user', content: userBrief },
];

console.log('─'.repeat(64));
console.log('REF-RENDERER LIVE TEST');
console.log('─'.repeat(64));
console.log('endpoint      :', CHATBOT_ORIGIN);
console.log('model         : qwen3-14b');
console.log('few-shots     :', fewShots.length, '(helix, strata)');
console.log('system tokens ~', system.length, 'chars');
const totalChars = messages.reduce((n, m) => n + m.content.length, 0);
console.log('total input   ~', totalChars, 'chars (≈', Math.round(totalChars / 3.5), 'tokens)');
console.log('brief         :', userBrief.slice(0, 100).replace(/\n/g, ' ') + (userBrief.length > 100 ? '…' : ''));
console.log('');

let result;
try {
  result = await qwenCall({ messages, temperature: 0.15, maxTokens: 8192 });
} catch (e) {
  console.error('❌ Modal call failed:', e.message);
  process.exit(1);
}

const clean = stripThink(result.content);
console.log('latency       :', result.latency, 'ms');
if (result.usage) {
  console.log('usage         : prompt=' + result.usage.prompt_tokens + ' completion=' + result.usage.completion_tokens + ' total=' + result.usage.total_tokens);
}
console.log('raw chars     :', result.content.length, '(clean:', clean.length + ')');

const parsed = extractJSON(clean);
if (!parsed) {
  console.log('');
  console.log('❌ JSON parse failed. First 500 chars of response:');
  console.log(clean.slice(0, 500));
  process.exit(2);
}

const v = validatePluginDSL(parsed);
console.log('');
console.log(v.ok ? '✅ DSL validates' : '⚠️  DSL has schema issues:');
if (!v.ok) {
  for (const e of v.errors) console.log('    -', e);
}
console.log('');
console.log('  scorecard:');
console.log(scorecard(parsed));

// Dump to disk for inspection
const outPath = path.join(repoRoot, 'scripts/.last-ref-test.json');
fs.writeFileSync(outPath, JSON.stringify(parsed, null, 2));
console.log('');
console.log('  full DSL  →', outPath);
