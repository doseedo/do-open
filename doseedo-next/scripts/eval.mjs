#!/usr/bin/env node
/**
 * Evaluation harness — runs the vision + DSL pipeline against every
 * image in scripts/evals/references/ and compares output against the
 * ground-truth expectations in scripts/evals/truth.js.
 *
 * Usage:
 *   node scripts/eval.mjs                 # run all
 *   node scripts/eval.mjs wavetable-synth # run one
 *   node scripts/eval.mjs --skip-vision   # use cached DSLs from outputs/*.dsl.json
 *
 * Artifacts written to scripts/evals/outputs/<name>/:
 *   dsl.json        — the generated DSL
 *   render.html     — Babel-standalone harness
 *   render.png      — headless Chromium screenshot
 *   vision.json     — raw Moondream answers + latencies
 *   score.json      — per-criterion score + gap list
 *
 * After a run, summary.md is rewritten at scripts/evals/summary.md with a
 * diffable table of results.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..');
const evalRoot = path.join(here, 'evals');
const refDir = path.join(evalRoot, 'references');
const outDir = path.join(evalRoot, 'outputs');
const harnessScript = path.join(here, 'build-render-harness.mjs');

const args = process.argv.slice(2);
const skipVision = args.includes('--skip-vision');
const onlyNames = args.filter((a) => !a.startsWith('--'));

// ────────────────────────────────────────────────────────────────
// Env + imports
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
const TEXT_ORIGIN = 'https://arlo--doseedo-chatbot-qwenchatbot-serve.modal.run';
const VISION_ORIGIN = 'https://arlo--doseedo-chatbot-qwenchatbot-vision.modal.run';

const dslDir = path.join(repoRoot, 'src/components/Plugins/PluginCreator/refRenderer');
const { validatePluginDSL, SCHEMA_FOR_PROMPT, BANNED_BRANDS } = await import(path.join(dslDir, 'pluginDSL.js'));
const { helixDSL } = await import(path.join(dslDir, 'goldens/helix.dsl.js'));
const { strataDSL } = await import(path.join(dslDir, 'goldens/strata.dsl.js'));
const { vhsDSL } = await import(path.join(dslDir, 'goldens/vhs.dsl.js'));
const { truth } = await import(path.join(evalRoot, 'truth.js'));

// ────────────────────────────────────────────────────────────────
// Vision & DSL calls (copy of test-vision-pipeline.mjs, kept local)

const visionQuestions = [
  { k: 'palette',  q: 'Describe the dominant colors of the interface in plain English. Mention the background tone (dark/light/cream/etc.), the accent / highlight color, and any LED glow color.' },
  { k: 'controls', q: 'List the controls you can see, grouped by region. Prefix each control with KNOB:, SLIDER:, BUTTON:, LED:, DISPLAY:, METER:, or KEYBOARD:. Count as accurately as you can.' },
  { k: 'text',     q: 'List any visible text labels, brand names, product names, model numbers, or preset names. Separate with commas.' },
];

async function visionAsk(imageBase64, prompt) {
  const t0 = Date.now();
  const r = await fetch(`${VISION_ORIGIN}/v1/vision/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${KEY}` },
    body: JSON.stringify({ image_base64: imageBase64, task: 'query', prompt }),
  });
  if (!r.ok) throw new Error(`vision ${r.status}: ${(await r.text()).slice(0,200)}`);
  const j = await r.json();
  return { answer: (j.answer || '').trim(), latency: Date.now() - t0 };
}

function buildSystemPrompt() {
  return `You are an expert audio-plugin UI designer. You produce a PluginDSL JSON object. A deterministic renderer consumes the DSL.

DESIGN PRINCIPLES:
1. Treat reference as ARCHETYPE, not clone target.
2. Palette accent hue ≥90° from any referenced product's signature.
3. Compose from the row + module vocabulary in the schema.
4. Density matters. Match the control inventory seen in the reference.
5. meta.name MUST be an invented brand. Banned: ${BANNED_BRANDS.join(', ')}.

ARCHETYPE DECISION TREE — evaluate in THIS ORDER. Pick the FIRST match.
Following few-shot examples will show one implementation of each. Do NOT
conflate them — the structure you emit MUST match the archetype your case
triggers here, not whichever few-shot is most recent.

  1. WAVETABLE / SUBTRACTIVE / FM SYNTH — if observations contain ANY of:
     "OSC" (A/B/C), "WT POS", "UNISON", "WAVETABLE", "PITCH BEND",
     "MATRIX", "GLOBAL" as tabs, "CUTOFF" and "RES" together, or a
     visible piano KEYBOARD
     → chassis="plugin-window", rows=[module-strip with wavetable-osc
       + filter + noise modules, mod-matrix with macros/env/lfo/velocity,
       keyboard-strip]. productType contains "synth" or "wavetable".
     → DO NOT use character-row. DO NOT use character-module.

  2. DIGITAL REVERB (RACK) — if a 3-digit model number (224, 480, 960,
     1176, 2254, LA-2A, etc) appears AND text includes PROGRAM, DECAY,
     PRE-DELAY, TAIL, CROSSOVER, REVERB TIME, or HEADROOM
     → chassis="rack-hardware", rows=[led-display, button-row (program
       style), button-row (param-led style), slider-bank]. productType
       contains "reverb".
     → DO NOT use module-strip. DO NOT use character-row.

  3. GRANULAR ENGINE — if text contains ANY of: GRAIN, GRAIN CONTROLS,
     STRETCH, DENSITY, GRANULATION, GRANULAR, XY CONTROL, CLOUD, REVERSE
     (in granular context), SCAN (granular context)
     → chassis="plugin-window", rows=[module-strip mixing one wavetable-osc
       (display.variant="granular-sample") + one xy-pad module (background
       "constellation") + 1-2 mod-lane modules (mode "step"|"curve") +
       macros]. productType contains "granular". 10-25 knobs. No slider-bank.
     → DO NOT use character-row. DO NOT use keyboard-strip.

  4. LO-FI / TAPE / VINYL / CHARACTER COLOR PROCESSOR — only if:
       (a) text does NOT contain any of: OSC, WT POS, PROGRAM, REVERB,
           GRANULAR, CUTOFF, WAVETABLE; AND
       (b) text DOES contain 2 or more of: TAPE, VINYL, WOBBLE, WOW,
           FLUTTER, CRUSH, WEAR, BITS, MAGNETIC, DROPOUTS, FLUX, MAGNITUDE
     → chassis="plugin-window", rows=[module-strip with 6 character-module
       columns (unique tints at hue ~30/60/120/175/230/270), character-row
       below with 6 hero knobs matching the columns 1:1, optional eq-strip
       footer]. productType contains "lofi", "lo-fi", "color", "tape", or
       "character".
     → DO NOT use wavetable-osc, filter, envelope, lfo, or keyboard-strip.

  5. TAPE SATURATOR / SINGLE-FUNCTION FX — DRIVE/BIAS/IPS/SATURATION with
     only 1-2 modules' worth of controls
     → module-strip with 1-2 knob-bank modules + eq-strip footer.

  6. EQ — CUTOFF/Q/FREQ/SHELF/PEAK + big curve display, no OSC
     → module-strip with knob-bank per band.

  7. COMPRESSOR — THRESHOLD/RATIO/ATTACK/RELEASE + GR meter
     → module-strip with one knob-bank + VU-meter.

GUARDRAIL: the character-row + character-module pattern is ONLY for
archetype 4. The xy-pad module is ONLY for archetype 3. Cross-contamination
produces garbage output.

${SCHEMA_FOR_PROMPT}

OUTPUT: ONE JSON object. No prose, no fences. Start {, end }.`;
}

// Ordering matters on Qwen3-14B: later examples have stronger recency
// influence, so we put the LEAST-common archetype (lo-fi character) FIRST
// and the most common ones (synth, reverb) LAST. Otherwise Qwen will
// pattern-match everything onto the newest exemplar.
const fewShots = [
  {
    user: 'Design a lo-fi / tape / vinyl character processor — 6-column multi-stage color plugin. Each column is a different character module (TAPE, WARP, TUBE, BITS, ROOM, TAPE/magnetic) with its own tint and small knob cluster. Below the module-strip, a hero "character-row" with 6 big knobs — one per module — as the main "amount" controls. Footer eq-strip with IN GAIN, EQ CUT band, OUT WIDTH/GAIN. Dark charcoal chassis, ice-blue accent. Use this ONLY for lo-fi/tape/vinyl/color archetypes — NEVER for synths or reverbs.',
    assistant: JSON.stringify(vhsDSL),
  },
  {
    user: 'Design a digital reverb in classic rack-hardware. 2-digit 7-seg readout, stereo LED meter, 8 program presets, 8 parameter buttons, 6 slider bank. Cream faceplate, red LED.',
    assistant: JSON.stringify(strataDSL),
  },
  {
    user: 'Design a wavetable synthesizer. Three oscillators (two wavetable, one granular), a noise source, two filters, macros, AMP envelope, LFO, velocity curve, keyboard. Dark chassis, amber accent.',
    assistant: JSON.stringify(helixDSL),
  },
];

function buildUserMsg(answers) {
  return `REFERENCE IMAGE — RAW OBSERVATIONS (from Moondream 2):

  Palette:
    ${answers.palette}

  Visible controls:
    ${answers.controls}

  Visible text / labels / product names:
    ${answers.text}

Apply the archetype decision tree from the system prompt. Pick the FIRST
matching archetype — do NOT mix structures from different archetypes.
IP safety: if visible text names a real product, invent an original brand
and pick an accent hue ≥90° away from the referenced product.`;
}

async function dslCall(answers) {
  const messages = [
    { role: 'system', content: buildSystemPrompt() },
    ...fewShots.flatMap(({ user, assistant }) => [
      { role: 'user', content: user },
      { role: 'assistant', content: assistant },
    ]),
    { role: 'user', content: buildUserMsg(answers) },
  ];
  const t0 = Date.now();
  const r = await fetch(`${TEXT_ORIGIN}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${KEY}` },
    body: JSON.stringify({
      model: 'qwen3-14b',
      messages,
      temperature: 0.15,
      max_tokens: 8192,
      stream: false,
      chat_template_kwargs: { enable_thinking: false },
      response_format: { type: 'json_object' },
    }),
  });
  const latency = Date.now() - t0;
  if (!r.ok) throw new Error(`chat ${r.status}: ${(await r.text()).slice(0,200)}`);
  const j = await r.json();
  const content = j?.choices?.[0]?.message?.content || '';
  return { content, usage: j.usage, latency };
}

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
    else if (c === '}') { depth--; if (depth === 0) { try { return JSON.parse(candidate.slice(i, k+1)); } catch { return null; } } }
  }
  return null;
}

// ────────────────────────────────────────────────────────────────
// Scoring

function parseOklch(s) {
  if (!s) return null;
  const m = /^oklch\(\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)/.exec(s);
  if (!m) return null;
  return { L: +m[1], C: +m[2], H: +m[3] };
}

function hueDistance(a, b) {
  const d = Math.abs(a - b) % 360;
  return Math.min(d, 360 - d);
}

function countControls(dsl) {
  let knobs = 0, sliders = 0, buttons = 0;
  const rowKinds = new Set();
  const moduleKinds = new Set();
  for (const row of dsl?.rows || []) {
    rowKinds.add(row.kind);
    for (const m of row.modules || []) {
      moduleKinds.add(m.kind);
      knobs += (m.knobs || []).length;
    }
    for (const p of row.panels || []) {
      moduleKinds.add(p.kind);
      knobs += (p.knobs || []).length;
    }
    sliders += (row.sliders || []).length;
    buttons += (row.buttons || []).length;
  }
  return { knobs, sliders, buttons, rowKinds: [...rowKinds], moduleKinds: [...moduleKinds] };
}

function score(dsl, truth) {
  const counts = countControls(dsl);
  const pal = parseOklch(dsl.palette?.accent);
  const criteria = [];

  // Archetype — primary + optional alternates
  const productType = (dsl.meta?.productType || '').toLowerCase();
  const candidates = [truth.archetype, ...(truth.archetypeAlt || [])].map((s) => s.toLowerCase());
  const archetypeOk = candidates.some((k) => productType.includes(k));
  criteria.push({
    name: 'archetype',
    pass: archetypeOk,
    detail: `got "${dsl.meta?.productType}", expected contains one of [${candidates.join(', ')}]`,
  });

  // Chassis
  const chassisOk = dsl.meta?.chassis === truth.chassis;
  criteria.push({ name: 'chassis', pass: chassisOk, detail: `got "${dsl.meta?.chassis}", expected "${truth.chassis}"` });

  // Canvas — ±20%
  const [tw, th] = truth.canvas;
  const [gw, gh] = dsl.meta?.canvas || [0,0];
  const canvasOk = gw >= tw * 0.8 && gw <= tw * 1.2 && gh >= th * 0.8 && gh <= th * 1.2;
  criteria.push({ name: 'canvas', pass: canvasOk, detail: `got ${gw}x${gh}, expected ~${tw}x${th}` });

  // Row kinds — all expected must appear
  const missing = truth.expectedRowKinds.filter((k) => !counts.rowKinds.includes(k));
  criteria.push({
    name: 'row-kinds',
    pass: missing.length === 0,
    detail: missing.length === 0
      ? `got [${counts.rowKinds.join(', ')}]`
      : `missing [${missing.join(', ')}]; got [${counts.rowKinds.join(', ')}]`,
  });

  // Knob count
  const knobsOk = counts.knobs >= truth.minKnobs && counts.knobs <= truth.maxKnobs;
  criteria.push({ name: 'knobs', pass: knobsOk, detail: `got ${counts.knobs}, expected ${truth.minKnobs}..${truth.maxKnobs}` });

  // Slider count
  const slidersOk = counts.sliders >= truth.minSliders && counts.sliders <= truth.maxSliders;
  criteria.push({ name: 'sliders', pass: slidersOk, detail: `got ${counts.sliders}, expected ${truth.minSliders}..${truth.maxSliders}` });

  // Module kinds — at least one of expected
  if (truth.moduleKinds.length > 0) {
    const hit = truth.moduleKinds.some((k) => counts.moduleKinds.includes(k));
    criteria.push({
      name: 'module-kinds',
      pass: hit,
      detail: hit ? `got [${counts.moduleKinds.join(', ')}]` : `expected any of [${truth.moduleKinds.join(', ')}], got [${counts.moduleKinds.join(', ')}]`,
    });
  }

  // Accent hue
  if (pal) {
    const hueD = hueDistance(pal.H, truth.accentHue);
    const hueOk = hueD <= (truth.accentHueTolerance ?? 30);
    criteria.push({ name: 'accent-hue', pass: hueOk, detail: `got H=${pal.H.toFixed(0)}, distance=${hueD.toFixed(0)}° (tolerance ${truth.accentHueTolerance ?? 30}°)` });
  }

  const passed = criteria.filter((c) => c.pass).length;
  return { criteria, passed, total: criteria.length, counts };
}

// ────────────────────────────────────────────────────────────────
// Runner

async function runOne(name) {
  const imgPath = path.join(refDir, `${name}.png`);
  if (!fs.existsSync(imgPath)) throw new Error(`missing image: ${imgPath}`);
  const t = truth[name];
  if (!t) throw new Error(`no truth entry for ${name}`);

  const caseDir = path.join(outDir, name);
  fs.mkdirSync(caseDir, { recursive: true });

  let visionResult;
  let dslObj;
  let dslLatency = null;
  let dslTokens = null;

  const cachedDsl = path.join(caseDir, 'dsl.json');
  const cachedVision = path.join(caseDir, 'vision.json');

  if (skipVision && fs.existsSync(cachedDsl) && fs.existsSync(cachedVision)) {
    dslObj = JSON.parse(fs.readFileSync(cachedDsl, 'utf8'));
    visionResult = JSON.parse(fs.readFileSync(cachedVision, 'utf8'));
    process.stdout.write(`[${name}] cached\n`);
  } else {
    process.stdout.write(`[${name}] vision…\n`);
    const imgB64 = fs.readFileSync(imgPath).toString('base64');
    const answers = {};
    const latencies = {};
    for (const { k, q } of visionQuestions) {
      const r = await visionAsk(imgB64, q);
      answers[k] = r.answer;
      latencies[k] = r.latency;
    }
    visionResult = { answers, latencies, totalMs: Object.values(latencies).reduce((a,b)=>a+b,0) };
    fs.writeFileSync(cachedVision, JSON.stringify(visionResult, null, 2));

    process.stdout.write(`[${name}] DSL…\n`);
    const { content, usage, latency } = await dslCall(answers);
    dslLatency = latency;
    dslTokens = usage;
    const parsed = extractJSON(content);
    if (!parsed) {
      fs.writeFileSync(path.join(caseDir, 'dsl.raw.txt'), content);
      throw new Error(`DSL parse failed for ${name}; raw in dsl.raw.txt`);
    }
    dslObj = parsed;
    fs.writeFileSync(cachedDsl, JSON.stringify(dslObj, null, 2));
  }

  // Validate
  const validation = validatePluginDSL(dslObj);

  // Score
  const sc = validation.ok ? score(dslObj, t) : { criteria: [{ name: 'validation', pass: false, detail: validation.errors.join('; ') }], passed: 0, total: 1 };

  // Render
  const htmlPath = path.join(caseDir, 'render.html');
  const pngPath  = path.join(caseDir, 'render.png');
  try {
    const { spawnSync } = await import('node:child_process');
    const res = spawnSync('node', [harnessScript, cachedDsl, htmlPath], { encoding: 'utf8' });
    if (res.status !== 0) {
      process.stderr.write(`[${name}] harness build failed:\n${res.stderr}\n`);
    } else {
      const { chromium } = await import('playwright');
      const browser = await chromium.launch({ headless: true });
      const [w, h] = dslObj.meta?.canvas || [1280, 800];
      const ctx = await browser.newContext({
        viewport: { width: w + 40, height: h + 40 },
        deviceScaleFactor: 2,
      });
      const page = await ctx.newPage();
      await page.goto(pathToFileURL(htmlPath).href, { waitUntil: 'networkidle' });
      await page.waitForFunction(() => window.__renderReady === true, { timeout: 15000 }).catch(() => {});
      await page.waitForTimeout(1000);
      const shell = await page.$('.rr-shell');
      if (shell) await shell.screenshot({ path: pngPath });
      else await page.screenshot({ path: pngPath, fullPage: true });
      await ctx.close();
      await browser.close();
    }
  } catch (e) {
    process.stderr.write(`[${name}] render failed: ${e.message}\n`);
  }

  const record = {
    name,
    validation,
    scorecard: sc,
    meta: dslObj.meta,
    paletteAccent: dslObj.palette?.accent,
    counts: sc.counts,
    latency: {
      visionMs: visionResult.totalMs,
      dslMs: dslLatency,
      usage: dslTokens,
    },
    visionAnswers: visionResult.answers,
    truthGaps: t.gaps || [],
  };
  fs.writeFileSync(path.join(caseDir, 'score.json'), JSON.stringify(record, null, 2));
  return record;
}

// ────────────────────────────────────────────────────────────────
// Entry

const names = onlyNames.length > 0
  ? onlyNames
  : Object.keys(truth).filter((n) => fs.existsSync(path.join(refDir, `${n}.png`)));

console.log('═'.repeat(72));
console.log('REF-RENDERER EVALUATION HARNESS');
console.log('═'.repeat(72));
console.log('cases:', names.join(', '));
console.log('mode :', skipVision ? 'cached (no backend calls)' : 'full pipeline');
console.log('');

const records = [];
for (const name of names) {
  try {
    const r = await runOne(name);
    const pct = Math.round((r.scorecard.passed / Math.max(1, r.scorecard.total)) * 100);
    console.log(`  ${name.padEnd(22)} ${pct.toString().padStart(3)}%  (${r.scorecard.passed}/${r.scorecard.total})`);
    records.push(r);
  } catch (e) {
    console.error(`  ${name.padEnd(22)} FAIL: ${e.message}`);
    records.push({ name, error: e.message });
  }
}

// Summary
const summaryPath = path.join(evalRoot, 'summary.md');
const rows = [];
rows.push(`# Ref-renderer evaluation — ${new Date().toISOString()}`);
rows.push('');
rows.push('| case | score | archetype | chassis | canvas | rows | knobs | sliders | accent |');
rows.push('|------|-------|-----------|---------|--------|------|-------|---------|--------|');
for (const r of records) {
  if (r.error) { rows.push(`| ${r.name} | ERROR | ${r.error} | | | | | | |`); continue; }
  const sc = r.scorecard;
  const pct = Math.round((sc.passed / Math.max(1, sc.total)) * 100);
  rows.push(
    `| ${r.name}` +
    ` | **${pct}%** (${sc.passed}/${sc.total})` +
    ` | ${r.meta?.productType || '—'}` +
    ` | ${r.meta?.chassis || '—'}` +
    ` | ${(r.meta?.canvas || []).join('×')}` +
    ` | ${r.counts?.rowKinds?.join(', ') || '—'}` +
    ` | ${r.counts?.knobs ?? '—'}` +
    ` | ${r.counts?.sliders ?? '—'}` +
    ` | \`${r.paletteAccent || '—'}\` |`
  );
}
rows.push('');
rows.push('## Per-case detail');
rows.push('');
for (const r of records) {
  if (r.error) continue;
  rows.push(`### ${r.name}`);
  rows.push('');
  for (const c of r.scorecard.criteria) {
    rows.push(`- ${c.pass ? '✅' : '❌'} **${c.name}** — ${c.detail}`);
  }
  if (r.truthGaps && r.truthGaps.length > 0) {
    rows.push('');
    rows.push('**Known vocabulary gaps** (from truth.js, these are not scored):');
    for (const g of r.truthGaps) rows.push(`- ${g}`);
  }
  rows.push('');
  rows.push(`_latency: vision ${Math.round(r.latency.visionMs/100)/10}s · dsl ${r.latency.dslMs?Math.round(r.latency.dslMs/100)/10:'—'}s · prompt ${r.latency.usage?.prompt_tokens || '—'} / completion ${r.latency.usage?.completion_tokens || '—'} tokens_`);
  rows.push('');
}
fs.writeFileSync(summaryPath, rows.join('\n'));
console.log('');
console.log('summary →', summaryPath);
