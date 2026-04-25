/**
 * Qwen chat client — talks to the Modal vLLM chatbot via the Next.js
 * server proxy at /api/chat (which injects VLLM_API_KEY). Use this when
 * you need direct OpenAI-style access to the self-hosted Qwen3-14B,
 * separate from the legacy /_chat/api DAW chatbot.
 *
 * The proxy route (app/api/chat/route.ts) forwards the body verbatim to
 * /v1/chat/completions on the Modal endpoint. We pass
 *   chat_template_kwargs: { enable_thinking: false }
 * by default so Qwen3 skips its <think>…</think> prefix — critical for
 * structured-JSON calls. As a belt-and-suspenders the non-streaming
 * client also strips any <think>…</think> block that leaks through.
 */

const CHAT_ENDPOINT = '/api/chat';
const VISION_ENDPOINT = '/api/vision';
const DEFAULT_MODEL = 'qwen3-14b';

const THINK_RE = /<think>[\s\S]*?<\/think>\s*/g;
const stripThink = (text) => (text || '').replace(THINK_RE, '');

/**
 * Non-streaming chat completion.
 *
 * @param {Object} opts
 * @param {Array<{role:string,content:string}>} opts.messages
 * @param {string}  [opts.model]            — defaults to qwen3-14b
 * @param {boolean} [opts.enableThinking]   — default false. Set true for
 *                                            open-ended design reasoning;
 *                                            leave false for JSON outputs.
 * @param {number}  [opts.temperature]      — default 0.7 (0 for JSON calls)
 * @param {number}  [opts.maxTokens]        — default 4096
 * @param {Object}  [opts.responseFormat]   — pass { type: 'json_object' }
 *                                            to request JSON from vLLM.
 * @param {number}  [opts.timeoutMs]        — default 300000 (5min; cold
 *                                            starts on Modal are ~50-80s).
 * @param {Array}   [opts.stop]
 * @returns {Promise<{text:string, raw:Object}>}
 */
export async function qwenChat({
  messages,
  model = DEFAULT_MODEL,
  enableThinking = false,
  temperature = 0.7,
  maxTokens = 4096,
  responseFormat,
  timeoutMs = 300000,
  stop,
} = {}) {
  if (!Array.isArray(messages) || messages.length === 0) {
    throw new Error('qwenChat: messages[] required');
  }

  const body = {
    model,
    messages,
    temperature,
    max_tokens: maxTokens,
    stream: false,
    chat_template_kwargs: { enable_thinking: !!enableThinking },
  };
  if (responseFormat) body.response_format = responseFormat;
  if (stop) body.stop = stop;

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);

  let res;
  try {
    res = await fetch(CHAT_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    throw new Error(`qwenChat ${res.status}: ${errText.slice(0, 400)}`);
  }

  const raw = await res.json();
  const content = raw?.choices?.[0]?.message?.content || '';
  return { text: stripThink(content), raw };
}

/**
 * Streaming chat completion. Emits token-by-token via onToken, resolves
 * with the final concatenated text.
 *
 * @param {Object} opts  same as qwenChat, plus:
 * @param {(delta:string, full:string) => void}   [opts.onToken]
 * @param {(thinking:string, full:string) => void} [opts.onThinking]
 *        — called separately for <think> tokens when enableThinking is
 *        true, so a UI can render a collapsed "thinking" pane.
 * @returns {Promise<{text:string, thinking:string}>}
 */
export async function qwenChatStream({
  messages,
  model = DEFAULT_MODEL,
  enableThinking = false,
  temperature = 0.7,
  maxTokens = 4096,
  responseFormat,
  onToken,
  onThinking,
  timeoutMs = 300000,
  stop,
} = {}) {
  const body = {
    model,
    messages,
    temperature,
    max_tokens: maxTokens,
    stream: true,
    chat_template_kwargs: { enable_thinking: !!enableThinking },
  };
  if (responseFormat) body.response_format = responseFormat;
  if (stop) body.stop = stop;

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);

  let res;
  try {
    res = await fetch(CHAT_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
  } catch (e) {
    clearTimeout(timer);
    throw e;
  }

  if (!res.ok || !res.body) {
    clearTimeout(timer);
    const errText = await res.text().catch(() => '');
    throw new Error(`qwenChatStream ${res.status}: ${errText.slice(0, 400)}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buf = '';
  let fullRaw = '';   // everything including <think> blocks
  let answer = '';    // post-</think> content
  let thinking = '';  // content inside <think>…</think>
  let inThink = false;

  const pushDelta = (delta) => {
    if (!delta) return;
    fullRaw += delta;

    // Walk char-by-char to detect <think>/</think> tags across chunk boundaries.
    let i = 0;
    while (i < delta.length) {
      if (!inThink) {
        const open = delta.indexOf('<think>', i);
        if (open === -1) {
          const chunk = delta.slice(i);
          answer += chunk;
          if (onToken) onToken(chunk, answer);
          break;
        } else {
          const chunk = delta.slice(i, open);
          if (chunk) {
            answer += chunk;
            if (onToken) onToken(chunk, answer);
          }
          i = open + 7;
          inThink = true;
        }
      } else {
        const close = delta.indexOf('</think>', i);
        if (close === -1) {
          const chunk = delta.slice(i);
          thinking += chunk;
          if (onThinking) onThinking(chunk, thinking);
          break;
        } else {
          const chunk = delta.slice(i, close);
          if (chunk) {
            thinking += chunk;
            if (onThinking) onThinking(chunk, thinking);
          }
          i = close + 8;
          inThink = false;
        }
      }
    }
  };

  try {
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (!payload || payload === '[DONE]') continue;
        try {
          const evt = JSON.parse(payload);
          const delta = evt?.choices?.[0]?.delta?.content || '';
          pushDelta(delta);
        } catch {
          // ignore malformed SSE frames
        }
      }
    }
  } finally {
    clearTimeout(timer);
  }

  return { text: answer.trim(), thinking: thinking.trim(), raw: fullRaw };
}

/**
 * Ask Qwen for a JSON object and parse it. Retries once if the first
 * response fails to parse or fails validation.
 *
 * Qwen3 + vLLM supports `response_format: {"type": "json_object"}` but
 * compliance isn't Claude-grade. We ask for JSON, try to parse, and on
 * failure resend with the parse/validation error inlined.
 *
 * @param {Object} opts  qwenChat opts plus:
 * @param {string}  opts.system
 * @param {string}  opts.user
 * @param {Array}   [opts.fewShots]  — [{user, assistant}, …]
 * @param {(obj:any) => {ok:true}|{ok:false, errors:string[]}} [opts.validate]
 * @returns {Promise<{data:any, text:string, attempts:number}>}
 */
export async function qwenChatJSON({
  system,
  user,
  fewShots = [],
  validate,
  model = DEFAULT_MODEL,
  temperature = 0.2,
  maxTokens = 8192,
  timeoutMs = 300000,
} = {}) {
  const baseMessages = [
    { role: 'system', content: system },
    ...fewShots.flatMap(({ user: u, assistant: a }) => [
      { role: 'user', content: u },
      { role: 'assistant', content: a },
    ]),
    { role: 'user', content: user },
  ];

  const attempt = async (messages) => {
    const { text } = await qwenChat({
      messages,
      model,
      temperature,
      maxTokens,
      timeoutMs,
      enableThinking: false,
      responseFormat: { type: 'json_object' },
    });
    const json = extractJSON(text);
    if (json === null) {
      return { ok: false, error: 'could not parse JSON from response', text };
    }
    if (validate) {
      const v = validate(json);
      if (!v.ok) {
        return {
          ok: false,
          error: `schema validation failed: ${v.errors.join('; ')}`,
          text,
          data: json,
        };
      }
    }
    return { ok: true, data: json, text };
  };

  const first = await attempt(baseMessages);
  if (first.ok) return { data: first.data, text: first.text, attempts: 1 };

  const retryMessages = [
    ...baseMessages,
    { role: 'assistant', content: first.text || '' },
    {
      role: 'user',
      content:
        `Your previous response was not valid. Error:\n${first.error}\n\n` +
        `Respond ONLY with valid JSON matching the schema. No prose, no ` +
        `markdown fences. Fix the error and reply with the corrected JSON object.`,
    },
  ];
  const second = await attempt(retryMessages);
  if (second.ok) return { data: second.data, text: second.text, attempts: 2 };

  const err = new Error(`qwenChatJSON failed after 2 attempts: ${second.error}`);
  err.lastText = second.text;
  err.lastData = second.data;
  throw err;
}

/**
 * Extract a JSON object from a model response. Handles:
 *   - pure JSON
 *   - ```json … ``` fences
 *   - preamble before the first {
 */
export function extractJSON(text) {
  if (!text) return null;
  const stripped = text.replace(THINK_RE, '').trim();

  // Fenced block
  const fenced = stripped.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidate = fenced ? fenced[1] : stripped;

  // Find outermost {...} or [...]
  const firstObj = candidate.indexOf('{');
  const firstArr = candidate.indexOf('[');
  const starts = [firstObj, firstArr].filter((i) => i >= 0);
  if (starts.length === 0) return null;
  const start = Math.min(...starts);
  const openCh = candidate[start];
  const closeCh = openCh === '{' ? '}' : ']';

  let depth = 0;
  let inStr = false;
  let esc = false;
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
        const slice = candidate.slice(start, i + 1);
        try { return JSON.parse(slice); } catch { return null; }
      }
    }
  }
  return null;
}

/** Probe the proxy — useful to warm a cold Modal container. */
export async function qwenHealth() {
  const r = await fetch(CHAT_ENDPOINT, { method: 'GET' });
  return r.json();
}

// ────────────────────────────────────────────────────────────────
// Vision — Moondream 2 via the /api/vision Next proxy.

/** Probe the vision proxy. Returns { vision_ready, model }. */
export async function qwenVisionHealth() {
  const r = await fetch(VISION_ENDPOINT, { method: 'GET' });
  try { return await r.json(); } catch { return { vision_ready: false }; }
}

/** Read a File/Blob as a base64 data URL (browser). */
export async function fileToDataUrl(file) {
  if (!file) throw new Error('fileToDataUrl: file required');
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

/**
 * One Moondream call. Accepts a base64 string or data URL (either works —
 * the Modal endpoint strips the "data:...;base64," prefix server-side).
 *
 * @param {Object} opts
 * @param {string}  opts.imageBase64      — raw base64 or data URL
 * @param {string}  [opts.prompt]         — used when task === 'query' (default)
 * @param {'query'|'detect'|'point'} [opts.task='query']
 * @param {string}  [opts.object]         — required for detect/point
 * @param {number}  [opts.timeoutMs]      — default 300000 (cold-start budget)
 */
export async function qwenVisionAnalyze({
  imageBase64,
  prompt,
  task = 'query',
  object,
  timeoutMs = 300000,
} = {}) {
  if (!imageBase64) throw new Error('qwenVisionAnalyze: imageBase64 required');
  if ((task === 'detect' || task === 'point') && !object) {
    throw new Error(`qwenVisionAnalyze: task="${task}" requires object`);
  }
  const body = {
    image_base64: imageBase64,
    task,
  };
  if (prompt) body.prompt = prompt;
  if (object) body.object = object;

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(VISION_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    throw new Error(`qwenVision ${res.status}: ${errText.slice(0, 400)}`);
  }
  return res.json();
}

/**
 * Higher-level reference-image analyst. Asks Moondream 2 only the
 * questions it's *good* at — describing colors, listing visible text
 * and control inventory, and reporting the physical form factor — then
 * stitches the answers into a text block that feeds generatePluginDSL().
 *
 * Why only descriptive questions: Moondream 2 (1.93B) does well on
 * enumerating what it sees (labels, colors, control counts) but poorly
 * at archetype classification (we measured it misclassifying a
 * wavetable synth as a "digital reverb"). Archetype + chassis mistakes
 * dominate downstream output quality, so we let Qwen3 *infer* those
 * from the raw observations + a short rulebook of label → archetype
 * cues, instead of trusting Moondream's guess.
 *
 * Cost: 4 sequential Moondream queries, ~1-3s each warm.
 *
 * @param {Object} opts
 * @param {string} opts.imageBase64
 * @returns {Promise<{ text: string, answers: Object }>}
 */
export async function analyzePluginReferenceImage({ imageBase64 } = {}) {
  if (!imageBase64) throw new Error('analyzePluginReferenceImage: imageBase64 required');

  // Only descriptive questions. Archetype AND chassis are inferred by
  // Qwen downstream — Moondream tends to pattern-match on the most
  // visually salient cue (e.g. misreads "compact" rack units as pedals,
  // wavetable tables as reverb visualizations) and those misclassifications
  // propagate into the DSL. Palette + controls + visible text is enough.
  const questions = [
    { k: 'palette',
      q: 'Describe the dominant colors of the interface in plain English. Mention the background tone (dark/light/cream/etc.), the accent / highlight color, and any LED glow color.' },
    { k: 'controls',
      q: 'List the controls you can see, grouped by region. Prefix each control with KNOB:, SLIDER:, BUTTON:, LED:, DISPLAY:, METER:, or KEYBOARD:. Count as accurately as you can.' },
    { k: 'text',
      q: 'List any visible text labels, brand names, product names, model numbers, or preset names. Separate with commas.' },
  ];

  const answers = {};
  for (const { k, q } of questions) {
    try {
      const r = await qwenVisionAnalyze({ imageBase64, task: 'query', prompt: q });
      answers[k] = (r?.answer || '').trim();
    } catch (e) {
      answers[k] = `(vision error: ${e.message})`;
    }
  }

  const text =
    `REFERENCE IMAGE — RAW OBSERVATIONS (from Moondream 2, a small VLM).\n` +
    `Treat these as descriptions, NOT conclusions. You (the DSL model) must\n` +
    `INFER the archetype and chassis yourself from the visible labels and\n` +
    `control inventory below.\n` +
    `\n` +
    `  Palette:\n    ${answers.palette}\n` +
    `\n` +
    `  Visible controls:\n    ${answers.controls}\n` +
    `\n` +
    `  Visible text / labels / product names:\n    ${answers.text}\n` +
    `\n` +
    `ARCHETYPE INFERENCE CUES — use the "Visible text" and "Visible controls"\n` +
    `fields above to decide:\n` +
    `  - OSC / LFO / ENV / FILTER / WAVETABLE / UNISON / GRANULAR / a piano\n` +
    `    KEYBOARD → wavetable-synth or similar synth (use module-strip row).\n` +
    `  - DECAY / PRE-DELAY / PROGRAM / CROSSOVER / TAIL / 7-seg DISPLAYs with\n` +
    `    a bank of labeled BUTTONs → digital-reverb (use led-display +\n` +
    `    button-row + slider-bank, STRATA-style).\n` +
    `  - DRIVE / BIAS / TAPE / IPS / CAL / VU METER → tape-saturator or\n` +
    `    analog-bus processor.\n` +
    `  - CUTOFF / Q / FREQUENCY / SHELF / PEAK + curve DISPLAY → EQ.\n` +
    `  - THRESHOLD / RATIO / ATTACK / RELEASE + gain-reduction METER →\n` +
    `    compressor.\n` +
    `\n` +
    `CHASSIS INFERENCE CUES — infer from the archetype + text:\n` +
    `  - Software synths / effects / EQs (tabs like MAIN/MIX/FX, preset bar) →\n` +
    `    plugin-window. Default for any plugin-ish UI.\n` +
    `  - Classic rack digital hardware — named with 3-digit model numbers\n` +
    `    (224, 480, 960, 1176, 2254, LA-2A), with a bank of PROGRAM buttons\n` +
    `    and a 7-seg DISPLAY + faceplate sliders → rack-hardware.\n` +
    `  - Minimal control set (≤6 knobs, 1-2 footswitches, no tabs, no preset\n` +
    `    bar) typical of guitar pedals → pedal. Only use if the control\n` +
    `    inventory is genuinely pedal-sparse.\n` +
    `\n` +
    `IP SAFETY: if "Visible text" names a real product (Serum, Lexicon, Valhalla,\n` +
    `FabFilter, Pro-Q, Waves SSL, etc.), invent an ORIGINAL brand name for\n` +
    `meta.name and pick a palette accent hue ≥90° away from that product's\n` +
    `signature color.`;

  return { text, answers };
}
