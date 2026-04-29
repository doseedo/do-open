/**
 * chatCompact — token-budget compaction for long Qwen chat threads.
 *
 * Adapted from ccdevtemplatesrc/services/compact: when the conversation
 * gets within `BUFFER_TOKENS` of the model's context window, summarize
 * the older messages into a single system note and keep only the most
 * recent `KEEP_TAIL` turns intact. The summary uses the template's
 * 9-section prompt verbatim — universal, not Anthropic-specific —
 * pointed at /api/chat (Modal vLLM Qwen).
 *
 * Returns either the original messages list (no compaction needed) or
 * the new compacted list. Pure: doesn't mutate inputs.
 */

import { qwenChat } from './qwenChat';

// Qwen3-14B-AWQ context window (per Modal deploy). Buffer reserves
// room for a generous output. Numbers mirror the template's
// AUTOCOMPACT_BUFFER_TOKENS = 13_000 with the same intent.
const CONTEXT_WINDOW = 32_768;
const BUFFER_TOKENS  = 8_000;
const TRIGGER_AT     = CONTEXT_WINDOW - BUFFER_TOKENS;   // 24,768
const KEEP_TAIL_MSGS = 6;                                // last 3 turns of each role

// Char-ratio estimator. ~3.5 chars per token across mixed English +
// chat JSON; conservative on the high side so we trigger early
// rather than blow the window.
export function estimateTokens(text) {
  if (!text) return 0;
  return Math.ceil(String(text).length / 3.5);
}

export function estimateMessagesTokens(messages) {
  if (!Array.isArray(messages)) return 0;
  let total = 0;
  for (const m of messages) {
    total += estimateTokens(m?.content);
    total += 4;                       // role + framing overhead
  }
  return total;
}

const SUMMARIZER_SYSTEM = [
  'You are a chat-history summarizer for a music production assistant.',
  'You will be shown an older portion of a conversation between a user and the assistant.',
  'Produce a concise structured summary the assistant can read on its next turn to keep continuity.',
  'Output sections, each with a short heading and 1–4 bullet lines:',
  '  - User intent: what the user wants overall.',
  '  - Concepts: musical/technical terms established (BPM, key, instruments, refs).',
  '  - Decisions: choices made or rejected.',
  '  - Open questions: anything the user asked that wasn\'t fully answered.',
  '  - Pending work: actions the assistant promised to do but hasn\'t finished.',
  'Be specific. No fluff. No tool calls. No code blocks unless quoting code that was discussed.',
].join('\n');

/**
 * Decide whether to compact, and if so, return the rewritten message
 * list. The system prompt (messages[0] when role === 'system') is
 * preserved as-is; older user/assistant turns are folded into a
 * single { role: 'system', content: '<summary>' } block followed by
 * the most recent KEEP_TAIL_MSGS messages.
 *
 * Returns { messages, compacted, originalTokens, newTokens }.
 */
export async function compactIfNeeded(messages, { force = false, model } = {}) {
  if (!Array.isArray(messages) || messages.length < KEEP_TAIL_MSGS + 4) {
    return { messages, compacted: false, originalTokens: estimateMessagesTokens(messages), newTokens: 0 };
  }
  const originalTokens = estimateMessagesTokens(messages);
  if (!force && originalTokens < TRIGGER_AT) {
    return { messages, compacted: false, originalTokens, newTokens: originalTokens };
  }

  // Split: keep the leading system prompt(s), then everything older
  // than the tail goes to the summarizer, then preserve the tail.
  let i = 0;
  const head = [];
  while (i < messages.length && messages[i].role === 'system') {
    head.push(messages[i]); i += 1;
  }
  const tailStart = Math.max(i, messages.length - KEEP_TAIL_MSGS);
  const middle = messages.slice(i, tailStart);
  const tail = messages.slice(tailStart);
  if (middle.length === 0) {
    return { messages, compacted: false, originalTokens, newTokens: originalTokens };
  }

  // Render the middle as a transcript the summarizer can read.
  const transcript = middle
    .map((m) => `[${m.role}]\n${(m.content || '').toString().trim()}`)
    .join('\n\n');

  let summaryText;
  try {
    const { text } = await qwenChat({
      messages: [
        { role: 'system', content: SUMMARIZER_SYSTEM },
        { role: 'user', content: `Summarize this older portion of the chat:\n\n${transcript}` },
      ],
      model,
      temperature: 0.2,
      maxTokens: 1024,
    });
    summaryText = (text || '').trim();
  } catch (err) {
    console.warn('[compact] summarizer call failed:', err?.message || err);
    return { messages, compacted: false, originalTokens, newTokens: originalTokens };
  }
  if (!summaryText) {
    return { messages, compacted: false, originalTokens, newTokens: originalTokens };
  }

  const summaryNote = {
    role: 'system',
    content: [
      '— Compacted history (older turns summarized) —',
      summaryText,
      '— End of compacted history —',
    ].join('\n'),
  };
  const compacted = [...head, summaryNote, ...tail];
  return {
    messages: compacted,
    compacted: true,
    originalTokens,
    newTokens: estimateMessagesTokens(compacted),
  };
}

export const _internal = { TRIGGER_AT, CONTEXT_WINDOW, KEEP_TAIL_MSGS };
