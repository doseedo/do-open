/*
 * StudioDevChat — themed chat panel for the studio.
 *
 * Backend: Modal vLLM Qwen3-14B-AWQ via the Next.js proxy at /api/chat
 * (see services/qwenChat.js + app/api/chat/route.ts). Replaces the
 * legacy /_chat/ws WebSocket path, which was a desktop-only chat
 * server (logic_engine/chat_server.py on ws://127.0.0.1:{port}/ws)
 * never deployed at doseedo.com → produced 100%-failure WS spam in
 * the browser.
 *
 * Persistence: messages cached in localStorage keyed by
 * `chat-${activeSessionId || 'local'}`, so the thread survives reloads
 * and panel switches. True web↔desktop sync requires a server-side
 * `chat_messages` table on the auth-service (Phase B in the unification
 * plan); this commit is the frontend-only first step.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { qwenChatStream } from '../../services/qwenChat';

const STORAGE_PREFIX = 'chat-';
const MAX_PERSISTED_MESSAGES = 200;        // soft cap so localStorage doesn't blow up

function _storageKey(sessionId) {
  return `${STORAGE_PREFIX}${sessionId || 'local'}`;
}

function _loadHistory(sessionId) {
  try {
    const raw = localStorage.getItem(_storageKey(sessionId));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch { return []; }
}

function _saveHistory(sessionId, messages) {
  try {
    const trimmed = messages.length > MAX_PERSISTED_MESSAGES
      ? messages.slice(-MAX_PERSISTED_MESSAGES)
      : messages;
    localStorage.setItem(_storageKey(sessionId), JSON.stringify(trimmed));
  } catch { /* quota — silently drop */ }
}

function _buildSystemPrompt(state) {
  const bpm = state?.bpm || 120;
  const key = state?.generationParams?.key || 'C';
  const trackCount = state?.buses?.reduce((t, b) => t + (b.tracks?.length || 0), 0) || 0;
  return [
    'You are the doseedo studio assistant — a music production copilot.',
    'Be concise. Suggest concrete actions when asked about mixing, chords, arrangement, sound design.',
    `DAW context: ${bpm} BPM, key ${key}, ${trackCount} track${trackCount === 1 ? '' : 's'}.`,
  ].join(' ');
}

export default function StudioDevChat({ onClose }) {
  const { state } = useApp();
  const sessionId = state?.activeSessionId || null;

  const [messages, setMessages] = useState(() => _loadHistory(sessionId));
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const scrollRef = useRef(null);
  const abortRef = useRef(null);

  // Re-load history when the active session swaps (e.g. user switches
  // projects without remounting the chat).
  useEffect(() => {
    setMessages(_loadHistory(sessionId));
    setError(null);
  }, [sessionId]);

  // Auto-scroll on new messages / token streaming.
  useEffect(() => {
    const el = scrollRef.current; if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  // Cancel any in-flight stream on unmount.
  useEffect(() => () => { abortRef.current?.abort?.(); }, []);

  const systemPrompt = useMemo(() => _buildSystemPrompt(state), [state?.bpm, state?.generationParams?.key, state?.buses]);

  const sendChat = useCallback(async (text) => {
    if (!text || busy) return;
    setError(null);

    // Append user + empty assistant placeholder; persist immediately
    // so a refresh mid-stream doesn't lose the question.
    const userMsg = { id: `u-${Date.now()}`, role: 'user', text };
    const asstId = `a-${Date.now()}`;
    const asstMsg = { id: asstId, role: 'assistant', text: '' };
    const baseMessages = messages.concat([userMsg, asstMsg]);
    setMessages(baseMessages);
    _saveHistory(sessionId, baseMessages.slice(0, -1));     // skip empty assistant from disk

    // Build Qwen-shaped messages: system + history + new user.
    const qwenMessages = [
      { role: 'system', content: systemPrompt },
      ...messages
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .map((m) => ({ role: m.role, content: m.text || m.content || '' })),
      { role: 'user', content: text },
    ];

    setBusy(true);
    try {
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      await qwenChatStream({
        messages: qwenMessages,
        onToken: (_delta, full) => {
          setMessages((prev) => prev.map((m) => m.id === asstId ? { ...m, text: full } : m));
        },
      });
      // Persist the final thread.
      setMessages((prev) => {
        _saveHistory(sessionId, prev);
        return prev;
      });
    } catch (err) {
      const msg = err?.message || String(err);
      console.warn('[chat] qwenChatStream failed:', msg);
      setError(msg);
      setMessages((prev) => prev.map((m) =>
        m.id === asstId ? { ...m, text: `⚠️ ${msg}` } : m
      ));
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }, [busy, messages, sessionId, systemPrompt]);

  const submit = (e) => {
    e.preventDefault();
    const txt = input.trim();
    if (!txt) return;
    sendChat(txt);
    setInput('');
  };

  const clearThread = () => {
    setMessages([]);
    _saveHistory(sessionId, []);
    setError(null);
  };

  return (
    <div className="sd-chat">
      <div className="sd-midi-toolbar">
        <div className="sd-midi-title">
          <span className="sd-midi-meta">CHAT</span>
          <span className="sd-midi-name" style={{ marginLeft: 6 }}>Qwen3-14B</span>
          <span className="sd-midi-meta" style={{ marginLeft: 8, color: 'var(--hifi-accent)' }}>
            {busy ? '◐ streaming' : '● ready'}
          </span>
        </div>
        <div className="sd-midi-spacer" />
        {messages.length > 0 && (
          <button className="sd-midi-btn" onClick={clearThread} title="Clear this session's thread">
            Clear
          </button>
        )}
        {onClose && <button className="sd-midi-btn" onClick={onClose}>Close</button>}
      </div>

      <div className="sd-chat-messages" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="sd-chat-empty">
            Ask about your mix, chords, progressions, sound design — Qwen3 on Modal answers.
          </div>
        )}
        {messages.map((m, i) => (
          <div key={m.id ?? i} className={`sd-chat-msg ${m.role || 'assistant'}`}>
            <div className="sd-chat-msg-role">{m.role || 'assistant'}</div>
            <div className="sd-chat-msg-body">{m.text ?? m.content ?? ''}</div>
          </div>
        ))}
        {busy && messages[messages.length - 1]?.text === '' && (
          <div className="sd-chat-typing">qwen is thinking…</div>
        )}
        {error && !busy && (
          <div className="sd-chat-typing" style={{ color: '#e07556' }}>error: {error}</div>
        )}
      </div>

      <form onSubmit={submit} className="sd-chat-input-row">
        <input
          className="sd-chat-input"
          placeholder={busy ? 'Streaming…' : 'Ask or command…'}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={busy}
        />
        <button className="sd-btn" type="submit" disabled={!input.trim() || busy}>Send</button>
      </form>
    </div>
  );
}
