/*
 * StudioDevChat — themed chat panel for the studio.
 *
 * Backend tier 1 (Phase B, this commit): server-side chat history via
 * /api/sessions/{sid}/chat-messages. The same endpoint the desktop
 * app writes to (Phase C), so a thread started on either side surfaces
 * on the other.
 *
 * Backend tier 2 (fallback / offline): localStorage write-through cache
 * keyed by `chat-${activeSessionId || 'local'}`. Used when:
 *   • no activeSessionId (local-only project not yet bootstrapped)
 *   • server fetch fails (offline, 401 mid-Clerk-handshake, etc.)
 * The cache is also re-populated on every successful server load so a
 * subsequent reload paints instantly while reconciling in the background.
 *
 * LLM: Modal vLLM Qwen3-14B-AWQ via the Next.js proxy at /api/chat
 * (services/qwenChat.js → app/api/chat/route.ts). Replaces the dead
 * /_chat/ws path that produced 100%-failure WS spam.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { qwenChatStream } from '../../services/qwenChat';
import { compactIfNeeded } from '../../services/chatCompact';
import {
  listChatMessages,
  appendChatMessage,
  streamChatMessages,
  newClientOpId,
} from '../../services/chatMessagesAPI';

const STORAGE_PREFIX = 'chat-';
const MAX_PERSISTED_MESSAGES = 200;        // soft cap so localStorage doesn't blow up
const SERVER_PAGE_LIMIT = 200;

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

function _serverRowToLocal(row) {
  return {
    id: row.id,
    role: row.role,
    text: row.content,
    clientOpId: row.client_op_id,
    createdAt: row.created_at,
    authorOrigin: row.author_origin,
    fromServer: true,
  };
}

export default function StudioDevChat({ onClose }) {
  const { state } = useApp();
  const sessionId = state?.activeSessionId || null;

  // Hydrate from localStorage immediately so the panel paints fast,
  // even before the server fetch lands.
  const [messages, setMessages] = useState(() => _loadHistory(sessionId));
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [serverSynced, setServerSynced] = useState(false);

  const scrollRef = useRef(null);
  const abortRef = useRef(null);

  // Reconcile from server whenever sessionId changes (and on mount).
  // Server returns newest-first; UI expects oldest-first. Falls back
  // silently to the localStorage view if the fetch fails. After the
  // initial GET lands, open an SSE stream so concurrent edits from
  // other clients (web tab, desktop session, peer) flow in live.
  useEffect(() => {
    setMessages(_loadHistory(sessionId));
    setServerSynced(false);
    setError(null);
    if (!sessionId) return;
    let cancelled = false;
    let unsubscribe = () => {};

    listChatMessages(sessionId, { limit: SERVER_PAGE_LIMIT })
      .then((rows) => {
        if (cancelled) return;
        const ordered = (Array.isArray(rows) ? rows : []).slice().reverse().map(_serverRowToLocal);
        setMessages(ordered);
        _saveHistory(sessionId, ordered);
        setServerSynced(true);

        // Subscribe to live tail. `since` = timestamp of newest known
        // message → server replays anything we missed between the GET
        // and the SSE subscribe, then pushes new rows as they land.
        const newest = ordered[ordered.length - 1];
        const sinceArg = newest?.createdAt || null;
        unsubscribe = streamChatMessages(sessionId, {
          since: sinceArg,
          onMessage: (row) => {
            if (cancelled) return;
            setMessages((prev) => {
              // Dedupe by client_op_id (the value the local POST minted)
              // and by server id. If we already have this message —
              // because we sent it ourselves and the local optimistic
              // append is still on screen — replace it with the server
              // copy (gets the canonical id + created_at). Otherwise
              // append.
              const opId = row.client_op_id || null;
              const idx = prev.findIndex((m) =>
                (opId && m.clientOpId === opId)
                || (m.id === row.id)
              );
              const mapped = _serverRowToLocal(row);
              if (idx >= 0) {
                if (prev[idx].fromServer && prev[idx].id === mapped.id) return prev;
                const next = prev.slice();
                next[idx] = { ...prev[idx], ...mapped };
                _saveHistory(sessionId, next);
                return next;
              }
              const next = prev.concat([mapped]);
              _saveHistory(sessionId, next);
              return next;
            });
          },
          onError: () => {
            // EventSource auto-reconnects on transport errors; nothing
            // to do here. We log on the first error in case it's a
            // genuine 401/403 that won't recover.
            if (!cancelled) console.warn('[chat] SSE error (will auto-reconnect)');
          },
        });
      })
      .catch((err) => {
        if (cancelled) return;
        if (err?.status !== 401 && err?.status !== 403) {
          console.warn('[chat] listChatMessages failed; using localStorage cache:', err?.message || err);
        }
        setServerSynced(false);
      });

    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [sessionId]);

  // Auto-scroll on new messages / token streaming.
  useEffect(() => {
    const el = scrollRef.current; if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  // Cancel any in-flight stream on unmount.
  useEffect(() => () => { abortRef.current?.abort?.(); }, []);

  const systemPrompt = useMemo(() => _buildSystemPrompt(state), [state?.bpm, state?.generationParams?.key, state?.buses]);

  // Best-effort POST that doesn't block the UI. Logs but never throws.
  const persistToServer = useCallback(async ({ role, content, clientOpId }) => {
    if (!sessionId) return null;
    try {
      return await appendChatMessage(sessionId, {
        role,
        content,
        clientOpId,
        authorOrigin: 'web',
      });
    } catch (err) {
      console.warn(`[chat] append ${role} failed (will retry on next flush):`, err?.message || err);
      return null;
    }
  }, [sessionId]);

  const sendChat = useCallback(async (text) => {
    if (!text || busy) return;
    setError(null);

    const userOp = newClientOpId();
    const asstOp = newClientOpId();
    const userMsg = {
      id: `u-${Date.now()}`,
      role: 'user',
      text,
      clientOpId: userOp,
    };
    const asstMsg = {
      id: `a-${Date.now()}`,
      role: 'assistant',
      text: '',
      clientOpId: asstOp,
    };
    const baseMessages = messages.concat([userMsg, asstMsg]);
    setMessages(baseMessages);
    _saveHistory(sessionId, baseMessages.slice(0, -1));

    // Fire user-message POST in parallel with the LLM stream so the
    // server has the row even if the assistant call fails.
    persistToServer({ role: 'user', content: text, clientOpId: userOp });

    const rawQwenMessages = [
      { role: 'system', content: systemPrompt },
      ...messages
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .map((m) => ({ role: m.role, content: m.text || m.content || '' })),
      { role: 'user', content: text },
    ];

    setBusy(true);
    // Compact older turns when we're approaching the context window.
    // Pure on no-op (returns the same array); on hit, the older middle
    // is replaced by a single system summary.
    let qwenMessages = rawQwenMessages;
    try {
      const result = await compactIfNeeded(rawQwenMessages);
      if (result.compacted) {
        console.log(`[chat] compacted ${result.originalTokens}→${result.newTokens} tokens`);
      }
      qwenMessages = result.messages;
    } catch (compactErr) {
      console.warn('[chat] compaction failed, sending full history:', compactErr?.message || compactErr);
    }

    let finalText = '';
    try {
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      await qwenChatStream({
        messages: qwenMessages,
        sessionId,           // tells qwenChat to POST /api/usage/record after stream
        clientOpId: asstOp,  // idempotent so retries don't double-count
        onToken: (_delta, full) => {
          finalText = full;
          setMessages((prev) => prev.map((m) => m.id === asstMsg.id ? { ...m, text: full } : m));
        },
      });
      setMessages((prev) => {
        _saveHistory(sessionId, prev);
        return prev;
      });
      // Persist the final assistant text. Idempotent on asstOp so a
      // retry from the user re-asking won't dupe.
      if (finalText) {
        persistToServer({ role: 'assistant', content: finalText, clientOpId: asstOp });
      }
    } catch (err) {
      const msg = err?.message || String(err);
      console.warn('[chat] qwenChatStream failed:', msg);
      setError(msg);
      setMessages((prev) => prev.map((m) =>
        m.id === asstMsg.id ? { ...m, text: `⚠️ ${msg}` } : m
      ));
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }, [busy, messages, sessionId, systemPrompt, persistToServer]);

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
    // Note: server-side messages remain. Adding a DELETE endpoint is a
    // follow-up; for now Clear just resets the local view so a reload
    // re-pulls from server.
  };

  return (
    <div className="sd-chat">
      <div className="sd-midi-toolbar">
        <div className="sd-midi-title">
          <span className="sd-midi-meta">CHAT</span>
          <span className="sd-midi-name" style={{ marginLeft: 6 }}>Qwen3-14B</span>
          <span className="sd-midi-meta" style={{ marginLeft: 8, color: 'var(--hifi-accent)' }}>
            {busy ? '◐ streaming' : sessionId && serverSynced ? '☁ synced' : '● ready'}
          </span>
        </div>
        <div className="sd-midi-spacer" />
        {messages.length > 0 && (
          <button className="sd-midi-btn" onClick={clearThread} title="Clear local view (server thread is preserved)">
            Clear
          </button>
        )}
        {onClose && <button className="sd-midi-btn" onClick={onClose}>Close</button>}
      </div>

      <div className="sd-chat-messages" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="sd-chat-empty">
            Ask about your mix, chords, progressions, sound design — Qwen3 on Modal answers, and the thread syncs across web + desktop when this session is server-synced.
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
