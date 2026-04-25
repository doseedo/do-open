/*
 * StudioDevChat — themed chat panel for /studio-dev. The chat WebSocket
 * itself is owned by StudioDev (so it survives panel switches and feeds
 * the A5 live param-delta listener); this component just renders against
 * the hook handle passed in via `chatWs`.
 *
 * Backwards-compatible: if the prop is missing (e.g. mounted outside
 * StudioDev), fall back to spinning up an own WS via useAgentWebSocket.
 * This keeps any one-off / preview embeddings working.
 */
import React, { useEffect, useRef, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { useAgentWebSocket } from '../ChatWindow/useAgentWebSocket';

// The useAgentWebSocket hook's real surface:
//   messages: [{ id, role:'user'|'assistant'|'system'|'tool', text, ... }]
//   busy:    bool — agent is working
//   wsState: 'connecting' | 'open' | 'closed'
//   sendChat(text): dispatches a user message
export default function StudioDevChat({ onClose, chatWs }) {
  const { state } = useApp();
  // Prefer the lifted hook from StudioDev. The fallback spins up its own
  // WS only when no parent passed one in — never both.
  // Always call the hook (rules of hooks), but disable the WS when the
  // parent already lifted one — that path returns sentinel state without
  // opening a connection.
  const fallback = useAgentWebSocket(state, { enabled: !chatWs });
  const hook = chatWs || fallback || {};
  const { messages = [], busy = false, wsState = 'closed', sendChat } = hook;

  const [input, setInput] = useState('');
  const scrollRef = useRef(null);

  // Auto-scroll on new messages.
  useEffect(() => {
    const el = scrollRef.current; if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages.length, busy]);

  const submit = (e) => {
    e.preventDefault();
    const txt = input.trim();
    if (!txt || typeof sendChat !== 'function') return;
    sendChat(txt);
    setInput('');
  };

  const isConnected = wsState === 'open';

  return (
    <div className="sd-chat">
      <div className="sd-midi-toolbar">
        <div className="sd-midi-title">
          <span className="sd-midi-meta">CHAT</span>
          <span className="sd-midi-name" style={{ marginLeft: 6 }}>Assistant</span>
          <span className="sd-midi-meta" style={{ marginLeft: 8,
                color: isConnected ? 'var(--hifi-accent)' : '#e07556' }}>
            {isConnected ? '● online' : wsState === 'connecting' ? '◐ connecting' : '○ offline'}
          </span>
        </div>
        <div className="sd-midi-spacer" />
        {onClose && <button className="sd-midi-btn" onClick={onClose}>Close</button>}
      </div>

      <div className="sd-chat-messages" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="sd-chat-empty">
            Ask about your mix, chords, progressions — it can also drive the studio.
          </div>
        )}
        {messages.map((m, i) => (
          <div key={m.id ?? i} className={`sd-chat-msg ${m.role || 'assistant'}`}>
            <div className="sd-chat-msg-role">{m.role || 'assistant'}</div>
            <div className="sd-chat-msg-body">{m.text ?? m.content ?? ''}</div>
          </div>
        ))}
        {busy && <div className="sd-chat-typing">assistant is thinking…</div>}
      </div>

      <form onSubmit={submit} className="sd-chat-input-row">
        <input
          className="sd-chat-input"
          placeholder={isConnected ? 'Ask or command…' : 'Connecting…'}
          value={input} onChange={(e) => setInput(e.target.value)}
          disabled={!isConnected}
        />
        <button className="sd-btn" type="submit" disabled={!input.trim() || !isConnected}>Send</button>
      </form>
    </div>
  );
}
