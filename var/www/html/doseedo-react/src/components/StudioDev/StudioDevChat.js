/*
 * StudioDevChat — themed chat panel for /studio-dev. Wraps the existing
 * useAgentWebSocket hook (same WS endpoint the production ChatWindow
 * uses), but with a minimal hi-fi UI.
 */
import React, { useEffect, useRef, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { useAgentWebSocket } from '../ChatWindow/useAgentWebSocket';

// The useAgentWebSocket hook's real surface:
//   messages: [{ id, role:'user'|'assistant'|'system'|'tool', text, ... }]
//   busy:    bool — agent is working
//   wsState: 'connecting' | 'open' | 'closed'
//   sendChat(text): dispatches a user message
export default function StudioDevChat({ onClose }) {
  const { state } = useApp();
  const hook = useAgentWebSocket(state) || {};
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
