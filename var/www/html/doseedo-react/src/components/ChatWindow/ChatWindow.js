import React, { useState, useRef, useEffect, useCallback, memo } from 'react';
import { useApp } from '../../context/AppContext';
import styles from './ChatWindow.module.css';
import { useAgentWebSocket } from './useAgentWebSocket';
import SlashCommandPalette from './SlashCommandPalette';

/**
 * ChatWindow — Full agent chatbot with streaming, tool execution, slash commands.
 * Replaces the legacy request/response chat with the same system as the desktop app.
 */

const CLAUDE_INDIGO = '#8B7FF0';
const STAR_FRAMES = ['\u280B','\u2819','\u2839','\u2838','\u283C','\u2834','\u2826','\u2827','\u2807','\u280F'];

// ── Helpers ──

function formatElapsed(startMs, endMs) {
  const ms = (endMs ?? Date.now()) - startMs;
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/** Highlight session objects: quoted strings, track N, BPM, Hz, dB, time sig */
function SessionText({ text }) {
  const pattern = /"[^"]+"|(?:\b(?:track|input|output|bus|channel)\s+\d+\b)|(?:\b[\d.]+\s*(?:BPM|bpm|Hz|kHz|dB|ms)\b)|(?:\b\d+\/\d+\b)/gi;
  const parts = [];
  let lastIndex = 0;
  let key = 0;
  for (const match of text.matchAll(pattern)) {
    const start = match.index;
    if (start > lastIndex)
      parts.push(<span key={key++} style={{ color: '#CCCCCC' }}>{text.slice(lastIndex, start)}</span>);
    parts.push(<span key={key++} style={{ color: '#8EC8F0' }}>{match[0]}</span>);
    lastIndex = start + match[0].length;
  }
  if (lastIndex < text.length)
    parts.push(<span key={key++} style={{ color: '#CCCCCC' }}>{text.slice(lastIndex)}</span>);
  return <span>{parts.length ? parts : <span style={{ color: '#CCCCCC' }}>{text}</span>}</span>;
}

// ── Tool Row ──

function ToolRow({ tool, frame }) {
  const glyph = tool.status === 'running'
    ? <span style={{ color: CLAUDE_INDIGO }}>{STAR_FRAMES[frame % STAR_FRAMES.length]}</span>
    : tool.status === 'ok'
    ? <span style={{ color: '#4CAF50' }}>{'\u2713'}</span>
    : <span style={{ color: '#F44336' }}>{'\u2717'}</span>;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '1px 0', fontSize: '0.82em' }}>
      {glyph}
      <span style={{ color: '#888888' }}>{tool.name.replace(/_/g, ' ')}</span>
      <span style={{ color: '#444444', fontSize: '0.88em' }}>
        {tool.endMs ? formatElapsed(tool.startMs, tool.endMs) : '\u2026'}
      </span>
      {tool.status === 'error' && tool.result && (
        <span style={{ color: '#F44336', opacity: 0.75 }}>{'\u2014'} {tool.result.slice(0, 60)}</span>
      )}
    </div>
  );
}

// ── Message Components ──

const UserMsg = memo(function UserMsg({ text }) {
  return (
    <div style={{
      background: '#1a1a1a',
      color: '#aaaaaa',
      padding: '3px 8px',
      borderRadius: 2,
      whiteSpace: 'pre-wrap',
      wordBreak: 'break-word',
    }}>
      <span style={{ color: CLAUDE_INDIGO, marginRight: 6 }}>{'\u276F'}</span>
      {text}
    </div>
  );
});

const SystemMsg = memo(function SystemMsg({ text }) {
  return (
    <div style={{ color: '#555555', fontStyle: 'italic', fontSize: '0.82em' }}>
      {text}
    </div>
  );
});

function AssistantMsg({ msg, frame }) {
  const verb = msg.verb ?? 'Generating';

  const streamingLabel = msg.streaming
    ? (
      <div style={{ color: CLAUDE_INDIGO, display: 'flex', alignItems: 'center', gap: 6 }}>
        <span>{STAR_FRAMES[frame % STAR_FRAMES.length]}</span>
        <span style={{ color: '#888888' }}>{verb}{'\u2026'}</span>
      </div>
    ) : null;

  const thinkingLine = msg.thinkingStatus === 'thinking'
    ? <div style={{ color: '#555555' }}>{'\u2234'} Thinking{'\u2026'}</div>
    : typeof msg.thinkingStatus === 'number'
    ? <div style={{ color: '#444444' }}>{'\u2234'} Thought for {(msg.thinkingStatus / 1000).toFixed(1)}s</div>
    : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {streamingLabel}
      {thinkingLine}
      {!msg.streaming && msg.text && (
        <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', lineHeight: 1.6 }}>
          <SessionText text={msg.text} />
        </div>
      )}
      {msg.tools.length > 0 && (
        <div style={{
          borderLeft: '1px solid #2a2a2a',
          paddingLeft: 10,
          marginTop: 2,
          display: 'flex',
          flexDirection: 'column',
        }}>
          <div style={{ color: '#333333', fontSize: '0.78em', marginBottom: 2 }}>{'\u23BF'}</div>
          {msg.tools.map(t => <ToolRow key={t.id} tool={t} frame={frame} />)}
        </div>
      )}
    </div>
  );
}

// ── Permission Dialog ──

function PermissionDialog({ request, onRespond }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setElapsed(e => e + 1), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'Enter') { e.preventDefault(); onRespond(true); }
      if (e.key === 'Escape') { e.preventDefault(); onRespond(false); }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onRespond]);

  return (
    <div className={styles.permissionDialog}>
      <div style={{ color: '#B1B9F9', fontSize: '0.85em', marginBottom: 4 }}>
        {'\u26A0'} Permission requested
      </div>
      <div style={{ color: '#ccc', fontSize: '0.82em' }}>
        <strong>{request.toolName?.replace(/_/g, ' ')}</strong>
      </div>
      {request.description && (
        <div style={{ color: '#888', fontSize: '0.78em', marginTop: 2 }}>{request.description}</div>
      )}
      <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
        <button className={styles.permBtnApprove} onClick={() => onRespond(true)}>
          Approve (Enter)
        </button>
        <button className={styles.permBtnDeny} onClick={() => onRespond(false)}>
          Deny (Esc)
        </button>
      </div>
      <div style={{ color: '#444', fontSize: '0.7em', marginTop: 4 }}>{60 - elapsed}s</div>
    </div>
  );
}

// ── Question Dialog ──

function QuestionDialog({ request, onRespond }) {
  const [answer, setAnswer] = useState('');
  const inputRef = useRef(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  const submit = () => {
    if (answer.trim()) onRespond(answer.trim());
  };

  return (
    <div className={styles.questionDialog}>
      <div style={{ color: '#8B7FF0', fontSize: '0.85em', marginBottom: 4 }}>
        {'\u2753'} Agent is asking:
      </div>
      <div style={{ color: '#ccc', fontSize: '0.82em', marginBottom: 8 }}>{request.question}</div>
      <div style={{ display: 'flex', gap: 6 }}>
        <input
          ref={inputRef}
          className={styles.questionInput}
          value={answer}
          onChange={e => setAnswer(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); submit(); } }}
          placeholder="Type your answer..."
        />
        <button className={styles.permBtnApprove} onClick={submit} disabled={!answer.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}

// ── Todo Panel ──

function TodoPanel({ todos }) {
  if (!todos || todos.length === 0) return null;
  const done = todos.filter(t => t.done).length;
  return (
    <div className={styles.todoPanel}>
      <div style={{ color: '#555', fontSize: '0.75em', marginBottom: 3 }}>
        Tasks {done}/{todos.length}
      </div>
      {todos.map((t, i) => (
        <div key={i} style={{ color: t.done ? '#4CAF50' : '#888', fontSize: '0.78em', display: 'flex', gap: 4 }}>
          <span>{t.done ? '\u2713' : '\u25C7'}</span>
          <span>{t.text}</span>
        </div>
      ))}
    </div>
  );
}

// ── Main ChatWindow Component ──

const ChatWindow = ({ onClose }) => {
  const { state } = useApp();
  const [input, setInput] = useState('');
  const [frame, setFrame] = useState(0);
  const [showCommands, setShowCommands] = useState(false);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  const {
    messages, busy, wsState, mode, model, contextPct,
    inputTokens, outputTokens, totalCost,
    pendingPermission, pendingQuestion, todos,
    sendChat, sendRaw, respondPermission, respondQuestion,
    setMode, setModel,
  } = useAgentWebSocket(state);

  // Spinner animation when streaming
  const isStreaming = messages.some(m => m.role === 'assistant' && m.streaming);
  useEffect(() => {
    if (!isStreaming) return;
    const id = setInterval(() => setFrame(f => f + 1), 80);
    return () => clearInterval(id);
  }, [isStreaming]);

  // Auto-scroll
  const lastMsg = messages[messages.length - 1];
  useEffect(() => {
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 30);
  }, [messages.length, lastMsg]);

  // Focus
  useEffect(() => { textareaRef.current?.focus(); }, []);

  // ── Slash commands ──
  const COMMANDS = [
    { name: '/help', desc: 'Show all commands', action: () => alert(COMMANDS.map(c => `${c.name} — ${c.desc}`).join('\n')) },
    { name: '/clear', desc: 'Clear conversation', action: () => sendRaw({ type: 'clear_history' }) },
    { name: '/status', desc: 'Show project status', action: () => sendRaw({ type: 'status_request' }) },
    { name: '/tracks', desc: 'List all tracks', action: () => sendChat('List all tracks in this session') },
    { name: '/session', desc: 'Show session info', action: () => sendChat('Show full session info') },
    { name: '/model', desc: 'Switch AI model', action: (args) => setModel(args) },
    { name: '/mode', desc: 'Set mode (default|auto|plan)', action: (args) => setMode(args) },
    { name: '/auto', desc: 'Toggle auto-approve', action: () => setMode(mode === 'auto' ? 'default' : 'auto') },
    { name: '/plan', desc: 'Toggle plan mode', action: () => setMode(mode === 'plan' ? 'default' : 'plan') },
    { name: '/compact', desc: 'Compress history', action: () => sendRaw({ type: 'compact' }) },
    { name: '/cost', desc: 'Show token usage', action: () => {
      const cost = `Tokens: ${inputTokens.toLocaleString()} in / ${outputTokens.toLocaleString()} out\nCost: $${totalCost.toFixed(4)}\nContext: ${(contextPct * 100).toFixed(0)}%`;
      alert(cost);
    }},
    { name: '/sync', desc: 'Sync session to web DAW', action: () => sendChat('Sync my current session to the web DAW') },
    { name: '/cancel', desc: 'Cancel current operation', action: () => sendRaw({ type: 'cancel' }) },
  ];

  const commandFilter = showCommands && input.startsWith('/') ? input.slice(1).toLowerCase() : '';
  const filteredCommands = showCommands
    ? COMMANDS.filter(c => !commandFilter || c.name.slice(1).startsWith(commandFilter) || c.desc.toLowerCase().includes(commandFilter))
    : [];

  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text || busy) return;

    // Slash command
    if (text.startsWith('/')) {
      const parts = text.split(' ');
      const cmdName = parts[0].toLowerCase();
      const args = parts.slice(1).join(' ');
      const cmd = COMMANDS.find(c => c.name === cmdName);
      if (cmd) {
        cmd.action(args);
        setInput('');
        setShowCommands(false);
        return;
      }
    }

    sendChat(text);
    setInput('');
    setShowCommands(false);
  }, [input, busy, sendChat, COMMANDS]);

  const handleChange = (e) => {
    const val = e.target.value;
    setInput(val);
    setShowCommands(val.startsWith('/'));
  };

  const [selectedCmdIdx, setSelectedCmdIdx] = useState(0);

  const handleKeyDown = (e) => {
    if (showCommands && filteredCommands.length > 0) {
      if (e.key === 'ArrowDown') { e.preventDefault(); setSelectedCmdIdx(i => Math.min(i + 1, filteredCommands.length - 1)); return; }
      if (e.key === 'ArrowUp') { e.preventDefault(); setSelectedCmdIdx(i => Math.max(i - 1, 0)); return; }
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const cmd = filteredCommands[selectedCmdIdx];
        if (cmd) {
          const parts = input.split(' ');
          cmd.action(parts.slice(1).join(' '));
          setInput('');
          setShowCommands(false);
        }
        return;
      }
      if (e.key === 'Escape') { e.preventDefault(); setShowCommands(false); return; }
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const visibleMessages = messages.filter(m => m.role !== 'system' || m.text?.trim());

  // Context bar color
  const ctxColor = contextPct > 0.8 ? '#F44336' : contextPct > 0.6 ? '#FFC107' : '#4CAF50';

  return (
    <div className={styles.chatWindow}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <span className={styles.headerDot} style={{
            background: wsState === 'open' ? '#4CAF50' : wsState === 'connecting' ? '#FFC107' : '#F44336'
          }} />
          <span className={styles.headerTitle}>D{'\u00F8'} Agent</span>
          <span className={styles.headerMode} style={{
            color: mode === 'auto' ? '#AF87FF' : mode === 'plan' ? '#489696' : '#888'
          }}>
            {mode}
          </span>
        </div>
        <div className={styles.headerRight}>
          {/* Context bar */}
          <div className={styles.contextBar} title={`Context: ${(contextPct * 100).toFixed(0)}%`}>
            <div className={styles.contextFill} style={{ width: `${contextPct * 100}%`, background: ctxColor }} />
          </div>
          <span className={styles.headerCost}>${totalCost.toFixed(2)}</span>
          {onClose && (
            <button className={styles.closeBtn} onClick={onClose} title="Close chat">
              {'\u2715'}
            </button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className={styles.messagesContainer}>
        {/* Welcome */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
          <div style={{ width: 36, height: 36, borderRadius: '50%', background: 'rgba(139,127,240,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: CLAUDE_INDIGO, fontSize: 18, fontWeight: 700 }}>
            D
          </div>
          <div>
            <div style={{ color: '#888', fontSize: '0.82em' }}>D{'\u00F8'} Agent v2.0</div>
            <div style={{ color: '#444', fontSize: '0.75em' }}>AI music production assistant</div>
          </div>
        </div>

        {wsState === 'connecting' && messages.length === 0 && (
          <div style={{ color: '#444', fontSize: '0.82em' }}>connecting{'\u2026'}</div>
        )}

        {visibleMessages.map(msg => {
          if (msg.role === 'system') return <div key={msg.id}><SystemMsg text={msg.text} /></div>;
          if (msg.role === 'user') return <div key={msg.id}><UserMsg text={msg.text} /></div>;
          return (
            <div key={msg.id} style={{ paddingLeft: 2 }}>
              <AssistantMsg msg={msg} frame={frame} />
            </div>
          );
        })}

        {/* Todos */}
        <TodoPanel todos={todos} />

        {/* Permission dialog */}
        {pendingPermission && (
          <PermissionDialog request={pendingPermission} onRespond={respondPermission} />
        )}

        {/* Question dialog */}
        {pendingQuestion && (
          <QuestionDialog request={pendingQuestion} onRespond={respondQuestion} />
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className={styles.inputContainer}>
        {/* Slash command palette */}
        {showCommands && filteredCommands.length > 0 && (
          <div className={styles.commandPalette}>
            {filteredCommands.map((cmd, i) => (
              <div
                key={cmd.name}
                className={`${styles.commandItem} ${i === selectedCmdIdx ? styles.commandItemSelected : ''}`}
                onClick={() => { cmd.action(''); setInput(''); setShowCommands(false); }}
                onMouseEnter={() => setSelectedCmdIdx(i)}
              >
                <span style={{ color: CLAUDE_INDIGO }}>{cmd.name}</span>
                <span style={{ color: '#666', fontSize: '0.85em', marginLeft: 8 }}>{cmd.desc}</span>
              </div>
            ))}
          </div>
        )}

        <div className={styles.inputWrapper} style={busy ? {
          background: `linear-gradient(90deg, ${CLAUDE_INDIGO} 0%, #A097F8 40%, #B8B0FF 50%, #A097F8 60%, ${CLAUDE_INDIGO} 100%)`,
          backgroundSize: '200% 100%',
          animation: 'shimmer 1.8s linear infinite',
        } : {}}>
          <div className={styles.inputInner}>
            <span style={{ color: CLAUDE_INDIGO, flexShrink: 0, paddingBottom: 1, fontSize: '0.9rem' }}>
              {'\u203A'}
            </span>
            <textarea
              ref={textareaRef}
              className={styles.input}
              value={input}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              placeholder={
                wsState !== 'open' ? 'connecting\u2026'
                : busy ? 'working\u2026'
                : mode === 'plan' ? 'plan mode\u2026'
                : mode === 'auto' ? 'auto mode\u2026'
                : 'message d\u00F8\u2026'
              }
              disabled={busy || wsState !== 'open'}
              rows={1}
            />
            <button
              className={styles.sendBtn}
              onClick={handleSend}
              disabled={!input.trim() || busy || wsState !== 'open'}
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" style={{ width: 14, height: 14 }}>
                <path d="M3.105 2.289a.75.75 0 00-.826.95l1.903 6.557H10.5a.75.75 0 010 1.5H4.182l-1.903 6.557a.75.75 0 00.826.95 28.896 28.896 0 0015.293-7.154.75.75 0 000-1.115A28.897 28.897 0 003.105 2.289z" />
              </svg>
            </button>
          </div>
        </div>
        <div style={{ textAlign: 'center', fontSize: '0.6rem', color: '#505050', marginTop: 3 }}>
          Enter to send {'\u2219'} Shift+Enter newline {'\u2219'} / commands
        </div>
      </div>
    </div>
  );
};

export default ChatWindow;
