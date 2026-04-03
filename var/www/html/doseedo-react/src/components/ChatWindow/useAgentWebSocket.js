/**
 * useAgentWebSocket — WebSocket hook for the agent chat system.
 * Connects to chat_server.py (same protocol as desktop app).
 *
 * For the web studio, the backend runs on the same server at /_chat/ws
 * (proxied by nginx to the chat_server.py FastAPI WebSocket endpoint).
 *
 * Falls back to REST streaming (/_chat/api/chat-stream) if WebSocket
 * connection fails, for environments that don't support WS.
 */

import { useState, useRef, useEffect, useCallback } from 'react';

// ── Determine WebSocket URL ──
function getWsUrl() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/_chat/ws`;
}

// ── Message ID generator ──
let _msgId = 0;
function nextId() { return `msg_${++_msgId}_${Date.now()}`; }

/**
 * Hook: manages WebSocket connection, message state, and agent protocol.
 * @param {Object} dawState - current DAW state from AppContext
 */
export function useAgentWebSocket(dawState) {
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [wsState, setWsState] = useState('connecting'); // 'connecting' | 'open' | 'closed'
  const [mode, setModeState] = useState('default');
  const [model, setModelState] = useState('claude-sonnet-4-20250514');
  const [contextPct, setContextPct] = useState(0);
  const [inputTokens, setInputTokens] = useState(0);
  const [outputTokens, setOutputTokens] = useState(0);
  const [totalCost, setTotalCost] = useState(0);
  const [pendingPermission, setPendingPermission] = useState(null);
  const [pendingQuestion, setPendingQuestion] = useState(null);
  const [todos, setTodos] = useState([]);

  const wsRef = useRef(null);
  const reconnectRef = useRef(null);
  const currentMsgRef = useRef(null); // track the current streaming assistant message

  // ── Send raw JSON over WS ──
  const sendRaw = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  // ── Send chat message ──
  const sendChat = useCallback((text) => {
    const userMsg = { id: nextId(), role: 'user', text };
    setMessages(prev => [...prev, userMsg]);
    setBusy(true);
    sendRaw({ type: 'chat', text });
  }, [sendRaw]);

  // ── Permission response ──
  const respondPermission = useCallback((approved) => {
    if (pendingPermission) {
      sendRaw({
        type: 'permission_response',
        request_id: pendingPermission.requestId,
        approved,
      });
      setPendingPermission(null);
    }
  }, [pendingPermission, sendRaw]);

  // ── Question response ──
  const respondQuestion = useCallback((answer) => {
    if (pendingQuestion) {
      sendRaw({
        type: 'question_response',
        request_id: pendingQuestion.requestId,
        answer,
      });
      setPendingQuestion(null);
    }
  }, [pendingQuestion, sendRaw]);

  // ── Mode/model setters ──
  const setMode = useCallback((m) => {
    sendRaw({ type: 'set_mode', mode: m });
  }, [sendRaw]);

  const setModel = useCallback((m) => {
    sendRaw({ type: 'set_model', model: m });
  }, [sendRaw]);

  // ── Helper: update current streaming assistant message ──
  const updateCurrentMsg = useCallback((updater) => {
    setMessages(prev => {
      const last = prev[prev.length - 1];
      if (last && last.role === 'assistant' && last.streaming) {
        const updated = { ...last, ...updater(last) };
        return [...prev.slice(0, -1), updated];
      }
      return prev;
    });
  }, []);

  // ── Ensure a streaming assistant message exists ──
  const ensureStreamingMsg = useCallback(() => {
    setMessages(prev => {
      const last = prev[prev.length - 1];
      if (last && last.role === 'assistant' && last.streaming) return prev;
      // Create new streaming message
      const msg = {
        id: nextId(),
        role: 'assistant',
        text: '',
        streaming: true,
        tools: [],
        thinkingStatus: null,
        verb: 'Generating',
      };
      return [...prev, msg];
    });
  }, []);

  // ── WebSocket message handler ──
  const handleMessage = useCallback((event) => {
    let data;
    try { data = JSON.parse(event.data); } catch { return; }

    switch (data.type) {
      case 'settings':
        if (data.model) setModelState(data.model);
        if (data.mode) setModeState(data.mode);
        break;

      case 'status':
        // Project status — add as system message
        if (data.summary) {
          setMessages(prev => [...prev, { id: nextId(), role: 'system', text: data.summary }]);
        }
        break;

      case 'system':
        setMessages(prev => [...prev, { id: nextId(), role: 'system', text: data.text }]);
        break;

      case 'error':
        setMessages(prev => [...prev, { id: nextId(), role: 'system', text: `Error: ${data.text}` }]);
        setBusy(false);
        break;

      case 'thinking_start':
        ensureStreamingMsg();
        updateCurrentMsg(() => ({ thinkingStatus: 'thinking' }));
        break;

      case 'thinking_done':
        updateCurrentMsg(() => ({ thinkingStatus: data.elapsed_ms || 0 }));
        break;

      case 'text_delta':
        ensureStreamingMsg();
        updateCurrentMsg(msg => ({ text: (msg.text || '') + data.text }));
        break;

      case 'tool_start':
        ensureStreamingMsg();
        updateCurrentMsg(msg => ({
          tools: [...msg.tools, {
            id: data.id,
            name: data.name,
            status: 'running',
            startMs: Date.now(),
          }],
          verb: data.name?.replace(/_/g, ' '),
        }));
        break;

      case 'tool_result':
        updateCurrentMsg(msg => ({
          tools: msg.tools.map(t =>
            t.id === data.id
              ? { ...t, status: data.ok ? 'ok' : 'error', result: data.text, endMs: Date.now() }
              : t
          ),
        }));
        break;

      case 'turn_done':
        // Finalize the streaming message
        updateCurrentMsg(msg => ({ streaming: false, verb: null }));
        setBusy(false);
        if (data.input_tokens) setInputTokens(prev => prev + data.input_tokens);
        if (data.output_tokens) setOutputTokens(prev => prev + data.output_tokens);
        if (data.total_cost != null) setTotalCost(data.total_cost);
        if (data.context_pct != null) setContextPct(data.context_pct);
        break;

      case 'permission_request':
        setPendingPermission({
          requestId: data.request_id,
          toolName: data.tool_name,
          toolInput: data.tool_input,
          description: data.description,
        });
        break;

      case 'question_request':
        setPendingQuestion({
          requestId: data.request_id,
          question: data.question,
        });
        break;

      case 'todos_updated':
        setTodos(data.todos || []);
        break;

      case 'mode_changed':
        setModeState(data.mode);
        break;

      case 'model_changed':
        setModelState(data.model);
        break;

      case 'sessions_list':
      case 'session_opened':
        // These are handled by the parent app, not the chat window
        break;

      case 'compact_done':
        setMessages(prev => [
          ...prev,
          { id: nextId(), role: 'system', text: 'History compacted.' },
        ]);
        break;

      case 'open_url':
        if (data.url) window.open(data.url, '_blank');
        break;

      default:
        // Unknown message type — ignore
        break;
    }
  }, [ensureStreamingMsg, updateCurrentMsg]);

  // ── WebSocket connection management ──
  useEffect(() => {
    let ws;
    let alive = true;

    function connect() {
      if (!alive) return;
      setWsState('connecting');

      const url = getWsUrl();
      ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!alive) return;
        setWsState('open');
        // Clear any reconnect timer
        if (reconnectRef.current) {
          clearTimeout(reconnectRef.current);
          reconnectRef.current = null;
        }
        // Send DAW context
        if (dawState?.bpm) {
          ws.send(JSON.stringify({
            type: 'daw_context',
            bpm: dawState.bpm,
            key: dawState.generationParams?.key || 'C',
            trackCount: dawState.buses?.reduce((t, b) => t + b.tracks.length, 0) || 0,
          }));
        }
      };

      ws.onmessage = handleMessage;

      ws.onclose = () => {
        if (!alive) return;
        setWsState('closed');
        wsRef.current = null;
        // Auto-reconnect after 3s
        reconnectRef.current = setTimeout(connect, 3000);
      };

      ws.onerror = () => {
        // onclose will fire after onerror
      };
    }

    connect();

    return () => {
      alive = false;
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      if (ws) ws.close();
    };
  }, [handleMessage]); // reconnect if handler ref changes (shouldn't with useCallback)

  return {
    messages,
    busy,
    wsState,
    mode,
    model,
    contextPct,
    inputTokens,
    outputTokens,
    totalCost,
    pendingPermission,
    pendingQuestion,
    todos,
    sendChat,
    sendRaw,
    respondPermission,
    respondQuestion,
    setMode,
    setModel,
  };
}
