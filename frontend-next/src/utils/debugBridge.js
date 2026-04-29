/**
 * Studio debug bridge — captures console output, network requests,
 * redux actions, unhandled errors, and ships them to /api/debug-log
 * so the assistant can tail the session without copy-pasting.
 *
 * Enable per-session via:
 *   - URL: append `?debug=1` (sticky — persisted to localStorage)
 *   - Console: `localStorage.doseedoDebugLog = 'on'` then reload
 *   - Console: `window.__doseedoDebug.enable()` / `.disable()` at runtime
 *
 * Captures (when enabled):
 *   console.log / warn / error / info
 *   window.fetch requests (URL, method, status, duration)
 *   window.onerror + unhandledrejection
 *   optional: AppContext dispatches via installRedux hook
 *
 * Batches in memory, flushes every 1 s or 25 entries (whichever first),
 * plus on beforeunload. Bounded memory — drops oldest if buffer > 200.
 */

const BUFFER_CAP = 200;
const FLUSH_INTERVAL_MS = 1000;
const FLUSH_SIZE = 25;
const ENDPOINT = '/api/debug-log';
const STORAGE_KEY = 'doseedoDebugLog';

let installed = false;
let enabled = false;
const buffer = [];
let flushTimer = null;
let sessionId = null;
const originals = {};

function makeSessionId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function serializeArg(a) {
  if (a === null || a === undefined) return a;
  const t = typeof a;
  if (t === 'string' || t === 'number' || t === 'boolean') return a;
  if (a instanceof Error) return { _err: a.name, message: a.message, stack: a.stack };
  try {
    // Shallow stringify — avoid giant DOM/AudioBuffer dumps.
    const s = JSON.stringify(a, (_k, v) => {
      if (v instanceof AudioBuffer) return `[AudioBuffer ${v.duration.toFixed(2)}s × ${v.numberOfChannels}ch]`;
      if (v instanceof ArrayBuffer) return `[ArrayBuffer ${v.byteLength}B]`;
      if (v instanceof Element) return `[${v.tagName}#${v.id || ''}]`;
      if (typeof v === 'function') return '[fn]';
      return v;
    });
    return s.length > 2000 ? s.slice(0, 2000) + '…[truncated]' : JSON.parse(s);
  } catch {
    return String(a);
  }
}

function push(entry) {
  if (!enabled) return;
  buffer.push({ t: Date.now(), sid: sessionId, ...entry });
  if (buffer.length > BUFFER_CAP) buffer.splice(0, buffer.length - BUFFER_CAP);
  if (buffer.length >= FLUSH_SIZE) flush();
  else if (!flushTimer) flushTimer = setTimeout(flush, FLUSH_INTERVAL_MS);
}

async function flush() {
  if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
  if (buffer.length === 0) return;
  const entries = buffer.splice(0, buffer.length);
  try {
    await fetch(ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ entries }),
      keepalive: true,
    });
  } catch {
    // drop on failure — no retry, no infinite loop
  }
}

function hookConsole() {
  ['log', 'info', 'warn', 'error'].forEach((level) => {
    originals[level] = console[level].bind(console);
    console[level] = (...args) => {
      originals[level](...args);
      push({ kind: 'console', level, args: args.map(serializeArg) });
    };
  });
}

function unhookConsole() {
  for (const level of Object.keys(originals)) {
    console[level] = originals[level];
  }
}

function hookFetch() {
  if (typeof window === 'undefined' || window.__doseedoDebugFetchPatched) return;
  window.__doseedoDebugFetchPatched = true;
  const orig = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const t0 = performance.now();
    const url = typeof args[0] === 'string' ? args[0] : args[0]?.url || '';
    const method = (args[1]?.method || (args[0] instanceof Request ? args[0].method : 'GET')).toUpperCase();
    // Don't log our own debug sink — avoid recursion.
    const isSink = url.endsWith(ENDPOINT) || url.includes('/api/debug-log');
    try {
      const res = await orig(...args);
      if (!isSink) {
        push({ kind: 'fetch', method, url: url.slice(0, 300), status: res.status, ms: Math.round(performance.now() - t0) });
      }
      return res;
    } catch (err) {
      if (!isSink) {
        push({ kind: 'fetch', method, url: url.slice(0, 300), error: String(err), ms: Math.round(performance.now() - t0) });
      }
      throw err;
    }
  };
}

function hookErrors() {
  if (typeof window === 'undefined') return;
  window.addEventListener('error', (e) => {
    push({ kind: 'error', msg: e.message, source: e.filename, line: e.lineno, col: e.colno, stack: e.error?.stack });
  });
  window.addEventListener('unhandledrejection', (e) => {
    push({ kind: 'unhandledrejection', reason: serializeArg(e.reason) });
  });
  window.addEventListener('beforeunload', () => { flush(); });
}

function hookGlobalBridge() {
  if (typeof window === 'undefined') return;
  window.__doseedoDebug = {
    enable() { localStorage.setItem(STORAGE_KEY, 'on'); enabled = true; flush(); },
    disable() { localStorage.removeItem(STORAGE_KEY); enabled = false; flush(); },
    flush,
    status: () => ({ enabled, buffered: buffer.length, sessionId }),
    mark: (label, data) => push({ kind: 'mark', label, data: serializeArg(data) }),
  };
}

export function installDebugBridge() {
  if (installed) return;
  installed = true;
  if (typeof window === 'undefined') return;

  // Auto-enable on localhost/dev so every /studio session captures logs
  // without the user having to remember ?debug=1. Opt-out via ?debug=0.
  // In production we stay dormant unless explicitly opted in.
  const isLocalHost = (() => {
    try {
      const h = location.hostname;
      return h === 'localhost' || h === '127.0.0.1' || h.endsWith('.local');
    } catch { return false; }
  })();

  try {
    const params = new URLSearchParams(window.location.search);
    if (params.get('debug') === '1') localStorage.setItem(STORAGE_KEY, 'on');
    if (params.get('debug') === '0') localStorage.setItem(STORAGE_KEY, 'off');
    const stored = localStorage.getItem(STORAGE_KEY);
    // Precedence: explicit 'off' wins on any host; 'on' wins; else localhost
    // auto-enables, prod stays off.
    if (stored === 'off') enabled = false;
    else if (stored === 'on') enabled = true;
    else enabled = isLocalHost;
  } catch {
    enabled = isLocalHost;
  }

  if (!enabled) {
    hookGlobalBridge();
    return;
  }

  sessionId = makeSessionId();
  hookConsole();
  hookFetch();
  hookErrors();
  hookGlobalBridge();
  push({ kind: 'session', msg: 'debugBridge installed', ua: navigator.userAgent, url: location.href });
  // Loud console banner so it's obvious the bridge is active.
  try {
    originals.log?.('%c[doseedo-debug] bridge active — /tmp/doseedo-studio-debug.jsonl', 'color:#8B7FF0;font-weight:bold');
  } catch {}
}

export function uninstallDebugBridge() {
  if (!installed) return;
  unhookConsole();
  enabled = false;
  installed = false;
}
