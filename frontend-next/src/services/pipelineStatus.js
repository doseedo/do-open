/*
 * pipelineStatus — lightweight event bus for the WebGPU/backend upload
 * pipeline. Services (basicPitchOnnx, latentPitch, sem4Decoder, the
 * StudioDev ingest flow, etc.) push events via logPipeline(); the
 * PipelineStatus component subscribes and renders a live status bar at
 * the bottom of the right sidebar.
 *
 * Keep the payload small — this is a UX feed, not a debug log. Use the
 * `stage` field as a short tag (e.g. "separate", "latentPitch",
 * "basicPitch", "chords"); `msg` is a human-readable one-liner.
 */

const listeners = new Set();
const events = [];
const MAX_EVENTS = 80;

function notify(evt) {
  for (const l of listeners) {
    try { l(evt, events); } catch (_) { /* listener crash doesn't kill others */ }
  }
}

/**
 * Emit a pipeline event.
 *
 * @param {string} stage  short tag, e.g. 'basicPitch' / 'separate' / 'chords'
 * @param {string} msg    one-line status message
 * @param {'info'|'ok'|'warn'|'error'} [kind='info']
 */
export function logPipeline(stage, msg, kind = 'info') {
  const e = { t: Date.now(), stage, msg, kind };
  events.push(e);
  if (events.length > MAX_EVENTS) events.shift();
  // Mirror to console so the full log is still visible in devtools.
  const tag = `[${stage}]`;
  if (kind === 'error') console.error(tag, msg);
  else if (kind === 'warn') console.warn(tag, msg);
  else console.log(tag, msg);
  notify(e);
}

/**
 * Subscribe to pipeline events. Callback receives (event, allEvents).
 * Fires once immediately with (null, currentEvents) so the component
 * can render its initial state without waiting for a new event.
 * Returns an unsubscribe function.
 */
export function subscribe(fn) {
  listeners.add(fn);
  try { fn(null, events); } catch (_) {}
  return () => listeners.delete(fn);
}

/** Clear the event log (new upload, session reset). */
export function clearPipelineLog() {
  events.length = 0;
  notify(null);
}

export function getPipelineLog() {
  return events.slice();
}
