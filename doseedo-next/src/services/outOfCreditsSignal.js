/**
 * Broadcast + subscribe to "user hit daily generation cap" (HTTP 429 from
 * the Modal generation gate). The API layer dispatches the event with
 * whatever the auth-service returned (`resets_at` + `remaining`); the
 * StudioDev OutOfCreditsModal listens for it and renders a themed popup.
 *
 * We use a CustomEvent on window instead of wiring through React context
 * so any fetch site can fire it without threading a dispatch through ten
 * layers of props. The modal is mounted once in StudioDev.
 */
const EVENT = 'doseedo:out-of-credits';

export function signalOutOfCredits({ resetsAt, remaining } = {}) {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(new CustomEvent(EVENT, {
    detail: { resetsAt: resetsAt || null, remaining: remaining ?? 0 },
  }));
}

export function onOutOfCredits(callback) {
  if (typeof window === 'undefined') return () => {};
  const handler = (e) => callback(e.detail || {});
  window.addEventListener(EVENT, handler);
  return () => window.removeEventListener(EVENT, handler);
}

/**
 * Inspect a fetch Response for the gate's 429 envelope and, if present,
 * broadcast the signal. Returns true if the response was a gate 429.
 * Caller should still throw/short-circuit — this only surfaces the UI.
 */
export async function handleMaybeCreditGate(response) {
  if (response.status !== 429) return false;
  try {
    const body = await response.clone().json();
    signalOutOfCredits({
      resetsAt: body?.resets_at,
      remaining: body?.remaining,
    });
  } catch (_) {
    signalOutOfCredits({});
  }
  return true;
}
