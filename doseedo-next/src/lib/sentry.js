/**
 * Sentry init for Doseedo Studio.
 *
 * Session Replay is deliberately OFF for launch — conflicts with the
 * "your audio stays on your device" copy, and the signal-to-noise on
 * the DAW canvas is bad for a replay player anyway. Revisit in ~6
 * months if we start seeing unreproducible user bugs.
 *
 * Source maps are uploaded by deploy.sh via sentry-cli, NOT published
 * alongside the bundle in the GCS frontend bucket. Errors are close
 * to useless in prod without source maps, and we don't want to leak
 * un-minified source publicly.
 *
 * DSN is injected via NEXT_PUBLIC_SENTRY_DSN at build time. If missing
 * (e.g., local dev without a DSN), init() no-ops — SDK is loaded but
 * never sends.
 */
import * as Sentry from '@sentry/react';

const DSN = process.env.NEXT_PUBLIC_SENTRY_DSN || '';
const ENV = process.env.NEXT_PUBLIC_SENTRY_ENV
  || (process.env.NODE_ENV === 'production' ? 'production' : 'development');
const RELEASE = process.env.NEXT_PUBLIC_SENTRY_RELEASE
  || process.env.NEXT_PUBLIC_GIT_SHA
  || undefined;

// Headers to drop anywhere they appear in breadcrumbs or captures.
const SCRUB_HEADERS = new Set([
  'authorization', 'cookie', 'x-internal-secret',
  'x-csrf-token', 'x-api-key', 'proxy-authorization', 'set-cookie',
]);

// Case-insensitive match for secret-ish body fields.
const SCRUB_BODY_KEY = /token|jwt|secret|password|api[_-]?key/i;

// Any HTTP span hitting these paths has its body dropped entirely.
// Keep the tag so we can still see that it happened in the Issues list.
const LEGACY_ROUTES = /\/api\/encode-(audio-latent|latents-bulk)\b/;

export function initSentry() {
  if (!DSN) {
    if (ENV !== 'development') {
      // eslint-disable-next-line no-console
      console.warn('[sentry] NEXT_PUBLIC_SENTRY_DSN not set — errors will NOT be captured');
    }
    return;
  }

  Sentry.init({
    dsn: DSN,
    environment: ENV,
    release: RELEASE,
    sendDefaultPii: false, // don't send IP, cookies, etc.
    // Error capture + perf, no replay.
    integrations: [
      Sentry.browserTracingIntegration(),
      Sentry.httpClientIntegration(),
    ],
    tracesSampleRate: ENV === 'production' ? 0.1 : 1.0,
    // Perf samples for common frontend routes.
    tracePropagationTargets: [
      'https://doseedo.com',
      /^\/api\//,
    ],
    beforeSend,
    beforeBreadcrumb,
  });
}

/**
 * The event-level scrubber. Catches anything headed for Sentry that
 * carries an auth token, a secret, or a payload from the legacy
 * client-side-encode routes we 410'd.
 */
function beforeSend(event, hint) {
  try {
    scrubRequest(event.request);
    scrubException(event);
    scrubExtra(event);
    scrubBreadcrumbs(event.breadcrumbs);
    // If the request URL matches a legacy route, tag the event and
    // strip the body regardless of content — keep the fact that a
    // stale client path was hit, lose the payload so we don't
    // accidentally log audio or latent bytes.
    if (event.request?.url && LEGACY_ROUTES.test(event.request.url)) {
      event.tags = { ...(event.tags || {}), legacy_route_hit: 'true' };
      if (event.request) delete event.request.data;
    }
  } catch (e) {
    // Never throw from beforeSend — if scrubbing fails, send a
    // minimal redacted event instead of dropping altogether so the
    // error itself still surfaces.
    return {
      message: '[sentry scrubber error — event redacted]',
      level: 'error',
      tags: { scrubber_failure: 'true' },
    };
  }
  return event;
}

function beforeBreadcrumb(crumb, hint) {
  try {
    if (crumb.category === 'fetch' || crumb.category === 'xhr') {
      scrubHeaders(crumb.data?.request_headers);
      scrubHeaders(crumb.data?.response_headers);
      const url = crumb.data?.url || '';
      if (LEGACY_ROUTES.test(url)) {
        crumb.data = { ...(crumb.data || {}), legacy_route_hit: true };
      }
    }
  } catch (_) { /* never throw from breadcrumbs */ }
  return crumb;
}

function scrubRequest(req) {
  if (!req) return;
  scrubHeaders(req.headers);
  scrubHeaders(req.cookies);
  req.data = scrubObject(req.data);
  // Query string on a URL can carry tokens too. Not common here but cheap.
  if (typeof req.query_string === 'string' && SCRUB_BODY_KEY.test(req.query_string)) {
    req.query_string = '[scrubbed]';
  }
}

function scrubHeaders(headers) {
  if (!headers || typeof headers !== 'object') return;
  for (const k of Object.keys(headers)) {
    if (SCRUB_HEADERS.has(k.toLowerCase())) headers[k] = '[scrubbed]';
  }
}

function scrubException(event) {
  if (!event.exception?.values) return;
  for (const ex of event.exception.values) {
    if (ex.stacktrace?.frames) {
      for (const f of ex.stacktrace.frames) {
        if (f.vars) f.vars = scrubObject(f.vars);
      }
    }
  }
}

function scrubExtra(event) {
  if (event.extra) event.extra = scrubObject(event.extra);
  if (event.contexts) {
    for (const k of Object.keys(event.contexts)) {
      event.contexts[k] = scrubObject(event.contexts[k]);
    }
  }
}

function scrubBreadcrumbs(crumbs) {
  if (!crumbs) return;
  for (const c of crumbs) {
    if (c.data) c.data = scrubObject(c.data);
  }
}

/**
 * Recursive object/array walk that redacts any key matching the
 * secret regex. Depth-bounded to avoid runaway recursion on circular
 * structures Sentry would otherwise choke on.
 */
function scrubObject(obj, depth = 0) {
  if (depth > 6 || obj == null) return obj;
  if (Array.isArray(obj)) return obj.map(v => scrubObject(v, depth + 1));
  if (typeof obj !== 'object') return obj;
  const out = {};
  for (const [k, v] of Object.entries(obj)) {
    if (SCRUB_BODY_KEY.test(k)) { out[k] = '[scrubbed]'; continue; }
    out[k] = scrubObject(v, depth + 1);
  }
  return out;
}

// Convenience re-exports so callers import from one place.
export { Sentry };
export const captureException = (err, ctx) => {
  if (!DSN) return;
  Sentry.captureException(err, ctx);
};
export const setUser = (userId) => {
  if (!DSN || !userId) return;
  Sentry.setUser({ id: String(userId) });
};
