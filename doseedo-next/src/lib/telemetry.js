/**
 * Product-telemetry event helper. Fires every event as a Sentry
 * breadcrumb (so it gets attached to any subsequent error capture)
 * AND as a separate info-level Sentry message so we can build queries
 * / issues on specific event names.
 *
 * Named events (these are the ones we actually care about — generic
 * errors go via Sentry.captureException separately):
 *
 *   webgpu.init_failed           {reason, platform}
 *   encoder.unavailable          {reason, platform}
 *   opfs.quota_exceeded          {requested_bytes, quota_bytes}
 *   decode.underrun              {stem, buffer_frames}
 *   generation.started           {route, params}
 *   generation.succeeded         {route, duration_ms}
 *   generation.failed            {route, reason, duration_ms}
 *   upload.rejected_non_latent   {mime_type}
 *
 * Add new events to PRODUCT_EVENTS so we keep the surface explicit —
 * anything else goes through Sentry.captureException or
 * Sentry.captureMessage directly.
 */
import { Sentry } from './sentry';

export const PRODUCT_EVENTS = Object.freeze({
  WEBGPU_INIT_FAILED:         'webgpu.init_failed',
  ENCODER_UNAVAILABLE:        'encoder.unavailable',
  OPFS_QUOTA_EXCEEDED:        'opfs.quota_exceeded',
  DECODE_UNDERRUN:            'decode.underrun',
  GENERATION_STARTED:         'generation.started',
  GENERATION_SUCCEEDED:       'generation.succeeded',
  GENERATION_FAILED:          'generation.failed',
  UPLOAD_REJECTED_NON_LATENT: 'upload.rejected_non_latent',
});

const VALID = new Set(Object.values(PRODUCT_EVENTS));

/**
 * Record a product-telemetry event.
 *
 * @param {string} name   one of PRODUCT_EVENTS values
 * @param {object} props  event-specific data (platform, reason, etc.)
 */
export function trackEvent(name, props = {}) {
  if (!VALID.has(name)) {
    // eslint-disable-next-line no-console
    console.warn(`[telemetry] unknown event ${name} — add to PRODUCT_EVENTS`);
  }
  const safeProps = coerceProps(props);

  // 1) Breadcrumb — attaches to the next error if any.
  try {
    Sentry.addBreadcrumb({
      category: 'product',
      message: name,
      data: safeProps,
      level: levelFor(name),
    });
  } catch (_) { /* sentry not init */ }

  // 2) Info-level message — gives each named event its own issue in
  //    Sentry so we can alert / query on it. Only for error-ish events
  //    to avoid flooding.
  if (isErrorEvent(name)) {
    try {
      Sentry.captureMessage(name, { level: levelFor(name), extra: safeProps });
    } catch (_) { /* sentry not init */ }
  }

  // 3) Local console in dev — removed in prod via tree-shaking below.
  if (process.env.NODE_ENV !== 'production') {
    // eslint-disable-next-line no-console
    console.log(`[telemetry] ${name}`, safeProps);
  }
}

// Events that deserve their own Sentry issue vs. trail-only breadcrumbs.
function isErrorEvent(name) {
  return name === PRODUCT_EVENTS.WEBGPU_INIT_FAILED
      || name === PRODUCT_EVENTS.ENCODER_UNAVAILABLE
      || name === PRODUCT_EVENTS.OPFS_QUOTA_EXCEEDED
      || name === PRODUCT_EVENTS.DECODE_UNDERRUN
      || name === PRODUCT_EVENTS.GENERATION_FAILED
      || name === PRODUCT_EVENTS.UPLOAD_REJECTED_NON_LATENT;
}

function levelFor(name) {
  if (name === PRODUCT_EVENTS.GENERATION_STARTED
   || name === PRODUCT_EVENTS.GENERATION_SUCCEEDED) return 'info';
  return 'warning';
}

/**
 * Coerce props to Sentry-safe shapes: primitives only at leaves,
 * depth cap, string truncation, and typed-array rejection (don't
 * accidentally send audio/latent bytes up as a "param").
 */
function coerceProps(obj, depth = 0) {
  if (obj == null || depth > 4) return obj ?? null;
  if (typeof obj === 'string') return obj.length > 500 ? obj.slice(0, 500) + '…' : obj;
  if (typeof obj === 'number' || typeof obj === 'boolean') return obj;
  if (obj instanceof ArrayBuffer || ArrayBuffer.isView(obj)) return `[${obj.byteLength || obj.length}B binary]`;
  if (Array.isArray(obj)) return obj.slice(0, 50).map(v => coerceProps(v, depth + 1));
  if (typeof obj === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(obj)) out[k] = coerceProps(v, depth + 1);
    return out;
  }
  return String(obj);
}

/** Platform string for the init-failed events. */
export function platformString() {
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown';
  const gpu = typeof navigator !== 'undefined' && 'gpu' in navigator ? 'webgpu:yes' : 'webgpu:no';
  return `${ua.slice(0, 120)} | ${gpu}`;
}
