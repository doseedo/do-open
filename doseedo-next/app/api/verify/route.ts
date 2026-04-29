import { NextRequest, NextResponse } from 'next/server';
import { checkVerifyRateLimit, rateLimitHeaders } from '@/lib/ratelimit';

/**
 * Server-side proxy for the public /verify page.
 *
 *   POST  multipart/form-data { file: <audio> }
 *   →     200 JSON
 *           {
 *             status: 'verified' | 'verified_pending' | 'not_found',
 *             watermarkConfidence: number,
 *             // when status === 'verified' or 'verified_pending':
 *             generationId, generatedAt, model, tier,
 *             attribution, audioSha256, attestationHash,
 *             attestationTx | null, polygonScanUrl | null
 *           }
 *
 * Pipeline:
 *   1. IP rate-limit (no Clerk gate — /verify is public).
 *   2. Stream the upload to the Modal watermark detector. Detector
 *      hashes the bytes with SHA-256 and returns it alongside the
 *      classifier confidence.
 *   3. Look the audio_sha256 up against the auth-service registry.
 *      The detector confidence is a "is this a doseedo file?" signal;
 *      the SHA-256 is the unique identifier into the row.
 *   4. Resolved row that's already on-chain → 'verified' with tx link.
 *      Resolved row but publisher hasn't anchored yet → 'verified_pending'.
 *      No row → 'not_found' (not a doseedo file, or re-encoded copy
 *      whose bytes no longer match the registered hash).
 *
 * Network: POLYGON_NETWORK ('amoy' | 'mainnet') drives the explorer
 * base URL. Set it once on the deployment and the same value flows
 * everywhere (verify, Privacy/Verify pages, StudioDev pill).
 */

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 60;

const WATERMARK_ORIGIN =
  process.env.WATERMARK_ORIGIN ||
  'https://arlo--doseedo-watermark-watermark-asgi.modal.run';

const AUTH_SERVICE_URL =
  (process.env.NEXT_PUBLIC_AUTH_ORIGIN || 'https://doseedo-api.fly.dev').replace(/\/$/, '');

const POLYGON_NETWORK = (process.env.POLYGON_NETWORK || 'mainnet').toLowerCase();
const POLYGON_SCAN_BASE =
  POLYGON_NETWORK === 'amoy'
    ? 'https://amoy.polygonscan.com/tx/'
    : 'https://polygonscan.com/tx/';

const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;

const RETRYABLE_STATUS = new Set([502, 503, 504]);
const MAX_RETRIES = 1;
const BACKOFF_MS = 1500;

async function fetchWithRetry(url: string, init: RequestInit): Promise<Response> {
  let lastErr: unknown = null;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const r = await fetch(url, init);
      if (!RETRYABLE_STATUS.has(r.status) || attempt === MAX_RETRIES) {
        if (attempt === MAX_RETRIES && RETRYABLE_STATUS.has(r.status)) {
          // eslint-disable-next-line no-console
          console.error(
            JSON.stringify({
              tag: 'modal-cold-start-exhausted',
              service: 'verify',
              upstream_status: r.status,
              url,
              attempts: attempt + 1,
            }),
          );
        }
        return r;
      }
      lastErr = new Error(`upstream ${r.status}`);
    } catch (e) {
      lastErr = e;
      if (attempt === MAX_RETRIES) throw e;
    }
    await new Promise(res => setTimeout(res, BACKOFF_MS * (attempt + 1)));
  }
  throw lastErr ?? new Error('fetch failed');
}

type WatermarkAttestationOut = {
  id: string;
  audio_sha256: string;
  seed: string | null;
  generation_id: string;
  user_id: number | null;
  tier: string;
  model_version: string;
  record_hash: string;
  metadata_uri: string | null;
  polygon_status: string | null;
  polygon_tx_hash: string | null;
  polygon_block_number: number | null;
  polygon_published_at: string | null;
  polygon_network: string | null;
  redacted_at: string | null;
  created_at: string;
};

async function lookupAttestation(audioSha256: string): Promise<
  { row: WatermarkAttestationOut } | { redacted: true } | null
> {
  try {
    const r = await fetch(
      `${AUTH_SERVICE_URL}/api/provenance/watermark/${encodeURIComponent(audioSha256)}`,
      {
        headers: { Accept: 'application/json' },
        cache: 'no-store',
      },
    );
    if (r.status === 410) return { redacted: true };
    if (r.status === 404) return null;
    if (!r.ok) {
      // eslint-disable-next-line no-console
      console.error(
        JSON.stringify({
          tag: 'auth-service-lookup-failed',
          service: 'verify',
          status: r.status,
          sha_prefix: audioSha256.slice(0, 12),
        }),
      );
      return null;
    }
    const row = (await r.json()) as WatermarkAttestationOut;
    return { row };
  } catch (e) {
    // eslint-disable-next-line no-console
    console.error(
      JSON.stringify({
        tag: 'auth-service-unreachable',
        service: 'verify',
        error: String(e).slice(0, 200),
      }),
    );
    return null;
  }
}

function clientIp(req: NextRequest): string {
  const xff = req.headers.get('x-forwarded-for');
  if (xff) return xff.split(',')[0].trim();
  return req.headers.get('cf-connecting-ip') || 'anon';
}

function buildPolygonScanUrl(network: string | null, tx: string): string {
  const base = (network || POLYGON_NETWORK) === 'amoy'
    ? 'https://amoy.polygonscan.com/tx/'
    : POLYGON_SCAN_BASE;
  return `${base}${tx}`;
}

export async function POST(req: NextRequest) {
  const ip = clientIp(req);
  const rl = await checkVerifyRateLimit(ip);
  if (!rl.success) {
    return NextResponse.json(
      { error: 'Rate limit exceeded', limit: rl.limit, reset: rl.reset },
      { status: 429, headers: rateLimitHeaders(rl) },
    );
  }

  const apiKey = process.env.VLLM_GATE_TOKEN || process.env.VLLM_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: 'Watermark gate token not configured on server' },
      { status: 500 },
    );
  }

  const contentType = req.headers.get('content-type');
  if (!contentType || !contentType.includes('multipart/form-data')) {
    return NextResponse.json(
      { error: 'Expected multipart/form-data with field "file"' },
      { status: 400 },
    );
  }

  const lenHeader = req.headers.get('content-length');
  if (lenHeader) {
    const len = Number(lenHeader);
    if (Number.isFinite(len) && len > MAX_UPLOAD_BYTES) {
      return NextResponse.json(
        { error: 'Upload too large', max_bytes: MAX_UPLOAD_BYTES, got_bytes: len },
        { status: 413 },
      );
    }
  }

  const upstream = await fetchWithRetry(`${WATERMARK_ORIGIN}/detect`, {
    method: 'POST',
    headers: {
      'content-type': contentType,
      Authorization: `Bearer ${apiKey}`,
    },
    body: req.body,
    // @ts-expect-error — duplex is required by undici when streaming a body
    duplex: 'half',
  });

  if (!upstream.ok) {
    const text = await upstream.text().catch(() => '');
    return NextResponse.json(
      {
        error: 'Detector unreachable',
        upstream_status: upstream.status,
        upstream_body: text.slice(0, 400),
      },
      { status: 502 },
    );
  }

  const detect = (await upstream.json()) as {
    found: boolean;
    confidence: number;
    audio_sha256: string;
    duration_sec: number;
    scanned_at: string;
  };

  // SHA-256 of the uploaded bytes is the unique key. The classifier
  // confidence is reported back to the user but doesn't drive the
  // verified/not-found split — a paid-tier export with no audio-side
  // mark still resolves cleanly via SHA-256, and a re-encoded copy
  // that retained the mark but lost byte-equality won't.
  const lookup = await lookupAttestation(detect.audio_sha256);

  if (lookup && 'redacted' in lookup) {
    return NextResponse.json({
      status: 'verified' as const,
      generationId: 'redacted',
      generatedAt: detect.scanned_at,
      model: 'redacted',
      tier: 'redacted',
      attribution: 'doseedo (record redacted at user request)',
      audioSha256: detect.audio_sha256,
      attestationHash: null,
      attestationTx: null,
      polygonScanUrl: null,
      watermarkConfidence: detect.confidence,
    });
  }

  if (!lookup) {
    return NextResponse.json({
      status: 'not_found' as const,
      watermarkConfidence: detect.confidence,
      duration_sec: detect.duration_sec,
      scanned_at: detect.scanned_at,
    });
  }

  // Narrowed manually: the prior guards eliminated `null` and the
  // `redacted` branch, but the discriminated-union narrowing here is
  // imperfect because `redacted` and `row` aren't on a single discriminator.
  const att = (lookup as { row: WatermarkAttestationOut }).row;
  const anchored = att.polygon_status === 'confirmed' || att.polygon_status === 'final';
  const txUrl = anchored && att.polygon_tx_hash
    ? buildPolygonScanUrl(att.polygon_network, att.polygon_tx_hash)
    : null;

  return NextResponse.json({
    status: anchored ? ('verified' as const) : ('verified_pending' as const),
    generationId: att.generation_id,
    generatedAt: att.created_at,
    model: att.model_version,
    tier: att.tier,
    attribution: att.user_id ? `doseedo user · ${att.user_id}` : 'doseedo anonymous',
    audioSha256: att.audio_sha256,
    attestationHash: '0x' + att.record_hash,
    attestationTx: anchored ? att.polygon_tx_hash : null,
    polygonScanUrl: txUrl,
    polygonStatus: att.polygon_status,
    polygonNetwork: att.polygon_network,
    watermarkConfidence: detect.confidence,
  });
}

export async function GET() {
  try {
    const r = await fetch(`${WATERMARK_ORIGIN}/health`, { cache: 'no-store' });
    const body = await r.text();
    return new NextResponse(body, {
      status: r.status,
      headers: {
        'content-type': r.headers.get('content-type') || 'application/json',
        'cache-control': 'no-store',
      },
    });
  } catch (err) {
    return NextResponse.json(
      { ok: false, error: String(err), origin: WATERMARK_ORIGIN },
      { status: 502, headers: { 'cache-control': 'no-store' } },
    );
  }
}
