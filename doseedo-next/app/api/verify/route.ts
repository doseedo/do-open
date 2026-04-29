import { NextRequest, NextResponse } from 'next/server';
import { checkVerifyRateLimit, rateLimitHeaders } from '@/lib/ratelimit';

/**
 * Server-side proxy for the public /verify page.
 *
 *   POST  multipart/form-data { file: <audio> }
 *   →     200 JSON
 *           {
 *             status: 'verified' | 'not_found',
 *             confidence: number,
 *             // when status === 'verified':
 *             generationId, generatedAt, model, tier,
 *             attribution, wallet|null,
 *             attestationHash, attestationTx, polygonScanUrl,
 *             watermarkConfidence
 *           }
 *
 * Pipeline:
 *   1. IP-rate-limit (no Clerk gate — /verify is a public utility).
 *   2. Stream the upload to the Modal watermark detector
 *      (modal/modal_watermark.py).
 *   3. If a watermark seed is returned, GET it from the Fly auth-service
 *      attestation registry (auth-service/app/routers/watermark_attestations.py).
 *   4. Fold detector + attestation into a single response shape.
 *
 * Why we hit auth-service rather than Neon directly: auth-service owns
 * every DB table in `bassify`. The Stemphonic watermark hook also
 * registers attestations through auth-service — this endpoint mirrors
 * the same boundary so /verify can never see a partial-write state and
 * GDPR redactions (auth-service-side) propagate to /verify for free.
 */

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 60;

const WATERMARK_ORIGIN =
  process.env.WATERMARK_ORIGIN ||
  'https://arlo--doseedo-watermark-watermark-asgi.modal.run';

const AUTH_SERVICE_URL =
  (process.env.NEXT_PUBLIC_AUTH_ORIGIN || 'https://doseedo-api.fly.dev').replace(/\/$/, '');

const POLYGON_SCAN_BASE =
  process.env.POLYGON_SCAN_BASE || 'https://polygonscan.com/tx/';

// 100 MB cap. Reject a slow uploader before we burn maxDuration on the
// stream. Modal would accept more; we don't want to pay for that I/O.
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
  seed: string;
  generation_id: string;
  user_id: number | null;
  tier: string;
  model_version: string;
  wallet: string | null;
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

async function lookupAttestation(seed: string): Promise<
  { row: WatermarkAttestationOut } | { redacted: true } | null
> {
  try {
    const r = await fetch(
      `${AUTH_SERVICE_URL}/api/provenance/watermark/${encodeURIComponent(seed)}`,
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
          seed_prefix: seed.slice(0, 8),
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
    seed: string | null;
    confidence: number;
    duration_sec: number;
    scanned_at: string;
  };

  if (!detect.found || !detect.seed) {
    return NextResponse.json({
      status: 'not_found' as const,
      confidence: detect.confidence,
      duration_sec: detect.duration_sec,
      scanned_at: detect.scanned_at,
    });
  }

  const lookup = await lookupAttestation(detect.seed);

  if (lookup && 'redacted' in lookup) {
    return NextResponse.json({
      status: 'verified' as const,
      generationId: 'redacted',
      generatedAt: detect.scanned_at,
      model: 'redacted',
      tier: 'redacted',
      attribution: 'doseedo (record redacted at user request)',
      wallet: null,
      attestationHash: '0x' + detect.seed,
      attestationTx: null,
      polygonScanUrl: null,
      watermarkConfidence: detect.confidence,
    });
  }

  if (!lookup) {
    // We saw a watermark we don't have on file. Either the registration
    // hop failed at gen-time or the indexer is lagging. Return verified
    // with limited fields so the user still gets a useful answer — the
    // watermark itself proves origin.
    return NextResponse.json({
      status: 'verified' as const,
      generationId: 'unindexed',
      generatedAt: detect.scanned_at,
      model: 'unknown',
      tier: 'unknown',
      attribution: 'doseedo (pre-registry or indexer lag)',
      wallet: null,
      attestationHash: '0x' + detect.seed,
      attestationTx: null,
      polygonScanUrl: null,
      watermarkConfidence: detect.confidence,
    });
  }

  const att = lookup.row;
  return NextResponse.json({
    status: 'verified' as const,
    generationId: att.generation_id,
    generatedAt: att.created_at,
    model: att.model_version,
    tier: att.tier,
    attribution: att.user_id ? `doseedo user · ${att.user_id}` : 'doseedo anonymous',
    wallet: att.wallet,
    attestationHash: '0x' + att.record_hash,
    attestationTx: att.polygon_tx_hash,
    polygonScanUrl: att.polygon_tx_hash ? `${POLYGON_SCAN_BASE}${att.polygon_tx_hash}` : null,
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
