import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import { checkVideoScoreRateLimit, rateLimitHeaders } from '@/lib/ratelimit';

/**
 * Server-side proxy to the Modal video-scoring endpoint
 * (modal/modal_video_scoring.py — `arlo--doseedo-video-scoring-…score.modal.run`).
 *
 * Mirrors app/api/vision/route.ts: Clerk-gated, rate-limited, retries
 * 502/503/504 once for cold starts. Differs in two ways:
 *
 *  1. Body is multipart/form-data (the video upload). We pass the raw
 *     request stream straight through — buffering it in Node memory
 *     would double latency and risk OOM on Vercel's smaller tiers.
 *  2. Response is **Server-Sent Events** (`text/event-stream`). We pipe
 *     `upstream.body` through verbatim so the studio drop zone sees
 *     progress events (`shots`, `scene`, `scene_done`, `midi`, `done`)
 *     as the Modal worker emits them.
 *
 * Pre-flight Content-Length check rejects oversized uploads before we
 * burn the round-trip and Modal CPU minutes.
 */

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const VIDEO_SCORE_ORIGIN =
  process.env.VIDEO_SCORE_ORIGIN ||
  'https://arlo--doseedo-video-scoring-videoscoring-score.modal.run';

// Scene detection on a 2-min 1080p clip can run ~30-60s. Cold start adds
// ~5-8 s with memory snapshots, more without. 300s is the Vercel cap; take
// all of it.
export const maxDuration = 300;

// Hard upload cap. Keeps a slow client from blowing the maxDuration
// budget streaming hundreds of MB. Modal's request body parser would
// happily accept more, but we don't want to pay for the I/O.
const MAX_UPLOAD_BYTES = 150 * 1024 * 1024; // 150 MB

const RETRYABLE_STATUS = new Set([502, 503, 504]);
const MAX_RETRIES = 1;
const BACKOFF_MS = 1500;

async function fetchWithRetry(url: string, init: RequestInit): Promise<Response> {
  let lastErr: unknown = null;
  let lastStatus = 0;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const r = await fetch(url, init);
      if (!RETRYABLE_STATUS.has(r.status) || attempt === MAX_RETRIES) {
        if (attempt === MAX_RETRIES && RETRYABLE_STATUS.has(r.status)) {
          // eslint-disable-next-line no-console
          console.error(
            JSON.stringify({
              tag: 'modal-cold-start-exhausted',
              service: 'video-score',
              upstream_status: r.status,
              url,
              attempts: attempt + 1,
            }),
          );
        }
        return r;
      }
      lastStatus = r.status;
      lastErr = new Error(`upstream ${r.status}`);
    } catch (e) {
      lastErr = e;
      if (attempt === MAX_RETRIES) {
        // eslint-disable-next-line no-console
        console.error(
          JSON.stringify({
            tag: 'modal-cold-start-exhausted',
            service: 'video-score',
            error: String(e).slice(0, 200),
            last_status: lastStatus,
            url,
            attempts: attempt + 1,
          }),
        );
        throw e;
      }
    }
    await new Promise(res => setTimeout(res, BACKOFF_MS * (attempt + 1)));
  }
  throw lastErr ?? new Error('fetch failed');
}

export async function POST(req: NextRequest) {
  let userId: string | null = null;
  if (process.env.CLERK_SECRET_KEY) {
    const a = await auth();
    userId = a.userId;
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
  }

  if (userId) {
    const rl = await checkVideoScoreRateLimit(userId);
    if (!rl.success) {
      return NextResponse.json(
        { error: 'Rate limit exceeded', limit: rl.limit, reset: rl.reset },
        { status: 429, headers: rateLimitHeaders(rl) },
      );
    }
  }

  const apiKey = process.env.VLLM_GATE_TOKEN || process.env.VLLM_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: 'VLLM_GATE_TOKEN not configured on server' },
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

  // Pre-flight size cap. Reject before forwarding.
  const lenHeader = req.headers.get('content-length');
  if (lenHeader) {
    const len = Number(lenHeader);
    if (Number.isFinite(len) && len > MAX_UPLOAD_BYTES) {
      return NextResponse.json(
        {
          error: 'Upload too large',
          max_bytes: MAX_UPLOAD_BYTES,
          got_bytes: len,
        },
        { status: 413 },
      );
    }
  }

  const upstream = await fetchWithRetry(`${VIDEO_SCORE_ORIGIN}/score`, {
    method: 'POST',
    headers: {
      'content-type': contentType,
      Authorization: `Bearer ${apiKey}`,
      Accept: 'text/event-stream',
    },
    body: req.body,
    // @ts-expect-error — duplex is required by undici when streaming a body
    duplex: 'half',
  });

  // Stream the upstream SSE body straight through — do NOT call .text(),
  // which would block until the worker emits the `done` event. We want
  // each event delivered to the browser as it's produced.
  const upstreamCt = upstream.headers.get('content-type') || 'text/event-stream';
  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: {
      'content-type': upstreamCt,
      'cache-control': 'no-store',
      // Ensure intermediate proxies (Vercel CDN / Cloudflare) don't buffer.
      'X-Accel-Buffering': 'no',
    },
  });
}

export async function GET() {
  try {
    const r = await fetch(`${VIDEO_SCORE_ORIGIN}/health`);
    const body = await r.text();
    return new NextResponse(body, {
      status: r.status,
      headers: {
        'content-type': r.headers.get('content-type') || 'application/json',
      },
    });
  } catch (err) {
    return NextResponse.json(
      { ok: false, error: String(err), origin: VIDEO_SCORE_ORIGIN },
      { status: 502 },
    );
  }
}
