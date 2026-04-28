import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import { checkVisionRateLimit, rateLimitHeaders } from '@/lib/ratelimit';

/**
 * Server-side proxy to the Modal vision endpoint
 * (arlo--doseedo-chatbot-qwenchatbot-vision.modal.run).
 *
 * Mirrors app/api/chat/route.ts: keeps VLLM_GATE_TOKEN out of the browser
 * bundle, gates POST behind a Clerk session. Accepts the same body shape
 * the Modal FastAPI expects —
 *   POST /api/vision { image_base64, prompt, task?, object? }
 * — forwards verbatim. GET is a passthrough health probe (unauthenticated
 * so uptime checks keep working).
 *
 * Vision responses are always JSON (Moondream doesn't stream) so no SSE
 * plumbing needed — simpler than the chat proxy.
 */

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const VISION_ORIGIN =
  process.env.VISION_ORIGIN ||
  'https://arlo--doseedo-chatbot-qwenchatbot-vision.modal.run';

// Bigger body allowance: a base64-encoded 1440×900 PNG can easily clear
// 1MB. Next.js's default 4MB route-handler body limit is fine, but we
// explicitly document the upper bound for reviewers.
export const maxDuration = 300; // Moondream query is ~1-3s warm, plus cold start budget

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
              service: 'vision',
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
            service: 'vision',
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
    const rl = await checkVisionRateLimit(userId);
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

  const body = await req.text();

  const upstream = await fetchWithRetry(`${VISION_ORIGIN}/v1/vision/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body,
    // @ts-expect-error — duplex is required by undici
    duplex: 'half',
  });

  const contentType = upstream.headers.get('content-type') || 'application/json';
  const text = await upstream.text();
  return new NextResponse(text, {
    status: upstream.status,
    headers: {
      'content-type': contentType,
      'cache-control': 'no-store',
    },
  });
}

export async function GET() {
  // Public health passthrough — no auth, maps to /health on the vision
  // app. Reports { vision_ready, model } once Moondream has loaded on GPU.
  try {
    const r = await fetch(`${VISION_ORIGIN}/health`);
    const body = await r.text();
    return new NextResponse(body, {
      status: r.status,
      headers: {
        'content-type': r.headers.get('content-type') || 'application/json',
      },
    });
  } catch (err) {
    return NextResponse.json(
      { ok: false, error: String(err), origin: VISION_ORIGIN },
      { status: 502 },
    );
  }
}
