import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import { checkChatRateLimit, rateLimitHeaders } from '@/lib/ratelimit';

/**
 * Server-side proxy to the Modal vLLM chatbot
 * (arlo--doseedo-chatbot-qwenchatbot-serve.modal.run).
 *
 * Keeps the gate token out of the browser bundle. Accepts an OpenAI-style
 * chat completion request body and forwards it verbatim, including the
 * `stream: true` case — we pipe the Modal SSE response straight to the
 * client so tokens arrive with the same latency as a direct call.
 *
 * POST is gated by Clerk: a signed-in session is required, otherwise the
 * Modal token is just a free LLM for the internet. GET stays open so
 * uptime probes can hit `/api/chat` for the health passthrough.
 */

export const runtime = 'nodejs';
// Streamed responses must be dynamic, not cached.
export const dynamic = 'force-dynamic';

const CHATBOT_ORIGIN =
  process.env.CHATBOT_ORIGIN ||
  'https://arlo--doseedo-chatbot-qwenchatbot-serve.modal.run';

// Bounded retry on cold-start / transient upstream failures. vLLM cold
// start is ~80s; if Modal returns 502/503/504 or the connection drops
// mid-handshake, retry once with a short backoff before bubbling up.
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
          // Retry budget exhausted on a transient upstream failure. Tag
          // for log-based alerting (Vercel → Sentry log drain etc.).
          // eslint-disable-next-line no-console
          console.error(
            JSON.stringify({
              tag: 'modal-cold-start-exhausted',
              service: 'chat',
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
            service: 'chat',
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
    const rl = await checkChatRateLimit(userId);
    if (!rl.success) {
      return NextResponse.json(
        { error: 'Rate limit exceeded', limit: rl.limit, reset: rl.reset },
        { status: 429, headers: rateLimitHeaders(rl) },
      );
    }
  }

  // VLLM_GATE_TOKEN is the canonical name (matches the Fly auth-service
  // secret). VLLM_API_KEY is kept as a fallback for older deployments.
  const apiKey = process.env.VLLM_GATE_TOKEN || process.env.VLLM_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: 'VLLM_GATE_TOKEN not configured on server' },
      { status: 500 },
    );
  }

  const body = await req.text();

  const upstream = await fetchWithRetry(`${CHATBOT_ORIGIN}/v1/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body,
    // vLLM cold start can take ~80s; don't let the platform time out early.
    // @ts-expect-error — duplex is required by undici for streaming bodies
    duplex: 'half',
  });

  const contentType = upstream.headers.get('content-type') || '';
  const isStream = contentType.includes('text/event-stream');

  if (!upstream.ok && !isStream) {
    const text = await upstream.text();
    return new NextResponse(text, {
      status: upstream.status,
      headers: { 'content-type': contentType || 'application/json' },
    });
  }

  // Stream the response body through unchanged.
  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: {
      'content-type': contentType || 'application/json',
      // Kill any edge/browser buffering on SSE.
      'cache-control': 'no-cache, no-transform',
      ...(isStream ? { connection: 'keep-alive' } : {}),
    },
  });
}

export async function GET() {
  // Pass-through health check so `/api/chat` can be probed without auth.
  // Maps to the Modal app's /health (no auth required).
  try {
    const r = await fetch(`${CHATBOT_ORIGIN}/health`);
    return NextResponse.json(
      { ok: r.ok, upstreamStatus: r.status, origin: CHATBOT_ORIGIN },
      { status: r.ok ? 200 : 502 },
    );
  } catch (err) {
    return NextResponse.json(
      { ok: false, error: String(err), origin: CHATBOT_ORIGIN },
      { status: 502 },
    );
  }
}
