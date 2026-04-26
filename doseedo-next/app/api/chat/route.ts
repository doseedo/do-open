import { NextRequest, NextResponse } from 'next/server';

/**
 * Server-side proxy to the Modal vLLM chatbot
 * (arlo--doseedo-chatbot-qwenchatbot-serve.modal.run).
 *
 * Keeps the gate token out of the browser bundle. Accepts an OpenAI-style
 * chat completion request body and forwards it verbatim, including the
 * `stream: true` case — we pipe the Modal SSE response straight to the
 * client so tokens arrive with the same latency as a direct call.
 */

export const runtime = 'nodejs';
// Streamed responses must be dynamic, not cached.
export const dynamic = 'force-dynamic';

const CHATBOT_ORIGIN =
  process.env.CHATBOT_ORIGIN ||
  'https://arlo--doseedo-chatbot-qwenchatbot-serve.modal.run';

export async function POST(req: NextRequest) {
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

  const upstream = await fetch(`${CHATBOT_ORIGIN}/v1/chat/completions`, {
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
