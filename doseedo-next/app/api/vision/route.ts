import { NextRequest, NextResponse } from 'next/server';

/**
 * Server-side proxy to the Modal vision endpoint
 * (arlo--doseedo-chatbot-qwenchatbot-vision.modal.run).
 *
 * Mirrors app/api/chat/route.ts: keeps VLLM_GATE_TOKEN out of the browser
 * bundle. Accepts the same body shape the Modal FastAPI expects —
 *   POST /api/vision { image_base64, prompt, task?, object? }
 * — forwards verbatim. GET is a passthrough health probe.
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

export async function POST(req: NextRequest) {
  const apiKey = process.env.VLLM_GATE_TOKEN || process.env.VLLM_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: 'VLLM_GATE_TOKEN not configured on server' },
      { status: 500 },
    );
  }

  const body = await req.text();

  const upstream = await fetch(`${VISION_ORIGIN}/v1/vision/analyze`, {
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
