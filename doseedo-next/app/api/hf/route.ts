import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import { HfInference } from '@huggingface/inference';

/**
 * Server-side proxy to the Hugging Face Inference API.
 *
 * Why: the previous wiring read NEXT_PUBLIC_HF_API_TOKEN on the client,
 * which baked the token into every browser bundle. This route holds
 * HF_API_TOKEN server-side and forwards SDK calls. The client surface in
 * src/services/huggingfaceAPI.js was rewritten to POST here.
 *
 * Body: { kind, model, inputs, parameters }
 *   kind = 'textGeneration' | 'textToAudio' | 'request' | 'modelInfo' | 'isModelReady'
 *
 * Response shape:
 *   - textToAudio       → audio/* binary blob
 *   - textGeneration    → JSON
 *   - request           → JSON (or binary if upstream returned binary)
 *   - modelInfo         → JSON (HF /api/models/<id>)
 *   - isModelReady      → JSON { ready: boolean }
 *
 * Auth: Clerk-gated. Anon callers get 401 so a leaked endpoint can't be
 * scraped into a free HF inference budget.
 */

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type ProxyKind = 'textGeneration' | 'textToAudio' | 'request' | 'modelInfo' | 'isModelReady';

interface ProxyBody {
  kind: ProxyKind;
  model: string;
  inputs?: unknown;
  parameters?: Record<string, unknown>;
}

let _hf: HfInference | null = null;
function getClient(): HfInference | null {
  const token = process.env.HF_API_TOKEN;
  if (!token) return null;
  if (!_hf) _hf = new HfInference(token);
  return _hf;
}

export async function GET() {
  // Lightweight configuration probe. The client uses this in place of
  // the old isHFConfigured() helper that read NEXT_PUBLIC_HF_API_TOKEN.
  return NextResponse.json({ configured: !!process.env.HF_API_TOKEN });
}

export async function POST(req: NextRequest) {
  if (process.env.CLERK_SECRET_KEY) {
    const a = await auth();
    if (!a.userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
  }

  const client = getClient();
  if (!client) {
    return NextResponse.json(
      { error: 'HF_API_TOKEN not configured on server' },
      { status: 503 },
    );
  }

  let body: ProxyBody;
  try {
    body = (await req.json()) as ProxyBody;
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const { kind, model, inputs, parameters } = body || ({} as ProxyBody);
  if (!kind || !model) {
    return NextResponse.json({ error: 'Missing { kind, model }' }, { status: 400 });
  }

  try {
    switch (kind) {
      case 'textGeneration': {
        const out = await client.textGeneration({
          model,
          inputs: inputs as string,
          parameters,
        });
        return NextResponse.json(out);
      }

      case 'textToAudio': {
        // v4 of @huggingface/inference renamed `textToAudio` → `textToSpeech`.
        // The endpoint is the same — any text → audio model (TTS or
        // musicgen) works through the same call shape.
        const blob = await client.textToSpeech({
          model,
          inputs: inputs as string,
          parameters: parameters as never,
        });
        const buf = Buffer.from(await blob.arrayBuffer());
        return new NextResponse(buf, {
          status: 200,
          headers: {
            'content-type': blob.type || 'audio/wav',
            'cache-control': 'no-store',
          },
        });
      }

      case 'request': {
        const out = await client.request({
          model,
          inputs,
          parameters,
        });
        if (out instanceof Blob) {
          const buf = Buffer.from(await out.arrayBuffer());
          return new NextResponse(buf, {
            status: 200,
            headers: {
              'content-type': out.type || 'application/octet-stream',
              'cache-control': 'no-store',
            },
          });
        }
        return NextResponse.json(out);
      }

      case 'modelInfo': {
        const r = await fetch(`https://huggingface.co/api/models/${model}`, {
          headers: { Authorization: `Bearer ${process.env.HF_API_TOKEN}` },
        });
        if (!r.ok) {
          return NextResponse.json(
            { error: `Failed to get model info: ${r.status}` },
            { status: r.status },
          );
        }
        return NextResponse.json(await r.json());
      }

      case 'isModelReady': {
        try {
          await client.textGeneration({
            model,
            inputs: 'test',
            parameters: { max_new_tokens: 1 },
          });
          return NextResponse.json({ ready: true });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          if (msg.includes('loading')) {
            return NextResponse.json({ ready: false });
          }
          // Non-loading errors usually mean the model exists but doesn't
          // accept text-generation — treat as ready so callers don't spin.
          return NextResponse.json({ ready: true, note: msg });
        }
      }

      default:
        return NextResponse.json({ error: `Unknown kind: ${kind}` }, { status: 400 });
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: `HF API Error: ${msg}` }, { status: 502 });
  }
}
