import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import { promises as fs } from 'fs';
import path from 'path';

/**
 * Dev debug-log sink. Client posts JSON {entries: [...]} and we append
 * one JSONL line per entry to /tmp/doseedo-studio-debug.jsonl so the
 * assistant can tail it without the user copy-pasting.
 *
 * Only enabled when the client opts in via ?debug=1 or
 * localStorage.doseedoDebugLog = 'on'. In production the file sink
 * becomes a no-op — logs go to stdout (Vercel/Fly picks them up via
 * `vercel logs` / `fly logs`).
 *
 * POST is Clerk-gated: anonymous clients can't dump arbitrary lines into
 * the platform log pipeline. GET stays open as a tiny health/summary
 * endpoint.
 */

const LOCAL_SINK = process.env.DOSEEDO_DEBUG_LOG_FILE || '/tmp/doseedo-studio-debug.jsonl';
const MAX_FILE_BYTES = 10 * 1024 * 1024;  // 10MB soft cap — rotate when exceeded

async function rotateIfLarge(filePath: string) {
  try {
    const stat = await fs.stat(filePath);
    if (stat.size > MAX_FILE_BYTES) {
      await fs.rename(filePath, `${filePath}.1`);
    }
  } catch {
    // file doesn't exist yet — nothing to rotate
  }
}

export async function POST(req: NextRequest) {
  if (process.env.CLERK_SECRET_KEY) {
    const { userId } = await auth();
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
  }

  let body: any;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'invalid json' }, { status: 400 });
  }
  const entries = Array.isArray(body?.entries) ? body.entries : [];
  if (entries.length === 0) return NextResponse.json({ ok: true, count: 0 });

  const tServer = Date.now();
  const lines = entries
    .map((e: any) => JSON.stringify({ ...e, tServer }))
    .join('\n') + '\n';

  if (process.env.NODE_ENV === 'production') {
    // Prod — let the platform log pipeline capture it.
    for (const e of entries) {
      // eslint-disable-next-line no-console
      console.log(`[studio-debug] ${JSON.stringify(e)}`);
    }
    return NextResponse.json({ ok: true, count: entries.length, sink: 'stdout' });
  }

  // Dev — append to file.
  try {
    await fs.mkdir(path.dirname(LOCAL_SINK), { recursive: true });
    await rotateIfLarge(LOCAL_SINK);
    await fs.appendFile(LOCAL_SINK, lines);
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
  return NextResponse.json({ ok: true, count: entries.length, sink: LOCAL_SINK });
}

export async function GET() {
  // Small health/summary endpoint so the user can curl to check state.
  try {
    const stat = await fs.stat(LOCAL_SINK);
    return NextResponse.json({ ok: true, sink: LOCAL_SINK, bytes: stat.size, mtime: stat.mtime });
  } catch {
    return NextResponse.json({ ok: true, sink: LOCAL_SINK, bytes: 0 });
  }
}
