import { NextRequest, NextResponse } from 'next/server';
import { createHmac, timingSafeEqual } from 'node:crypto';
import { checkGateRateLimit, rateLimitHeaders } from '@/lib/ratelimit';

/**
 * Beta password-gate validation.
 *
 *   POST /api/auth/gate  { password: string }
 *   →    200 + Set-Cookie: dsd_gate=<hmac>; HttpOnly; Secure; SameSite=Lax
 *        401 on mismatch
 *
 * The password value lives only in the server-side `PROTECTED_PASSWORD`
 * env var — it never ships to the browser bundle. A successful POST sets
 * an HttpOnly cookie so a future hardening pass can require it on the
 * gated routes via middleware. Today the App.js PasswordGate still uses
 * sessionStorage for its view-state cache, but the *secret material* is
 * server-side, which closes the "grep the bundle for `oatmealbeach`"
 * vector.
 */

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const COOKIE_NAME = 'dsd_gate';
const COOKIE_TTL_S = 60 * 60 * 8; // 8h — beta UX, refreshes on next visit

function tsEqual(a: string, b: string): boolean {
  // node's timingSafeEqual requires equal-length buffers; pad to the
  // longer of the two with zeros so we never short-circuit on length.
  const len = Math.max(a.length, b.length, 1);
  const ba = Buffer.alloc(len, 0);
  const bb = Buffer.alloc(len, 0);
  ba.write(a);
  bb.write(b);
  let eq = timingSafeEqual(ba, bb);
  // Final guard for true equality — timingSafeEqual on padded buffers
  // would treat 'foo' and 'foo\0' as equal otherwise.
  return eq && a.length === b.length;
}

function clientIp(req: NextRequest): string {
  const xff = req.headers.get('x-forwarded-for');
  if (xff) return xff.split(',')[0].trim();
  return req.headers.get('cf-connecting-ip') || 'anon';
}

export async function POST(req: NextRequest) {
  const ip = clientIp(req);
  const rl = await checkGateRateLimit(ip);
  if (!rl.success) {
    return NextResponse.json(
      { error: 'too many attempts — wait a minute' },
      { status: 429, headers: rateLimitHeaders(rl) },
    );
  }

  const expected = process.env.PROTECTED_PASSWORD || '';
  if (!expected) {
    // Misconfiguration — fail open with 503 rather than letting anyone
    // through with an empty match.
    return NextResponse.json(
      { error: 'gate not configured on server' },
      { status: 503 },
    );
  }

  let payload: { password?: unknown } = {};
  try {
    payload = await req.json();
  } catch {
    return NextResponse.json({ error: 'bad json' }, { status: 400 });
  }
  const presented = typeof payload.password === 'string' ? payload.password : '';

  if (!tsEqual(presented, expected)) {
    return NextResponse.json({ ok: false }, { status: 401 });
  }

  // Mint a tiny HMAC so a stolen cookie value can't be hand-forged
  // without the server secret. Reusing PROTECTED_PASSWORD as the HMAC
  // key is fine — it never leaves the server and the cookie payload
  // doesn't reveal it.
  const ts = Math.floor(Date.now() / 1000).toString(36);
  const mac = createHmac('sha256', expected).update(ts).digest('hex').slice(0, 16);
  const value = `${ts}.${mac}`;

  const res = NextResponse.json({ ok: true });
  res.cookies.set(COOKIE_NAME, value, {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    path: '/',
    maxAge: COOKIE_TTL_S,
  });
  return res;
}
