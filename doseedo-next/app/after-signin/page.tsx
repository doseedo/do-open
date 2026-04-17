'use client';

import { useAuth } from '@clerk/nextjs';
import { useEffect, useState } from 'react';

/**
 * Post-Clerk-sign-in bridge.
 *
 * Clerk's <SignIn /> redirects here after a successful sign-in. We grab the
 * Clerk session JWT, POST it to /api/auth/clerk-bridge on Fly, which sets
 * the legacy `access_token` + `username` cookies the CRA app expects.
 * Then we redirect to /dashboard so the CRA app mounts with the user
 * already authenticated.
 */
export default function AfterSignIn() {
  const { getToken, isLoaded, isSignedIn } = useAuth();
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!isLoaded) return;
    if (!isSignedIn) {
      window.location.href = '/sign-in';
      return;
    }

    (async () => {
      try {
        const token = await getToken();
        if (!token) throw new Error('no Clerk token');
        const r = await fetch('/api/auth/clerk-bridge', {
          method: 'POST',
          credentials: 'include',
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!r.ok) {
          const body = await r.text();
          throw new Error(`bridge ${r.status}: ${body}`);
        }
        window.location.href = '/dashboard';
      } catch (e: any) {
        setErr(e?.message || 'sign-in bridge failed');
      }
    })();
  }, [isLoaded, isSignedIn, getToken]);

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
        gap: '1rem',
        color: '#eee',
        background: '#0a0a0f',
        fontFamily: 'Inter, system-ui, sans-serif',
      }}
    >
      {err ? (
        <>
          <div style={{ color: '#ff6b6b' }}>Sign-in failed: {err}</div>
          <a href="/sign-in" style={{ color: '#9aa' }}>Try again →</a>
        </>
      ) : (
        <div>Signing you in…</div>
      )}
    </div>
  );
}
