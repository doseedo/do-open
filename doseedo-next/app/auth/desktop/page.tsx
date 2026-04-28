'use client';

/**
 * Desktop OAuth handoff page.
 *
 * Phase 2 of the auth migration: the Electron app no longer asks for an
 * email/password. Instead it generates a PKCE code, opens the system browser
 * to /auth/desktop?code=<pkce>, and we mint a Desktop dsk_live_ key on the
 * user's behalf, stash it server-side under sha256(code), and bounce back
 * to the desktop via a `doseedo://auth?code=…` deep link.
 *
 * The page never sees the API key. Only /desktop-handoff (server) sees it,
 * and only the desktop can fetch it back via /desktop-claim with the same
 * code. So a leaked URL is useless to anyone but the desktop instance that
 * generated it.
 */

import { useEffect, useState } from 'react';
import { useAuth } from '@clerk/nextjs';

const AUTH_SERVICE =
  process.env.NEXT_PUBLIC_AUTH_ORIGIN || 'https://doseedo-api.fly.dev';

export default function DesktopAuthPage() {
  const { isLoaded, isSignedIn, getToken } = useAuth();
  const [status, setStatus] = useState<'pending' | 'success' | 'error'>('pending');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isLoaded) return;

    const code = new URLSearchParams(window.location.search).get('code') || '';
    if (!code || code.length < 32 || code.length > 128) {
      setStatus('error');
      setError('Missing or invalid auth code in URL.');
      return;
    }

    if (!isSignedIn) {
      // Bounce to Clerk sign-in with this URL as the return target so we
      // come right back here with the code preserved.
      const returnTo = window.location.pathname + window.location.search;
      window.location.href = `/sign-in?redirect_url=${encodeURIComponent(returnTo)}`;
      return;
    }

    (async () => {
      try {
        const token = await getToken();
        if (!token) throw new Error('No Clerk session token');

        // Pick a friendly host label. The detail isn't critical — it shows up
        // in /settings/api-keys so the user can recognize which device a key
        // belongs to. Kept short to fit the auth-service's 80-char limit.
        const ua = navigator.userAgent;
        const host = ua.match(/Mac/) ? 'Mac' : ua.match(/Windows/) ? 'Windows' : ua.match(/Linux/) ? 'Linux' : 'Desktop';

        const resp = await fetch(`${AUTH_SERVICE}/api/auth/desktop-handoff`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ code, client_label: `Desktop / ${host}` }),
        });
        if (!resp.ok) {
          const body = await resp.text().catch(() => '');
          throw new Error(`Handoff failed (${resp.status})${body ? `: ${body.slice(0, 200)}` : ''}`);
        }

        // Now redirect to the deep link. The Electron app's main-process
        // open-url handler picks this up and forwards the code to the
        // renderer, which calls /desktop-claim to fetch the api_key.
        setStatus('success');
        window.location.href = `doseedo://auth?code=${encodeURIComponent(code)}`;
      } catch (e: any) {
        setStatus('error');
        setError(e?.message || 'Handoff failed');
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
        padding: '24px',
        gap: '16px',
        background: 'var(--wb-bg, #e8e6e1)',
        color: 'var(--wb-ink, #15181c)',
        fontFamily: 'var(--wb-font-sans, Inter, -apple-system, BlinkMacSystemFont, sans-serif)',
        textAlign: 'center',
      }}
    >
      {status === 'pending' && (
        <div
          style={{
            color: 'var(--wb-ink-mute, #7c7e85)',
            fontFamily: 'var(--wb-font-mono, "JetBrains Mono", monospace)',
            fontSize: '11px',
            letterSpacing: '0.8px',
            textTransform: 'uppercase',
          }}
        >
          Authorizing your desktop app…
        </div>
      )}

      {status === 'success' && (
        <div style={{ maxWidth: 420 }}>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 600, marginBottom: '12px' }}>
            Almost there.
          </h2>
          <p style={{ color: 'var(--wb-ink-mute, #7c7e85)', marginBottom: '8px' }}>
            If the Doseedo desktop app didn't reopen automatically, return to it manually —
            your sign-in is complete.
          </p>
          <p
            style={{
              color: 'var(--wb-ink-mute, #7c7e85)',
              fontFamily: 'var(--wb-font-mono, "JetBrains Mono", monospace)',
              fontSize: '11px',
              letterSpacing: '0.8px',
              textTransform: 'uppercase',
              marginTop: '20px',
            }}
          >
            You can close this window.
          </p>
        </div>
      )}

      {status === 'error' && (
        <div style={{ maxWidth: 420 }}>
          <h2
            style={{
              fontSize: '1.2rem',
              fontWeight: 600,
              marginBottom: '12px',
              color: 'var(--wb-accent-warm, #c94f2c)',
            }}
          >
            Couldn't authorize the desktop app.
          </h2>
          <p style={{ color: 'var(--wb-ink-mute, #7c7e85)', marginBottom: '8px' }}>{error}</p>
          <p style={{ color: 'var(--wb-ink-mute, #7c7e85)' }}>
            Try signing in again from the desktop app, or contact support if this keeps
            happening.
          </p>
        </div>
      )}
    </div>
  );
}
