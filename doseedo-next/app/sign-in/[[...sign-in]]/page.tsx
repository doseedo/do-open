'use client';

import { SignIn, useAuth } from '@clerk/nextjs';
import { useEffect } from 'react';
import { doseedoClerkAppearance } from '../../clerk-appearance';

export default function SignInPage() {
  const { isLoaded, isSignedIn } = useAuth();

  // Already signed in? Skip straight to the bridge so the CRA cookies get set.
  useEffect(() => {
    if (isLoaded && isSignedIn) {
      window.location.replace('/after-signin');
    }
  }, [isLoaded, isSignedIn]);

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '2rem 1rem',
        background:
          'radial-gradient(ellipse at top left, rgba(102,126,234,0.18) 0%, transparent 55%),' +
          'radial-gradient(ellipse at bottom right, rgba(139,92,246,0.18) 0%, transparent 55%),' +
          '#070711',
      }}
    >
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1.75rem',
          width: '100%',
          maxWidth: '420px',
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '0.35rem',
          }}
        >
          <a
            href="/"
            style={{
              fontFamily: 'Inter, system-ui, sans-serif',
              fontSize: '1.85rem',
              fontWeight: 700,
              letterSpacing: '-0.02em',
              color: '#ffffff',
              textDecoration: 'none',
              background: 'linear-gradient(135deg, #667eea 0%, #8b5cf6 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}
          >
            doseedo
          </a>
          <div
            style={{
              fontFamily: 'Inter, system-ui, sans-serif',
              fontSize: '0.875rem',
              color: 'rgba(255,255,255,0.55)',
            }}
          >
            Welcome back
          </div>
        </div>
        {isSignedIn ? (
          <div style={{ color: 'rgba(255,255,255,0.7)', fontFamily: 'Inter, sans-serif' }}>
            Redirecting…
          </div>
        ) : (
          <SignIn
            path="/sign-in"
            routing="path"
            signUpUrl="/sign-up"
            forceRedirectUrl="/after-signin"
            fallbackRedirectUrl="/after-signin"
            appearance={doseedoClerkAppearance}
          />
        )}
      </div>
    </div>
  );
}
