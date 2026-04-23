'use client';

import { SignIn, useAuth } from '@clerk/nextjs';
import { useEffect } from 'react';
import { doseedoClerkAppearance } from '../../clerk-appearance';

export default function SignInPage() {
  const { isLoaded, isSignedIn } = useAuth();

  useEffect(() => {
    if (isLoaded && isSignedIn) {
      window.location.replace('/after-signin');
    }
  }, [isLoaded, isSignedIn]);

  return (
    <div
      style={{
        minHeight: '100vh',
        background: 'var(--wb-bg, #e8e6e1)',
        color: 'var(--wb-ink, #15181c)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '24px',
        fontFamily: 'var(--wb-font-sans, Inter, -apple-system, BlinkMacSystemFont, sans-serif)',
      }}
    >
      <div
        style={{
          width: '100%',
          maxWidth: '360px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '20px',
        }}
      >
        {isSignedIn ? (
          <div
            style={{
              color: 'var(--wb-ink-mute, #7c7e85)',
              fontFamily: 'var(--wb-font-mono, "JetBrains Mono", monospace)',
              fontSize: '11px',
              letterSpacing: '0.8px',
              textTransform: 'uppercase',
            }}
          >
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
