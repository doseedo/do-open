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
        background: '#0a0a0a',
        color: '#fff',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '24px',
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "Inter", sans-serif',
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
          <div style={{ color: '#666', fontSize: '0.85em' }}>Redirecting…</div>
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
