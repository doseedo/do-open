'use client';

import { SignUp, useAuth } from '@clerk/nextjs';
import { useEffect } from 'react';
import { doseedoClerkAppearance } from '../../clerk-appearance';

export default function SignUpPage() {
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
        <div style={{ textAlign: 'center' }}>
          <a
            href="/"
            style={{
              color: '#fff',
              textDecoration: 'none',
              fontSize: '1.4em',
              fontWeight: 700,
              letterSpacing: '-0.5px',
            }}
          >
            Doseedo
          </a>
          <div style={{ color: '#666', fontSize: '0.85em', marginTop: 4 }}>
            Create your studio account
          </div>
        </div>
        {isSignedIn ? (
          <div style={{ color: '#666', fontSize: '0.85em' }}>Redirecting…</div>
        ) : (
          <SignUp
            path="/sign-up"
            routing="path"
            signInUrl="/sign-in"
            forceRedirectUrl="/after-signin"
            fallbackRedirectUrl="/after-signin"
            appearance={doseedoClerkAppearance}
          />
        )}
      </div>
    </div>
  );
}
