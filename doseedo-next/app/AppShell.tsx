'use client';

import dynamic from 'next/dynamic';
import { useAuth } from '@clerk/nextjs';
import { useEffect } from 'react';

// CSS import chain ported from src/index.js — order matters.
import '@/styles/colors.css';
import '@/styles/theme-workbench.css'; // Master workbench theme — body.workbench-theme on /studio, /dashboard, ...
import '@/styles/theme-hifi-purple.css'; // Dev variant — scoped to body.theme-hifi-purple (applied on /studio)
import '@/styles/glass-theme-background.css';
import '@/styles/liquid-glass.css';
import '@/assets/css/original-style5.css';
import '@/assets/css/App.css';
// :root blocks extracted from *.module.css files (Next.js CSS modules forbid
// non-pure selectors, so they live globally here instead).
import '@/styles/module-extracted-root.css';
import '@/styles/module-extracted-global.css';

// Load the CRA App as a client-only chunk. ssr:false prevents Next.js
// from trying to render browser-only deps (wavesurfer, onnxruntime-web,
// vexflow, @xyflow/react) on the server, which would break the build.
const App = dynamic(() => import('@/App'), {
  ssr: false,
  loading: () => null,
});

/**
 * Bridges Clerk's `useAuth().getToken()` into a global accessor so non-React
 * service modules (sessionAPI, gcsUploadService, generationAPI via httpClient)
 * can attach `Authorization: Bearer <jwt>` as a dev-mode fallback when the
 * `.doseedo.com` auth cookie isn't in scope (e.g. on localhost).
 */
function ClerkTokenBridge() {
  const { getToken, isLoaded, isSignedIn } = useAuth();
  useEffect(() => {
    if (typeof window === 'undefined') return;
    // Always install the getter — httpClient tolerates failures.
    (window as any).__clerkGetToken = async () => {
      try {
        return await getToken();
      } catch {
        return null;
      }
    };
    // Also prime a static snapshot for contexts that can't await.
    if (isLoaded && isSignedIn) {
      getToken()
        .then((t) => {
          if (t) window.localStorage.setItem('clerk_token', t);
        })
        .catch(() => {});
    }
  }, [getToken, isLoaded, isSignedIn]);
  return null;
}

export default function AppShell() {
  return (
    <>
      <ClerkTokenBridge />
      <App />
    </>
  );
}
