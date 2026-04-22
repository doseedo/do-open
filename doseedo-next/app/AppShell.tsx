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
// original-style5.css (3.5k lines, pre-workbench DAW chrome) is lazy-
// loaded on /studio-legacy by src/App.js so the workbench routes
// don't inherit its legacy input[type=range] / body / button rules.
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
  const { getToken } = useAuth();
  useEffect(() => {
    if (typeof window === 'undefined') return;
    // Expose a live token getter. Callers that can await use this; they
    // must NOT read from localStorage['clerk_token'], because that snapshot
    // survives Clerk dev→prod instance swaps and returns tokens whose
    // signing kid isn't in the new JWKS (auth-service then 401s).
    (window as any).__clerkGetToken = async () => {
      try {
        return await getToken();
      } catch {
        return null;
      }
    };
    // Clean up any stale snapshot left over from an earlier Clerk instance.
    try { window.localStorage.removeItem('clerk_token'); } catch {}
  }, [getToken]);
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
