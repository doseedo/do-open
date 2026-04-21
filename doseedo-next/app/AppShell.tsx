'use client';

import dynamic from 'next/dynamic';

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

export default function AppShell() {
  return <App />;
}
