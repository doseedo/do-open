/**
 * /dev/validation — internal A/B fidelity tool.
 *
 * Note on file location:
 *   The spec for R12 lists this file at `src/app/dev/validation/page.js`,
 *   but the Doseedo-Next App Router root is at `app/`, not `src/app/`.
 *   Next.js will only auto-route the actual `app/dev/validation/page.js`
 *   file; this `src/` copy exists to satisfy the deliverables manifest
 *   and to act as a single source of truth for the page body. The
 *   `app/dev/validation/page.js` re-exports from here.
 *
 *   If a future build refactors the app root to live under `src/app/`,
 *   the `app/` copy can simply be deleted and Next.js will pick up this
 *   file directly.
 */

'use client';

import dynamic from 'next/dynamic';

// Load the panel client-only — it pokes at AudioContext on mount and would
// crash any SSR pass.
const ValidationPanel = dynamic(
  () => import('../../../components/Plugins/ValidationPanel/ValidationPanel'),
  { ssr: false, loading: () => <div style={{ padding: 16, color: '#8a93a0', fontFamily: 'monospace' }}>Loading validation panel…</div> },
);

export default function ValidationPage() {
  return <ValidationPanel />;
}

// Force dynamic rendering — entire page is browser-only.
export const dynamicParams = true;
