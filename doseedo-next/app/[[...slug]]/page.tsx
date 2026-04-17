import AppShell from '../AppShell';

// The catch-all ([[...slug]]) lets Next.js serve every path through the
// client shell. react-router-dom inside App.js reads window.location and
// handles routing itself — Next just needs to render the shell everywhere.
export default function CatchAllPage() {
  return <AppShell />;
}

// Force dynamic rendering — the app is cookie-gated and fully client-side.
export const dynamic = 'force-dynamic';
