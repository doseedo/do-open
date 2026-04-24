import { clerkMiddleware } from '@clerk/nextjs/server';
import { NextResponse, type NextRequest } from 'next/server';

// Clerk is wired in but no routes are gated on Clerk auth yet.
// The existing CRA-era cookie auth (authService / username cookie) still
// owns /studio, /dashboard, etc. — Clerk is available via useAuth()/useUser()
// hooks for progressive migration.
//
// Local-dev fallback: if CLERK_SECRET_KEY isn't set, skip clerkMiddleware so
// `npm run dev` works without provisioning Clerk first.
const hasClerkSecret = !!process.env.CLERK_SECRET_KEY;

// Rewrite (not redirect) unauthenticated visitors at "/" to "/home" so
// crawlers and link unfurlers see the Framer marketing page at the canonical
// URL. Authenticated visitors fall through to the SPA. The /home path is
// proxied to Framer by next.config.js rewrites.
function landingGate(request: NextRequest): NextResponse | undefined {
  if (request.nextUrl.pathname !== '/') return;
  if (request.cookies.has('username')) return;
  return NextResponse.rewrite(new URL('/home', request.url));
}

export default hasClerkSecret
  ? clerkMiddleware((_auth, request) => landingGate(request))
  : (request: NextRequest) => landingGate(request) ?? NextResponse.next();

export const config = {
  matcher: [
    // Skip Next.js internals, static files, and /api/* (those are rewrites to backends).
    '/((?!_next|.*\\..*|api).*)',
  ],
};
