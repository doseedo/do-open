import { clerkMiddleware } from '@clerk/nextjs/server';
import { NextResponse, type NextRequest } from 'next/server';

// Clerk is now the source of truth for the landing gate. The CRA-era
// `username` cookie is still set by /after-signin's clerk-bridge call (so
// CRA pages like /dashboard, /studio that read the cookie keep working
// during the migration), but it's no longer trusted here — landing-gate
// uses Clerk session auth directly.
//
// Local-dev fallback: if CLERK_SECRET_KEY isn't set, skip clerkMiddleware
// so `npm run dev` works without provisioning Clerk first. In that case
// we can't tell signed-in from anonymous, so we always fall through to
// the SPA (the dev's job to mock auth as needed).
const hasClerkSecret = !!process.env.CLERK_SECRET_KEY;

// Rewrite (not redirect) anonymous visitors at "/" to "/home" so crawlers
// and link unfurlers see the Framer marketing page at the canonical URL.
// Authenticated visitors fall through to the SPA. The /home path is
// proxied to Framer by next.config.js rewrites.
function landingRewrite(request: NextRequest): NextResponse {
  return NextResponse.rewrite(new URL('/home', request.url));
}

export default hasClerkSecret
  ? clerkMiddleware(async (clerkAuth, request) => {
      if (request.nextUrl.pathname !== '/') return;
      const { userId } = await clerkAuth();
      if (userId) return; // signed in — fall through to SPA
      return landingRewrite(request);
    })
  : (request: NextRequest) => {
      if (request.nextUrl.pathname !== '/') return NextResponse.next();
      // No Clerk in dev — just show the SPA at "/".
      return NextResponse.next();
    };

export const config = {
  matcher: [
    // Skip Next.js internals and static files. We deliberately INCLUDE
    // /api/* — Clerk's `auth()` helper needs clerkMiddleware() to have
    // run upstream of the route handler, otherwise it throws
    // "auth() was called but Clerk can't detect usage of clerkMiddleware()"
    // and every gated API route 500s. The landing-rewrite block below
    // bails on anything that isn't `/`, so widening the matcher just
    // attaches the auth context without changing behavior elsewhere.
    '/((?!_next|.*\\..*).*)',
  ],
};
