import { clerkMiddleware } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';

// Clerk is wired in but no routes are gated on Clerk auth yet.
// The existing CRA-era cookie auth (authService / access_token cookie) still
// owns /studio, /dashboard, etc. — Clerk is available via useAuth()/useUser()
// hooks for progressive migration.
//
// Local-dev fallback: if CLERK_SECRET_KEY isn't set, export a no-op
// middleware so `npm run dev` works without provisioning Clerk first.
// clerkMiddleware() asserts the key at every request otherwise.
const hasClerkSecret = !!process.env.CLERK_SECRET_KEY;
export default hasClerkSecret ? clerkMiddleware() : () => NextResponse.next();

export const config = {
  matcher: [
    // Skip Next.js internals, static files, and /api/* (those are rewrites to backends).
    '/((?!_next|.*\\..*|api).*)',
  ],
};
