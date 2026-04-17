import { clerkMiddleware } from '@clerk/nextjs/server';

// Clerk is wired in but no routes are gated on Clerk auth yet.
// The existing CRA-era cookie auth (authService / access_token cookie) still
// owns /studio, /dashboard, etc. — Clerk is available via useAuth()/useUser()
// hooks for progressive migration.
//
// When we're ready to flip a route to Clerk, switch to clerkMiddleware((auth, req) => {
//   if (isProtectedRoute(req)) auth().protect();
// });
export default clerkMiddleware();

export const config = {
  matcher: [
    // Skip Next.js internals, static files, and /api/* (those are rewrites to backends).
    '/((?!_next|.*\\..*|api).*)',
  ],
};
