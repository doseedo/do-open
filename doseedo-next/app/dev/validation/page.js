/**
 * Real Next.js App Router entry for /dev/validation.
 *
 * Re-exports the page from `src/app/dev/validation/page.js` (which is the
 * R12-spec-canonical location). The body of the page lives there.
 */

export { default } from '../../../src/app/dev/validation/page';

// Force dynamic — the page mounts an AudioContext on the client.
export const dynamic = 'force-dynamic';
