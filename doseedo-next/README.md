# doseedo-next

Next.js 14 shell that hosts the existing CRA `doseedo-react` app on Vercel.

## What this is

A tier-1 port: the CRA `src/` tree is **synced** from `../var/www/html/doseedo-react/`
(the source of truth) and mounted inside a client-only `[[...slug]]` catch-all
route. `react-router-dom` keeps handling routing inside `App.js` — Next.js just
serves the shell and proxies API calls.

## Source of truth + sync

The canonical frontend lives at `../var/www/html/doseedo-react/src/`. Keep
editing there. `doseedo-next/src/` is **generated** by `sync.sh`, which
rsyncs and re-applies a small set of idempotent transforms
(`REACT_APP_` → `NEXT_PUBLIC_`, `.module.css` `:root` extraction, etc.).

```bash
npm run sync     # pull latest from doseedo-react + transform
npm run dev      # sync runs automatically (predev hook), then starts dev server
npm run build    # sync runs automatically (prebuild hook), then builds
```

On Vercel, `prebuild` means every deploy pulls the latest `doseedo-react/src/`
out of the same repo automatically — no need to commit `doseedo-next/src/`
after each frontend edit (though you can commit it if you want diff visibility).

Everything server-side is:
1. Static shell HTML at every route (cookie-gated redirect from `/` → `/home`)
2. `next.config.js` rewrites that proxy `/api/*`, `/home`, `/token`, etc. to
   the same Cloud Run / Modal / Framer origins the GCLB `bassify` url-map hit.

No `main.[hash].js` bundle hash matching — Next.js splits per-route and
deployments are tracked via Vercel deployment IDs.

## Local dev

```bash
cp .env.local.example .env.local   # fill in what you need
npm install
npm run dev                         # runs sync.sh, then localhost:3000
```

`/api/*` rewrites will hit the real prod Cloud Run / Modal backends. To point
at staging, override `NEXT_PUBLIC_AUTH_ORIGIN` / `_CHATBOT_ORIGIN` / `_MODAL_ORIGIN` /
`_BACKEND_ORIGIN` in `.env.local`.

## Production build

```bash
npm run build    # next build
npm start        # next start
```

## Deploy to Vercel

1. Push the repo to GitHub.
2. In Vercel dashboard → **Add New Project** → import the repo, set the root
   directory to `doseedo-next/`.
3. Set project env vars (all the `NEXT_PUBLIC_*` entries from `.env.local.example`).
4. Deploy. You'll get `doseedo-next-<hash>.vercel.app`.
5. Add `doseedo.com` as a custom domain; Vercel issues the TXT record for
   verification.
6. At Cloudflare, change the apex A/AAAA record to
   `CNAME doseedo.com → cname.vercel-dns.com` (proxied, orange-cloud). Set CF
   SSL to **Full (strict)**. Keep the existing GCLB record live on a spare
   hostname (`legacy.doseedo.com`) for 24h as a rollback.

### CLI alternative

```bash
export VERCEL_TOKEN=...     # create at https://vercel.com/account/tokens
npx vercel link
npx vercel deploy --prod
```

**Never commit the token** or paste it into chat. Store in a password manager
or a keychain, and pass via env var.

## Key files

| File                              | Purpose                                              |
|-----------------------------------|------------------------------------------------------|
| `app/layout.tsx`                  | HTML shell, fonts, favicons, pre-React auth gate     |
| `app/AppShell.tsx`                | Client-only dynamic import of the CRA `src/App.js`   |
| `app/[[...slug]]/page.tsx`        | Catch-all route → renders `AppShell`                 |
| `next.config.js`                  | Webpack tweaks + all `/api/*` → backend rewrites     |
| `src/`                            | CRA source copied verbatim (`REACT_APP_` → `NEXT_PUBLIC_`) |
| `src/styles/module-extracted-root.css` | `:root` blocks lifted out of `.module.css` files  |
| `src/styles/module-extracted-global.css` | Pure-`:global()` blocks lifted out of `.module.css` |

## What won't work without extra work

- **WebSockets** (`/ws/collab`, `/_chat/ws`). Vercel rewrites don't proxy WS
  traffic. Either point the client at a dedicated WS hostname (e.g.
  `wss://collab.doseedo.com` → Cloud Run backend via CF) or keep the GCLB
  route for `/ws/*` on a separate subdomain.
- **Source map upload to Sentry.** The CRA `deploy.sh` handled this; here it
  needs to run as a Vercel post-build step with `@sentry/webpack-plugin` or
  the Sentry CLI. Not wired up yet.

## Gotchas / non-obvious bits

1. **onnxruntime-web aliased to the UMD bundle** (`ort.min.js`). The default
   ESM bundle uses `new URL('ort.bundle.min.mjs', import.meta.url)` to load a
   threaded worker chunk, which Next.js's minifier rejects. The app already
   points `ort.env.wasm.wasmPaths` at jsdelivr CDN, so the UMD bundle is fine.
2. **`process.env.PUBLIC_URL`** is set to `''` in `next.config.js` — CRA
   code uses `${process.env.PUBLIC_URL}/assets/...` and without the shim those
   URLs would stringify to `"undefined/assets/..."` and 404.
3. **`.module.css` purity**: several files had top-level `:root { ... }` or
   bare `:global(.third-party-class) { ... }` blocks. Next.js's CSS-Modules
   rejects any selector without a local class/id. Those blocks are extracted
   into `src/styles/module-extracted-{root,global}.css` and imported globally
   from `app/AppShell.tsx`.
4. **`html2canvas`** was dynamic-imported but missing from CRA's
   `package.json`. Added here as an explicit dep.
5. **`shader-utils.js` → `.ts`**: two `shader-utils.js` files contained
   TypeScript `interface` declarations. CRA's Babel was lenient; Next's TS
   checker isn't. Renamed to `.ts` so they parse correctly.

## Rollback plan

If Vercel deploy misbehaves, revert the Cloudflare apex DNS from the Vercel
CNAME back to the GCLB A record. The Cloud Run service (`doseedo-frontend`)
and GCS bucket (`doseedo-frontend-static`) are untouched until you explicitly
tear them down.
