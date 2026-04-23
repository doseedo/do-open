# Launch Runbook

Operational procedures for the doseedo production stack. Keep this file
terse — one section per procedure, each reproducible from a cold start.

## Stack

| Tier | Service | Host | Code |
|---|---|---|---|
| Auth / data | `doseedo-api` | `doseedo-api.fly.dev` (Fly) | `~/Downloads/doseedo-desktop/auth-service` |
| Frontend | Next 14 | `doseedo.com` (Vercel, GitHub integration) | `doseedo-next/` (mirror of `var/www/html/doseedo-react/src`) |
| Audio GPU | Modal app `doseedo-stemphonic` | `arlo--doseedo-stemphonic-stemphonic-wsgi.modal.run`, fronted by `doseedo.com/<path>` via Vercel rewrites | `modal/modal_stemphonic.py` + `stemphonic_server.py` |
| Identity | Clerk production instance | `clerk.doseedo.com` (JWKS), `accounts.doseedo.com`, `clkmail.doseedo.com` | `pk_live_…` / `sk_live_…` |
| DB | Neon Postgres | via `DATABASE_URL` on Fly | — |
| Redis | Upstash | via `REDIS_URL` on Fly (**db=0 only**) | — |
| Blob storage | Cloudflare R2 | via `R2_*` env on Fly | — |

## Request lifecycle for gated routes (`/separate-stems`, `/api/generate-stemphonic`, etc.)

```
Browser ─► doseedo.com ─► Modal (stemphonic_server.py)
                           │
                           ▼ before_request gate (modal_stemphonic.py)
                           │  forwards Authorization / Cookie / X-API-Key
                           │  + X-Internal-Secret (from Modal secret doseedo-gate)
                           ▼
                           Fly doseedo-api /internal/generation/consume
                           ├─► verify_clerk_token (Clerk JWT) OR decode_access_token (legacy)
                           ├─► Upstash INCR gen:daily:{env}:{user_id}:{YYYY-MM-DD}
                           └─► returns 200 {allowed, remaining} / 401 / 429 / 503
```

**Fail-open invariants** (things that *must* hold):

- Modal's `doseedo-gate` secret `INTERNAL_SECRET` **byte-exact matches** Fly's `INTERNAL_SECRET`. Use `printf`, never `echo`, when piping (trailing newline footgun).
- `AUTH_SERVICE_URL` in Modal secret → `https://doseedo-api.fly.dev`. If this points at a dead URL, every gated request fails closed with 503.
- Fly has `CLERK_PUBLISHABLE_KEY` (`pk_live_…`) + `CLERK_SECRET_KEY` (`sk_live_…`) set. The publishable key is what the JWT verifier decodes to recover the JWKS issuer; if it's missing/stale, `verify_clerk_token` returns `None` and everything falls through to legacy JWT, which Clerk-only users don't have → 401.
- `REDIS_URL` points at Upstash. Upstash only supports `db=0`. Never pass `db=N` to `aioredis.from_url` — namespace by key prefix instead.

## Routine deploy

Each tier ships independently. Changes to shared contract (gate headers, Clerk claims, internal-secret format) require deploying in order: Fly → Modal → Vercel.

### Fly auth-service
```bash
cd ~/Downloads/doseedo-desktop/auth-service
flyctl deploy --app doseedo-api
# Smoke: returns 200 {allowed:true,…}
SECRET=$(flyctl ssh console --app doseedo-api -C 'printenv INTERNAL_SECRET' | tail -1)
TOKEN="<valid JWT — see below>"
curl -sS -X POST https://doseedo-api.fly.dev/internal/generation/consume \
  -H "Content-Type: application/json" -H "X-Internal-Secret: $SECRET" \
  -H "Authorization: Bearer $TOKEN" -d '{"endpoint":"/separate-stems"}'
```

Minting a test legacy JWT (for smoke tests, not user flows):
```bash
flyctl ssh console --app doseedo-api --command 'sh -c "python -c \"import asyncio, sys; sys.path.insert(0,\\\"/app\\\"); from app.security import create_access_token; from app.database import get_session_factory; from app.models import User; from sqlalchemy import select
async def m():
    async with get_session_factory()() as db:
        r = await db.execute(select(User).where(User.is_active==True).limit(1))
        u = r.scalar_one_or_none()
        print(create_access_token({\\\"sub\\\": str(u.id)}))
asyncio.run(m())
\""'
```

### Modal stemphonic
```bash
cd ~/Downloads/Do/modal && modal deploy modal_stemphonic.py
```
Image rebuilds run 3–7 min. The deploy CLI occasionally crashes with `H2Connection` gRPC errors mid-build — the server-side build usually still completes; re-run `modal deploy` once to let it finish. Smoke test:
```bash
curl -sS -o /dev/null -w "HTTP %{http_code}\n" -X POST \
  https://arlo--doseedo-stemphonic-stemphonic-wsgi.modal.run/separate-stems \
  -H "Authorization: Bearer $TOKEN"
# 400 "No audioFile in request" = gate passed, reached handler ✓
# 401 "Authentication required for generation" = gate rejected (bad)
# 503 "Generation service unavailable" = auth-service 5xx (bad)
```

### Vercel frontend
Push to `main` — Vercel's GitHub integration auto-deploys. Manual deploy from repo root (the project's root-dir setting *insists on being run from `Do/`, not `Do/doseedo-next/`*):
```bash
cd ~/Downloads/Do && vercel --prod --yes
```

## Secrets & rotation

### Coordinates

| Secret | Fly (`doseedo-api`) | Modal (`doseedo-gate`) | Vercel |
|---|---|---|---|
| `INTERNAL_SECRET` | ✓ | ✓ | — |
| `AUTH_SERVICE_URL` | — | ✓ (`https://doseedo-api.fly.dev`) | — |
| `DISABLE_ALL_GENERATION` | — | ✓ (`"true"` → kill switch) | — |
| `CLERK_PUBLISHABLE_KEY` | ✓ | — | `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` |
| `CLERK_SECRET_KEY` | ✓ | — | ✓ |
| `DATABASE_URL` | ✓ | — | — |
| `REDIS_URL` | ✓ | — | — |
| `R2_*` (4 keys) | ✓ | — | — |

### Rotating `INTERNAL_SECRET`
Must be byte-identical on Fly *and* Modal. Use `printf`, not `echo`:
```bash
NEW=$(openssl rand -hex 32)
[ ${#NEW} -eq 64 ] || { echo "bad length"; exit 1; }

# 1. Fly
flyctl secrets set --app doseedo-api INTERNAL_SECRET="$NEW"
# (Fly auto-restarts both machines)

# 2. Modal — preserve the other two keys
modal secret create doseedo-gate --force \
  "AUTH_SERVICE_URL=https://doseedo-api.fly.dev" \
  "INTERNAL_SECRET=$NEW" \
  "DISABLE_ALL_GENERATION=false"
cd ~/Downloads/Do/modal && modal deploy modal_stemphonic.py

# 3. Verify (no JWT → expect 401 "Authentication required", NOT 403)
curl -sS -o /dev/null -w "%{http_code}\n" -X POST https://doseedo-api.fly.dev/internal/generation/consume \
  -H "X-Internal-Secret: $NEW" -H "Content-Type: application/json" -d '{}'
```
**403 after rotation = the secret doesn't match.** Recheck length (64) and re-run. **503 in the browser after rotation = Modal didn't pick up the new value** — redeploy Modal.

### Rotating Clerk keys (dev → prod or prod → prod)
```bash
# 1. Fly
flyctl secrets set --app doseedo-api \
  CLERK_PUBLISHABLE_KEY="pk_live_…" \
  CLERK_SECRET_KEY="sk_live_…"

# 2. Vercel — needs rm + add for each of (production, preview, development)
cd ~/Downloads/Do/doseedo-next
for env in production preview development; do
  vercel env rm NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY $env --yes 2>/dev/null
  # NOTE: `vercel env add` needs an empty branch arg after `preview`, else it 500s silently
  vercel env add NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY $env "" --value "pk_live_…" --yes
done
vercel env rm CLERK_SECRET_KEY production --yes
vercel env add CLERK_SECRET_KEY production --value "sk_live_…" --yes

# 3. Verify JWKS resolves before redeploy
curl -sS https://clerk.doseedo.com/.well-known/jwks.json | head -c 80
# Should start with {"keys":[{"use":"sig",...

# 4. Redeploy
cd ~/Downloads/Do && vercel --prod --yes
```
Email linking: existing local users auto-link to their new Clerk `sub` via `resolve_user_from_clerk_claims` → email match → `users.clerk_user_id` update. No manual migration needed.

## Pre-merge checklist

Before merging anything that touches these files:

- [ ] **`modal/modal_stemphonic.py` image deps** — if removing a `pip_install` line, `grep -rn "import <pkg>\|from <pkg>" stemphonic_server.py modal/` must return zero hits. Past incident: demucs was removed from the image with a comment saying "replaced by latent student models", but `stemphonic_server.py:2997` still does `import demucs.api` → every `/separate-stems` call 500'd with `ModuleNotFoundError`.
- [ ] **`app/routers/generation_gate.py`** — any change to the identity-resolution block must mirror `app/deps.py:get_current_user`'s Clerk-first → legacy-JWT-fallback order. Drifting here re-breaks Clerk users.
- [ ] **`modal_stemphonic.py` `before_request` hook** — the pre-check must accept any of: `Authorization`, `X-API-Key`, `access_token` cookie, `__session` cookie. If you add another cookie/header shape on the frontend, add it here too or users 401 at Modal's door.
- [ ] **Removing a `.pip_install` or `.apt_install` line** — run `modal app logs doseedo-stemphonic` for a minute after deploying to catch cold-start import errors. They don't surface until the first real request hits that code path.
- [ ] **`aioredis.from_url` calls in auth-service** — never pass `db=N`. Past incident: `db=1` in `generation_gate.py` → `ResponseError: Only 0th database is supported` → 500 → every `/separate-stems` call 503'd.

## Debugging a failed gated request

Symptom → what to check first:

| User sees | Modal logs (`modal app logs doseedo-stemphonic`) | Fly logs (`flyctl logs --app doseedo-api`) | Likely cause |
|---|---|---|---|
| `503 Generation service unavailable` | `gate: unexpected HTTP 500 from auth-service` | traceback near `/internal/generation/consume` | auth-service bug — check `db=N` on aioredis, Clerk setting, DB connection |
| `503 Generation service unavailable` | `gate: unexpected HTTP 403 from auth-service` | 403s on `POST /internal/generation/consume` | `INTERNAL_SECRET` mismatch — re-run rotation |
| `503 Generation service unavailable` | `gate: auth-service unreachable` | (no traffic) | `AUTH_SERVICE_URL` in Modal secret points at a dead host |
| `401 Authentication required for generation` | `POST /separate-stems -> 401` | (no traffic — pre-check fired) | Browser sent no accepted auth shape — check Modal's pre-check & frontend attaching `Authorization: Bearer <clerk_jwt>` from `window.__clerkGetToken()` |
| `401 Authentication required for generation` | `POST /separate-stems -> 401` | `/internal/generation/consume -> 401` | Clerk JWT invalid or `CLERK_PUBLISHABLE_KEY` on Fly points at wrong instance (JWKS can't verify signature) |
| `429 Daily generation limit reached` | — | `/internal/generation/consume -> 429` | Expected — free tier cap, 10/day. |
| `500 ModuleNotFoundError: No module named 'X'` | Python traceback in Modal handler | — | Missing pip dep in image — add to `modal_stemphonic.py` image chain and redeploy |

## Known foot-guns

- **`vercel env add NAME preview`** silently no-ops when the `preview` arg is the last positional. Fix: pass `""` as a trailing branch arg — `vercel env add NEXT_PUBLIC_FOO preview "" --value … --yes`.
- **Modal CLI `H2Connection` gRPC crash** — `AttributeError: 'H2Connection' object has no attribute '_frame_dispatch_table'` appears alongside deploys. Harmless; the server-side deploy usually completes. Workaround: `pip install -U 'h2>=4.3,<5'` and re-run `modal deploy`.
- **Modal `file modified during build process`** — Modal aborts image build if any file it's copying changes mid-copy. On Mac deploys, `_DO2` is the full monorepo (`~/Downloads/Do`), so any live watcher can trip this — observed offenders: `.claude/scheduled_tasks.lock` (a running Claude Code session), `doseedo-next/**/*.module.css` (webpack/vercel dev), `.DS_Store` (Finder). Fix: add the offending path to `_ignore_patterns()` in `modal/modal_stemphonic.py`. The base ignore list already excludes `.claude/`, `doseedo-next/`, `var/`, `node_modules/`, `.next/`, `.vercel/`, `.DS_Store`, `*.swp`, `*.tmp`, `*.log` for this reason — add more if you see the error again on a new path. (Changing `_ignore_patterns` invalidates the image layer cache, so the first deploy after a change rebuilds all pip layers.)
- **Trailing newline in secret files** — `gcloud secrets versions add --data-file=-` stored `echo $VAL` with the newline, causing 403s. Always use `printf %s` when writing secrets to files.
- **Vercel project root-dir** — Vercel's project setting pins root at `Do/` (not `Do/doseedo-next/`). Running `vercel --prod` from `doseedo-next/` fails with `path "~/Downloads/Do/doseedo-next/doseedo-next" does not exist`. Run from `Do/` instead.
- **Clerk dev-mode cookies don't attach to doseedo.com** — `__session` in dev lives on `*.clerk.accounts.dev`, so `credentials: 'include'` on a doseedo.com fetch won't carry it. The frontend must call `window.__clerkGetToken()` and pass the JWT as `Authorization: Bearer`. Production mode (`clerk.doseedo.com` CNAME) sets the cookie on `.doseedo.com` so it rides along automatically.

## Kill switch

One-line shutdown of all paid-gen routes:
```bash
modal secret create doseedo-gate --force \
  "AUTH_SERVICE_URL=https://doseedo-api.fly.dev" \
  "INTERNAL_SECRET=$(flyctl ssh console --app doseedo-api -C 'printenv INTERNAL_SECRET' | tail -1)" \
  "DISABLE_ALL_GENERATION=true"
cd ~/Downloads/Do/modal && modal deploy modal_stemphonic.py
```
All gated routes return `503 {"error":"Generation is temporarily disabled"}` within ~1 minute. Revert by setting `DISABLE_ALL_GENERATION=false` and redeploying.
