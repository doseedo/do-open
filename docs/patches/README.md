# Patches pending upstream

These are changes that have already been **shipped to prod** via
`gcloud builds submit` from a local working tree, but haven't made it
back into the appropriate GitHub repo. If they aren't committed to
main on their respective repos, the next deploy from that repo will
silently overwrite prod with the old code.

## step1-gate-env-isolation.patch

**Target repo:** `github.com/doseedo/doseedo-desktop`
**Target file:** `auth-service/app/routers/generation_gate.py`
**Live as of:** Cloud Run revision `doseedo-auth-00057-48q`
  (image tag `doseedo-auth:stg-iso-1776151044`)
**Baseline this patches against:** whatever was in the Mac agent's
  uncommitted local tree when they ran `gcloud builds submit` and
  produced the `doseedo-auth:0fe71dd…` image. That commit never
  reached origin/main — the build was submitted from the working
  directory directly.

### What it does
- Reads `ENVIRONMENT` env var (default `"prod"`). Namespaces the
  Redis quota key: `gen:daily:{ENVIRONMENT}:{user_id}:{date}`.
- Accepts EITHER `INTERNAL_SECRET` or `INTERNAL_SECRET_STAGING` as
  the `X-Internal-Secret` header. Tags the telemetry row with which
  one matched (`caller_env=prod|staging`) and with `cross_env=true`
  if the caller env doesn't match the server's env.
- Adds `env`, `caller_env`, `cross_env` fields to every
  `generation.attempted` row's `properties`.

### How to apply
On the Mac agent's clone of `doseedo-desktop`, from its root:
```bash
# 1. Make sure the local branch already contains the generation_gate.py
#    file that was built into the 0fe71dd image (i.e., the existing
#    uncommitted or unpushed work).
git status auth-service/app/routers/generation_gate.py
# If the file is new/uncommitted, commit it first so the patch has a
# baseline to apply against:
git add auth-service/app/routers/generation_gate.py
git commit -m "Baseline generation_gate from 0fe71dd (pre-env-isolation)"

# 2. Apply the env-isolation patch. The path in the patch is relative
#    to auth-service/ so cd there first.
cd auth-service
git apply /path/to/Do/docs/patches/step1-gate-env-isolation.patch

# 3. Verify — file should now have ENVIRONMENT + _match_internal_secret.
grep -E "ENVIRONMENT|_match_internal_secret|gen:daily:.ENVIRONMENT" \
    app/routers/generation_gate.py

# 4. Commit + push + trigger Cloud Build.
git add app/routers/generation_gate.py
git commit -m "Gate env-isolation: env-namespaced Redis keys + dual-secret accept

Same change already live on prod Cloud Run as revision
doseedo-auth-00057-48q / image tag stg-iso-1776151044, submitted
via gcloud builds submit from the VM-side extracted source. This
commit brings main in sync."
git push origin main

# 5. Post-push verification — the next Cloud Build should be a no-op
#    (same code path as what's already deployed). If it produces a
#    different Cloud Run revision that starts serving differently,
#    the patch didn't apply cleanly.
```

If `git apply` fails because the Mac's local version diverged from
the baseline, `generation_gate.patched.py` in this directory is the
exact current-on-prod content — you can copy it directly into place
as the authoritative version.

### Verifying prod is still right after the push
After the Mac push + Cloud Build, run the 11-shot smoke on a fresh
free-tier user; the 11th request must still 429. Also query
`telemetry_events` and confirm new rows have `properties->>'env'`,
`properties->>'caller_env'`, `properties->>'cross_env'` populated —
if those fields go null the new image doesn't have the patch.

## Why a patch directory exists at all
Cloud Build accepts local source tarballs, which means a commit
doesn't have to exist on GitHub for code to reach prod. That's
convenient but lets source control drift happen silently. Every
entry in this directory is a case where drift happened and is
waiting to be reconciled. When each one's been merged upstream,
delete its patch file and update this README. Empty directory is
the goal state.
