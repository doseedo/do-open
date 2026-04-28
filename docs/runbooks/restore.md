# Neon Postgres restore runbook

The auth-service stores everything in a single Neon Postgres database
(`bassify` on project `doseedo-db` — see `auth-service/fly.toml` for the
exact connection name). Neon offers point-in-time-restore (PITR) via
branches; this runbook is the procedure to recover from accidental
schema or data loss.

## Retention

- **Free tier**: 1-day PITR window.
- **Launch / Scale tiers**: 7-day or 30-day, configurable in the Neon
  console under *Project → Settings → Storage*.

Verify the current setting periodically. A 1-day window will not save
you from a Friday-night DELETE that goes unnoticed until Monday.

```
# Open https://console.neon.tech/app/projects → doseedo-db → Settings → Storage
```

## When to restore

Restore from a branch when ANY of:

1. A bad migration ran in production and left rows in an inconsistent
   state. (Alembic `downgrade` is preferable when it works — only
   restore if `downgrade` itself is broken or destructive.)
2. A bulk DELETE / UPDATE went out without a WHERE clause.
3. Application logic deleted user data we needed to keep.
4. A compromised credential was used to mutate data — restore to a
   point before the breach window.

Do **not** restore for: a single user asking to undo their own action
(use app-level soft-delete restoration instead), or for read-replica
issues (Neon branches are independent compute, restart instead).

## The procedure

### 1. Snapshot the bad state first

Before anything else, create a Neon branch from the *current* state.
This preserves the bad data for forensic review and means a wrong
restore decision is itself reversible.

```
neon branches create --project-id <project-id> --name "incident-$(date +%Y%m%d-%H%M)"
```

(Or via the console: *Branches → Create branch → from "Current state"*.)

### 2. Identify the target restore time

PITR resolution is 1 second. Pick a time *just before* the bad write:
inspect logs (`fly logs --app doseedo-api`, Sentry events) to bracket
the incident. **When in doubt, go earlier** — you can always replay
correct writes; you cannot un-restore.

### 3. Create a restore branch

```
neon branches create \
  --project-id <project-id> \
  --name "restore-$(date +%Y%m%d-%H%M)" \
  --parent main \
  --pitr <ISO 8601 timestamp>
```

(Or via the console: *Branches → Create branch → from "Time travel"
→ pick timestamp*.)

This produces a fresh compute with the database state from that exact
moment.

### 4. Verify the restore branch

Connect to the restore branch (its own connection string in the Neon
console) and check the rows are the way you want them:

```
psql "<restore-branch-connection-string>"
\dt
SELECT count(*) FROM users;
SELECT count(*) FROM api_keys WHERE is_active;
-- spot-check the specific rows you expect to be back
```

### 5. Promote the restore branch to primary

The cleanest cutover is to point the Fly app at the restore branch and
demote `main`:

```
fly secrets set DATABASE_URL="<restore-branch-connection-string>" --app doseedo-api
fly deploy --app doseedo-api --strategy immediate
```

Then in the Neon console: rename the old `main` to `main-bad-<date>`
and rename the restore branch to `main`. The `DATABASE_URL` keeps
working because Neon connection strings are branch-scoped, not
name-scoped.

### 6. Replay correct writes (if any)

If there were legitimate writes between the restore point and now,
replay them by hand. Inspect:

- `fly logs` for the affected window (POST/PUT/PATCH/DELETE)
- Sentry breadcrumbs
- Application audit tables (if any)

There is no automated replay. Only do this if the writes are
recoverable from logs.

### 7. Tell users

If the restore lost legitimate data, the affected users need to know.
Don't be silent — they'll notice eventually and silence reads as
cover-up.

## Test the restore

> "We have backups" without a tested restore is "we have files of
> unknown utility."

Once a quarter:

1. Pick a recent timestamp (5 minutes ago is fine).
2. Run steps 3–4 above into a throwaway branch.
3. Verify the data round-trips cleanly.
4. Delete the test branch.

Note the date of the last successful test in this file so the next
person on call can see when it was last verified:

- Last verified restore: **(never — schedule one)**

## Connection-string handoff to apps

The Fly secret `DATABASE_URL` is the only place the connection string
lives in production. Rotating it requires a `fly deploy` because the
auth-service reads it once at startup (see `app/database.py`).

Modal apps don't talk to Postgres directly today — only the Fly
auth-service does. If that changes, add the Modal secret here and
update this runbook.
