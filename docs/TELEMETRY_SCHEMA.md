# Telemetry Schema

Ground-truth record of what the auth-service gate router
(`auth-service/app/routers/generation_gate.py`) currently writes to
the `telemetry_events` Postgres table. This doc is the canonical
spec — an earlier design note referred to `modal.call_attempted`
with `bucket`/`user_tier` properties; that was never implemented and
was never in the code path. If a dashboard or alert references those
names, update it to the schema below or stop relying on it.

## Table
`telemetry_events` (see `auth-service/app/models.py` for the SQLAlchemy
definition). Indexed on `(event, created_at)`.

| column | type | notes |
|---|---|---|
| `id` | uuid | generated server-side |
| `event` | text | see event names below |
| `user_id` | text | nullable for system events |
| `properties` | jsonb | event-specific payload |
| `created_at` | timestamptz | utc |

## Generation gate events

One row is written on every call to `/internal/generation/consume`
(i.e., the Modal `before_request` hook fires this for every gated
generation route).

### `event = "generation.attempted"`

Written four ways from `generation_gate.py`:

| `properties` shape | when |
|---|---|
| `{ "gated": true,  "reason": "kill_switch", "endpoint": "<route>" }` | `DISABLE_ALL_GENERATION=true` in the gate env |
| `{ "gated": false, "tier": "<plan>", "endpoint": "<route>", "unlimited": true }` | paid / unlimited tiers |
| `{ "gated": true,  "tier": "<plan>", "endpoint": "<route>" }` | free-tier daily quota exhausted |
| `{ "gated": false, "tier": "<plan>", "endpoint": "<route>" }` | free-tier under quota — the 200 path |

**All four rows share:** `event="generation.attempted"`, `user_id=<jwt sub>`, `properties.gated`, `properties.endpoint`. Non-uniform fields: `tier` (every row except the kill-switch path), `reason` (only on kill-switch), `unlimited` (only on paid/unlimited tier).

### Common query recipes

```sql
-- 11-shot smoke test for a user
SELECT created_at,
       properties->>'gated'    AS gated,
       properties->>'endpoint' AS endpoint,
       properties->>'tier'     AS tier,
       properties->>'reason'   AS reason
  FROM telemetry_events
 WHERE event='generation.attempted'
   AND user_id='<uid>'
   AND created_at > now() - interval '1 hour'
 ORDER BY created_at;

-- Gate efficacy in the last day (blocked vs served)
SELECT properties->>'endpoint' AS endpoint,
       properties->>'gated'    AS gated,
       COUNT(*)
  FROM telemetry_events
 WHERE event='generation.attempted'
   AND created_at > now() - interval '24 hours'
 GROUP BY 1, 2
 ORDER BY 1, 2;
```

## Fields that are NOT emitted

For anyone carrying over old diagrams or prose:

- No `bucket` key — the quota key shape is not exposed in telemetry (it's always `gen:daily:{user_id}:{YYYY-MM-DD}` in Redis and is implicit)
- No `user_tier` key — the property is `tier`
- No `modal.call_attempted` event — the name is `generation.attempted`
- No per-call `remaining_quota` or `limit` — those surface to the client in the 429 response body but aren't logged to telemetry

## If you need to rename these

The code is in `auth-service/app/routers/generation_gate.py` (commit
`0fe71dd` introduced it). If any rename lands, also bump this doc
in the same commit so the two sources of truth don't drift again.
