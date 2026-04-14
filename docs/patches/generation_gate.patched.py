"""
Internal endpoint: POST /internal/generation/consume

Called by the Modal Stemphonic container before executing any GPU
generation task. Validates the caller's JWT, checks/increments their
daily quota in Redis, and writes a telemetry event.

Authentication (dual-secret, for prod↔staging isolation):
  - Caller must present X-Internal-Secret matching EITHER
    INTERNAL_SECRET (prod) or INTERNAL_SECRET_STAGING (staging).
  - The secret that matched determines the "caller_env" tag on the
    request. It does NOT override ENVIRONMENT — that's what the
    server identifies itself as. Cross-env requests (prod secret
    hitting a staging server, or vice versa) are still authorized
    but tagged cross_env=true in telemetry so we can alert on them.
  - User is identified via Authorization: Bearer <jwt> or access_token
    cookie forwarded from the browser request.

Quota:
  - Free tier: FREE_DAILY_CAP generations per UTC day.
  - is_admin=True or generation_tier in ('b2b', 'admin'): unlimited.
  - Redis key: gen:daily:{env}:{user_id}:{YYYY-MM-DD}, TTL 86400s.
    The {env} prefix — driven by the ENVIRONMENT env var (default
    "prod") — keeps staging testing from burning prod user quotas.

Kill switch:
  - DISABLE_ALL_GENERATION=true → 503 for all users immediately.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import redis.asyncio as aioredis
from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Request, status
from jose import JWTError
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.database import get_db
from app.models import TelemetryEvent, User
from app.security import decode_access_token

log = logging.getLogger("generation_gate")

router = APIRouter(prefix="/internal", tags=["internal"])

FREE_DAILY_CAP = 10
UNLIMITED_TIERS = {"b2b", "admin"}

# Environment this gate instance is serving. Controls the Redis key
# namespace (gen:daily:{ENVIRONMENT}:{user_id}:{date}). Defaults to
# "prod" so existing prod deploys that don't set ENVIRONMENT keep
# using the same key format they did before this change.
ENVIRONMENT = os.environ.get("ENVIRONMENT", "prod").lower()


# ---------------------------------------------------------------------------
# Internal auth guard (dual-secret)
# ---------------------------------------------------------------------------

def _match_internal_secret(presented: str | None) -> str | None:
    """Return the env tag (`prod` | `staging`) whose secret matched,
    or None if neither did. Empty env values never match."""
    if not presented:
        return None
    prod_secret    = os.environ.get("INTERNAL_SECRET", "")
    staging_secret = os.environ.get("INTERNAL_SECRET_STAGING", "")
    if prod_secret and presented == prod_secret:
        return "prod"
    if staging_secret and presented == staging_secret:
        return "staging"
    return None


def _require_internal_secret(
    request: Request,
    x_internal_secret: str | None = Header(default=None, alias="X-Internal-Secret"),
) -> str:
    """Return the caller_env of whichever secret validated, or 403."""
    caller_env = _match_internal_secret(x_internal_secret)
    if caller_env is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    # Stash on request.state so the consume handler can tag the
    # telemetry row without re-running the header parse.
    request.state.caller_env = caller_env
    return caller_env


# ---------------------------------------------------------------------------
# Redis helper
# ---------------------------------------------------------------------------

async def _redis() -> aioredis.Redis:
    settings = get_settings()
    url = settings.redis_url
    return aioredis.from_url(url, decode_responses=True, db=1)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ConsumeRequest(BaseModel):
    endpoint: str = "/api/generate-stemphonic"


class ConsumeResponse(BaseModel):
    allowed: bool
    user_id: int
    tier: str
    remaining: int


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/generation/consume", response_model=ConsumeResponse)
async def consume_generation_quota(
    body: ConsumeRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(_require_internal_secret),
    authorization: str | None = Header(default=None),
    access_token: str | None = Cookie(default=None),
):
    """
    Validate the user JWT, check quota, increment counter, write telemetry.

    Returns 200 {allowed: true}  — proceed with generation.
    Returns 429 {allowed: false} — daily cap exceeded.
    Returns 401                  — no valid JWT.
    Returns 503                  — kill switch active.
    """
    # Tags common to every telemetry row emitted from this handler.
    # env       = the server's ENVIRONMENT
    # caller_env= which secret matched (prod | staging)
    # cross_env = caller_env ≠ env (a staging modal forged a prod secret
    #             or vice versa — legal-but-flag-for-alerting case)
    caller_env: str = getattr(request.state, "caller_env", "prod")
    common_tags: dict = {
        "env": ENVIRONMENT,
        "caller_env": caller_env,
        "cross_env": caller_env != ENVIRONMENT,
    }

    # --- kill switch ---
    if os.environ.get("DISABLE_ALL_GENERATION", "false").lower() == "true":
        db.add(TelemetryEvent(
            event="generation.attempted",
            user_id=None,
            properties={"gated": True, "reason": "kill_switch", "endpoint": body.endpoint, **common_tags},
        ))
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generation is temporarily disabled",
        )

    # --- resolve JWT ---
    token: str | None = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
    elif access_token:
        token = access_token

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        payload = decode_access_token(token)
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user_id = int(sub)
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    tier: str = getattr(user, "generation_tier", "free") or "free"
    unlimited = user.is_admin or tier in UNLIMITED_TIERS

    # --- unlimited path ---
    if unlimited:
        db.add(TelemetryEvent(
            event="generation.attempted",
            user_id=user_id,
            properties={"gated": False, "tier": tier, "endpoint": body.endpoint, "unlimited": True, **common_tags},
        ))
        await db.commit()
        return ConsumeResponse(allowed=True, user_id=user_id, tier=tier, remaining=999)

    # --- quota check (free tier) ---
    # Env-namespaced Redis key — staging testing does NOT consume prod
    # user quotas even when they share the same Redis instance. With
    # fully-isolated staging infra this is belt + suspenders, but it
    # means we stay safe if/when auth backends ever get merged again.
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"gen:daily:{ENVIRONMENT}:{user_id}:{date_str}"

    r = await _redis()
    try:
        count_str = await r.get(key)
        count = int(count_str) if count_str else 0

        if count >= FREE_DAILY_CAP:
            db.add(TelemetryEvent(
                event="generation.attempted",
                user_id=user_id,
                properties={
                    "gated": True, "tier": tier, "endpoint": body.endpoint,
                    "count": count, "cap": FREE_DAILY_CAP,
                    **common_tags,
                },
            ))
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Daily generation limit reached ({FREE_DAILY_CAP}/day). Upgrade to continue.",
                headers={"Retry-After": "86400", "X-RateLimit-Remaining": "0"},
            )

        new_count = await r.incr(key)
        if new_count == 1:
            await r.expire(key, 86400)

        remaining = max(0, FREE_DAILY_CAP - new_count)

        db.add(TelemetryEvent(
            event="generation.attempted",
            user_id=user_id,
            properties={
                "gated": False, "tier": tier, "endpoint": body.endpoint,
                "count": new_count, "remaining": remaining,
                **common_tags,
            },
        ))
        await db.commit()

        return ConsumeResponse(allowed=True, user_id=user_id, tier=tier, remaining=remaining)

    finally:
        await r.aclose()
