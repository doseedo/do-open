/**
 * Per-user rate limiter for Modal-backed API routes.
 *
 * Why: /api/chat and /api/vision are Clerk-gated, but a single signed-in
 * user can still drain the Modal GPU budget (vLLM holds a container warm
 * for ~15min, and one chatty client can run inference back-to-back). We
 * cap each user to a sliding window of requests so a runaway loop or
 * abusive script can't turn into a billing event.
 *
 * Storage: Upstash Redis via REST (serverless-native, no connection
 * pool). The auth-service uses its own Upstash db for collab presence
 * via the redis:// protocol — that's a separate concern.
 *
 * Configuration:
 *   UPSTASH_REDIS_REST_URL   — from Upstash console, e.g. https://xxx.upstash.io
 *   UPSTASH_REDIS_REST_TOKEN — from Upstash console
 *
 * If the env vars are missing the limiter no-ops (allow). That keeps
 * dev/local working without provisioning Upstash, and means a misconfig
 * in Vercel doesn't lock everyone out — failures-open is the right
 * default for a billing-protection limiter (vs. a security limiter,
 * where you'd fail closed).
 */
import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis';

export type RateLimitResult = {
  success: boolean;
  limit: number;
  remaining: number;
  reset: number;
};

let _chatLimiter: Ratelimit | null = null;
let _visionLimiter: Ratelimit | null = null;
let _videoScoreLimiter: Ratelimit | null = null;

function _getRedis(): Redis | null {
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;
  if (!url || !token) return null;
  return new Redis({ url, token });
}

function _getChatLimiter(): Ratelimit | null {
  if (_chatLimiter) return _chatLimiter;
  const redis = _getRedis();
  if (!redis) return null;
  // 60 requests / minute per user — generous for interactive chat,
  // tight enough to bound a runaway script (3600/hour vs unbounded).
  _chatLimiter = new Ratelimit({
    redis,
    limiter: Ratelimit.slidingWindow(60, '1 m'),
    analytics: true,
    prefix: 'rl:chat',
  });
  return _chatLimiter;
}

function _getVisionLimiter(): Ratelimit | null {
  if (_visionLimiter) return _visionLimiter;
  const redis = _getRedis();
  if (!redis) return null;
  // Vision is heavier (image upload + Moondream inference). 20/min/user.
  _visionLimiter = new Ratelimit({
    redis,
    limiter: Ratelimit.slidingWindow(20, '1 m'),
    analytics: true,
    prefix: 'rl:vision',
  });
  return _visionLimiter;
}

export async function checkChatRateLimit(userId: string): Promise<RateLimitResult> {
  const lim = _getChatLimiter();
  if (!lim) return { success: true, limit: 0, remaining: 0, reset: 0 };
  return lim.limit(userId);
}

export async function checkVisionRateLimit(userId: string): Promise<RateLimitResult> {
  const lim = _getVisionLimiter();
  if (!lim) return { success: true, limit: 0, remaining: 0, reset: 0 };
  return lim.limit(userId);
}

function _getVideoScoreLimiter(): Ratelimit | null {
  if (_videoScoreLimiter) return _videoScoreLimiter;
  const redis = _getRedis();
  if (!redis) return null;
  // Video scoring is the heaviest path: 50MB+ upload + scenedetect +
  // multiple Moondream calls per scene. 5/min/user is generous for an
  // interactive studio session and tight enough that an automated upload
  // loop can't burn a Modal day's worth of CPU minutes in seconds.
  _videoScoreLimiter = new Ratelimit({
    redis,
    limiter: Ratelimit.slidingWindow(5, '1 m'),
    analytics: true,
    prefix: 'rl:video-score',
  });
  return _videoScoreLimiter;
}

export async function checkVideoScoreRateLimit(userId: string): Promise<RateLimitResult> {
  const lim = _getVideoScoreLimiter();
  if (!lim) return { success: true, limit: 0, remaining: 0, reset: 0 };
  return lim.limit(userId);
}

let _verifyLimiter: Ratelimit | null = null;
function _getVerifyLimiter(): Ratelimit | null {
  if (_verifyLimiter) return _verifyLimiter;
  const redis = _getRedis();
  if (!redis) return null;
  // /verify is a public utility — keyed by IP, not user id. Tight enough
  // that an automated scanner can't replay all of doseedo's outputs in
  // bulk; loose enough that a curious viewer can drop a couple of files
  // back-to-back without hitting it.
  _verifyLimiter = new Ratelimit({
    redis,
    limiter: Ratelimit.slidingWindow(10, '1 m'),
    analytics: true,
    prefix: 'rl:verify',
  });
  return _verifyLimiter;
}

export async function checkVerifyRateLimit(key: string): Promise<RateLimitResult> {
  const lim = _getVerifyLimiter();
  if (!lim) return { success: true, limit: 0, remaining: 0, reset: 0 };
  return lim.limit(key);
}

export function rateLimitHeaders(r: RateLimitResult): Record<string, string> {
  if (!r.limit) return {};
  return {
    'X-RateLimit-Limit': String(r.limit),
    'X-RateLimit-Remaining': String(r.remaining),
    'X-RateLimit-Reset': String(r.reset),
  };
}
