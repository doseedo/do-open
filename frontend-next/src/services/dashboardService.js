/**
 * Dashboard Service
 * Aggregates data from localStorage sessions, cloud session API,
 * and community API for all dashboard sections. Falls back gracefully
 * when backend endpoints are unavailable.
 */

import * as sessionService from './sessionService';
import * as sessionAPI from './sessionAPI';
import * as communityAPI from './communityAPI';

// ── Helpers ──────────────────────────────────────────────────────

function formatRelativeTime(timestamp) {
  const ts = typeof timestamp === 'number' ? timestamp : new Date(timestamp).getTime();
  if (!ts || isNaN(ts)) return '';
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days} days ago`;
  const weeks = Math.floor(days / 7);
  return weeks === 1 ? '1 week ago' : `${weeks} weeks ago`;
}

function formatCount(n) {
  if (typeof n !== 'number') return String(n || 0);
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function countTracks(state) {
  if (!state?.buses) return 0;
  return state.buses.reduce((sum, bus) => sum + (bus.tracks?.length || 0), 0);
}

/** Read a session from localStorage without fully hydrating its state —
 *  we only need top-level metadata for the dashboard cards. */
function readSessionMeta(name) {
  try {
    const raw = localStorage.getItem(`session-${name}`);
    if (!raw) return null;
    const session = JSON.parse(raw);
    return {
      projectName: session.projectName || name,
      timestamp: session.timestamp || 0,
      trackCount: countTracks(session.state),
      daw: session.state?.importSource || session.state?.metadata?.daw || 'Doseedo',
      collabs: session.state?.collaborators || [],
    };
  } catch {
    return null;
  }
}

/** Normalise the various response shapes the backend can return. */
function unwrapList(response) {
  if (Array.isArray(response)) return response;
  if (response?.results) return response.results;
  if (response?.sessions) return response.sessions;
  if (response?.creations) return response.creations;
  if (response?.items) return response.items;
  return [];
}

function authorName(c) {
  const a = c.author || c.username || c.user || c.owner;
  if (!a) return 'Anonymous';
  return typeof a === 'object' ? (a.username || a.name || 'Anonymous') : a;
}

// ── Jump Back In ─────────────────────────────────────────────────

export async function getRecentSessions() {
  const byName = new Map(); // dedup local + cloud by name

  // 1. Local sessions (always available, fast)
  try {
    const projects = sessionService.getProjects();
    for (const name of projects) {
      const meta = readSessionMeta(name);
      if (!meta) continue;
      byName.set(meta.projectName, {
        id: meta.projectName,
        name: meta.projectName,
        daw: meta.daw,
        time: formatRelativeTime(meta.timestamp),
        collabs: meta.collabs,
        _ts: meta.timestamp,
        trackCount: meta.trackCount,
        source: 'local',
      });
    }
  } catch (e) {
    console.warn('[dashboard] local sessions:', e.message);
  }

  // 2. Cloud sessions (may fail — that's OK)
  try {
    const res = await sessionAPI.getUserSessions();
    const list = unwrapList(res);
    for (const s of list) {
      const name = s.name || s.session_id || s.id;
      if (byName.has(name)) continue; // local copy wins
      const ts = s.updated_at ? new Date(s.updated_at).getTime()
               : s.created_at ? new Date(s.created_at).getTime() : 0;
      byName.set(name, {
        id: s.id || s.session_id || name,
        name,
        daw: s.metadata?.daw || 'Doseedo',
        time: formatRelativeTime(ts),
        collabs: s.collaborators || [],
        _ts: ts,
        trackCount: s.metadata?.trackCount || s.metadata?.fileCount || 0,
        source: 'cloud',
      });
    }
  } catch {
    // Cloud sessions unavailable — local-only is fine
  }

  const sessions = [...byName.values()];
  sessions.sort((a, b) => b._ts - a._ts);
  return sessions.slice(0, 8);
}

// ── Activity Feed ────────────────────────────────────────────────
// Aggregates from community creations. A dedicated /api/activity
// endpoint would be better long-term; for now we synthesise events
// from the creations feed.

export async function getActivityFeed() {
  try {
    const res = await communityAPI.listCreations({ sort: '-created_at', limit: 15 });
    const items = unwrapList(res);

    return items.slice(0, 10).map(c => {
      const who = authorName(c);
      const title = c.title || c.name || 'Untitled';
      let action;
      if (c.fork_of || c.forked_from) action = `forked "${title}"`;
      else action = `published "${title}"`;

      return { who, action, time: formatRelativeTime(c.created_at), id: c.id };
    });
  } catch {
    return [];
  }
}

// ── Live Now (Presence) ──────────────────────────────────────────
// No WebSocket presence backend yet. Returns empty array so the UI
// shows the placeholder state. When the backend is ready, this
// function will call the presence endpoint.

export async function getLiveUsers() {
  return [];
}

// ── Trending ─────────────────────────────────────────────────────

const TRENDING_SORT = {
  trending: '-likes_count',
  new: '-created_at',
  most_forked: '-forks_count',
  your_genre: '-created_at', // personalised sorting later
};

export async function getTrending(filter = 'trending') {
  // Try community creations first (has social stats)
  try {
    const params = { limit: 6, sort: TRENDING_SORT[filter] || '-likes_count' };
    const res = await communityAPI.listCreations(params);
    const items = unwrapList(res);
    if (items.length > 0) {
      return items.slice(0, 6).map(mapTrendingItem);
    }
  } catch {
    // Fall through to public sessions
  }

  // Fallback: public sessions
  try {
    const res = await sessionAPI.getPublicSessions({ limit: 6 });
    const items = unwrapList(res);
    return items.slice(0, 6).map(s => ({
      id: s.id || s.session_id,
      name: s.name || s.title || 'Untitled',
      creator: authorName(s),
      tags: s.tags || s.genres || [],
      plays: formatCount(s.plays_count || s.downloads_count || 0),
      forks: s.forks_count || 0,
      stems: s.metadata?.trackCount || 0,
      daw: s.metadata?.daw || 'Doseedo',
    }));
  } catch {
    return [];
  }
}

function mapTrendingItem(c) {
  return {
    id: c.id,
    name: c.title || c.name || 'Untitled',
    creator: authorName(c),
    tags: Array.isArray(c.tags) ? c.tags
        : Array.isArray(c.genres) ? c.genres
        : typeof c.genre === 'string' ? [c.genre] : [],
    plays: formatCount(c.plays_count || c.downloads_count || 0),
    forks: c.forks_count || 0,
    stems: c.stems_count || c.metadata?.trackCount || 0,
    daw: c.metadata?.daw || 'Doseedo',
  };
}

// ── Made for You ─────────────────────────────────────────────────
// Until a recommendation engine exists, we show recent public
// creations with a rotating reason label.

const MFY_REASONS = [
  'Similar timbre to your recent work',
  'Popular with producers you follow',
  'Matches your recent sessions',
  'Trending in your genre',
  'New from a creator you liked',
  'Stems that fit your projects',
  'Based on your listening history',
  'Recommended for you',
];

export async function getMadeForYou() {
  try {
    const res = await communityAPI.listCreations({ limit: 8, sort: '-created_at' });
    const items = unwrapList(res);
    return items.slice(0, 4).map((c, i) => ({
      id: c.id,
      name: c.title || c.name || 'Untitled',
      reason: MFY_REASONS[i % MFY_REASONS.length],
    }));
  } catch {
    return [];
  }
}

// ── Your Week ────────────────────────────────────────────────────
// Computed entirely from localStorage — no backend call needed.

export function getWeekStats() {
  const projects = sessionService.getProjects();
  const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000;

  let sessionsEdited = 0;
  let totalTracks = 0;
  const collabSet = new Set();

  for (const name of projects) {
    const meta = readSessionMeta(name);
    if (!meta || meta.timestamp < cutoff) continue;
    sessionsEdited++;
    totalTracks += meta.trackCount;
    if (Array.isArray(meta.collabs)) {
      meta.collabs.forEach(c => collabSet.add(c));
    }
  }

  // Rough studio-time heuristic: ~30 min per session edit
  const estMins = sessionsEdited * 30;
  const h = Math.floor(estMins / 60);
  const m = estMins % 60;
  const timeStr = h > 0 ? `${h}h ${m}m` : `${m}m`;

  return [
    { value: String(sessionsEdited), label: 'Sessions edited' },
    { value: String(totalTracks), label: 'Stems generated' },
    { value: String(collabSet.size), label: 'Collaborators active' },
    { value: timeStr, label: 'Time in studio' },
  ];
}
