/**
 * PresenceAvatars — horizontal row of peer avatars in the studio menubar.
 *
 * Driven by `useCollabPresence(sessionId, username)`. Renders up to 5 visible
 * avatars + "+N" overflow chip. Each avatar is a colored circle showing the
 * peer's initials; color is seeded from the user_id (or username) so the
 * same person always paints the same hue across reloads.
 *
 * Self avatar is rendered separately by StudioDev.js (the existing
 * `.wb-user-avatar` <img>), so this component only shows OTHER peers.
 *
 * Tooltip: "<username> · <role>"; clicking does nothing for now (could
 * later open a "follow cursor" / DM panel).
 */
import React, { useMemo } from 'react';

const MAX_VISIBLE = 5;

/** Stable hash → 0..359 hue. Seeded from user_id (preferred) or username. */
function _hueFor(seed) {
  const s = String(seed || '');
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h) % 360;
}

function _initialsFor(name) {
  const s = String(name || '').trim();
  if (!s) return '?';
  const parts = s.split(/\s+/).filter(Boolean);
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function PeerCircle({ peer, size = 22 }) {
  const seed = peer.user_id != null ? `u${peer.user_id}` : peer.username || peer.source_id;
  const hue = _hueFor(seed);
  const initials = _initialsFor(peer.username || peer.source_id);
  const tooltip = `${peer.username || 'Anonymous'} · ${peer.role || 'view'}`;
  return (
    <div
      title={tooltip}
      className="wb-presence-avatar"
      style={{
        width: size,
        height: size,
        borderRadius: '50%',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: 9,
        fontWeight: 600,
        letterSpacing: 0.3,
        color: 'rgba(0,0,0,0.78)',
        background: `hsl(${hue}, 70%, 72%)`,
        border: peer.role === 'edit'
          ? '1px solid hsl(' + hue + ', 70%, 32%)'
          : '1px solid var(--wb-ink-mute)',
        marginLeft: -4,
        boxShadow: '0 0 0 1px var(--wb-surface)',
        userSelect: 'none',
        flex: '0 0 auto',
      }}
    >
      {initials}
    </div>
  );
}

function OverflowChip({ count, hiddenPeers, size = 22 }) {
  const tooltip = hiddenPeers
    .map((p) => `${p.username || 'Anonymous'} (${p.role || 'view'})`)
    .join('\n');
  return (
    <div
      title={tooltip}
      className="wb-presence-avatar wb-presence-avatar--overflow"
      style={{
        width: size,
        height: size,
        borderRadius: '50%',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: 9,
        fontWeight: 600,
        color: 'var(--wb-ink)',
        background: 'var(--wb-surface)',
        border: '1px solid var(--wb-ink-mute)',
        marginLeft: -4,
        userSelect: 'none',
        flex: '0 0 auto',
      }}
    >
      +{count}
    </div>
  );
}

export default function PresenceAvatars({ peers = [], connected = false }) {
  const sorted = useMemo(() => {
    return [...peers].sort((a, b) => {
      // Stable: edit-roles first, then by joined_at.
      if (a.role !== b.role) return a.role === 'edit' ? -1 : 1;
      const aT = a.joined_at ? Date.parse(a.joined_at) : 0;
      const bT = b.joined_at ? Date.parse(b.joined_at) : 0;
      return aT - bT;
    });
  }, [peers]);

  if (!sorted.length) {
    // Render nothing when no peers — avoids menubar clutter.
    return null;
  }

  const visible = sorted.slice(0, MAX_VISIBLE);
  const hidden = sorted.slice(MAX_VISIBLE);

  return (
    <div
      className="wb-presence-row"
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        marginLeft: 6,
        opacity: connected ? 1 : 0.5,
      }}
      aria-label={`${sorted.length} other ${sorted.length === 1 ? 'collaborator' : 'collaborators'} in this session`}
    >
      {visible.map((p) => (
        <PeerCircle key={p.source_id} peer={p} />
      ))}
      {hidden.length > 0 && <OverflowChip count={hidden.length} hiddenPeers={hidden} />}
    </div>
  );
}
