/**
 * ShareModal — manage share-link invites for the active session.
 *
 * Two paths:
 *   • "Copy collab link (existing collaborators only)" — the legacy URL
 *     `${origin}/studio?session=<sid>`. Recipient must already be on the
 *     session's ACL (they were a peer at some prior point), or they must
 *     also accept a share-token URL — without one they can only open
 *     sessions they own.
 *   • Invite link — owner mints a `dsk_live_`-style share token via
 *     `POST /api/sessions/{sid}/share` (sharesAPI.js), with role+expiry,
 *     then shares the resulting URL `${origin}/studio?session=<sid>&share_token=<hex>`.
 *     The /studio page already reads `?share_token=` (useSessionSync.js), so
 *     the recipient can view/edit per the role baked into the token.
 *
 * Owner-only management: the create form + revoke buttons are visible only
 * when `GET /api/sessions/{sid}/shares` returns 200 (server enforces
 * ownership). Non-owners see only the legacy "copy" path.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  listShares,
  createShare,
  revokeShare,
  buildInviteUrl,
} from '../../services/sharesAPI';

const EXPIRY_OPTIONS = [
  { label: '1 hour',  hours: 1 },
  { label: '24 hours', hours: 24 },
  { label: '7 days',  hours: 24 * 7 },
  { label: '30 days', hours: 24 * 30 },
  { label: 'Never',   hours: null },
];

function _fmtExpiry(iso) {
  if (!iso) return 'Never expires';
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return 'Unknown';
  const ms = t - Date.now();
  if (ms <= 0) return 'Expired';
  const h = Math.round(ms / 3_600_000);
  if (h < 24) return `Expires in ${h}h`;
  const d = Math.round(h / 24);
  return `Expires in ${d}d`;
}

function _isActive(share) {
  if (share.revoked_at) return false;
  if (share.expires_at) {
    const t = Date.parse(share.expires_at);
    if (Number.isFinite(t) && t <= Date.now()) return false;
  }
  return true;
}

async function _copy(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try { window.prompt('Copy this link:', text); } catch {}
    return false;
  }
}

export default function ShareModal({ sessionId, onClose }) {
  const [shares, setShares] = useState([]);
  const [canManage, setCanManage] = useState(false);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [role, setRole] = useState('view');
  const [expiry, setExpiry] = useState(EXPIRY_OPTIONS[1]); // 24h default
  const [copiedToken, setCopiedToken] = useState(null);
  const [legacyCopied, setLegacyCopied] = useState(false);
  const [error, setError] = useState(null);

  // Initial load: probe ownership via listShares.
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    listShares(sessionId)
      .then((rows) => {
        if (cancelled) return;
        setCanManage(true);
        setShares(Array.isArray(rows) ? rows : []);
      })
      .catch((e) => {
        if (cancelled) return;
        if (e?.status === 403 || e?.status === 401) {
          setCanManage(false);
          setShares([]);
        } else {
          setError(e?.message || 'Could not load share links');
          setCanManage(false);
        }
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [sessionId]);

  const handleCreate = useCallback(async () => {
    if (!sessionId || creating) return;
    setCreating(true);
    setError(null);
    try {
      const row = await createShare(sessionId, {
        role,
        expiresInHours: expiry.hours,
      });
      setShares((prev) => [row, ...prev]);
      setCopiedToken(row.token);
      await _copy(buildInviteUrl(sessionId, row.token));
      setTimeout(() => setCopiedToken((cur) => (cur === row.token ? null : cur)), 2400);
    } catch (e) {
      setError(e?.message || 'Could not create invite link');
    } finally {
      setCreating(false);
    }
  }, [sessionId, role, expiry, creating]);

  const handleRevoke = useCallback(async (token) => {
    if (!sessionId || !token) return;
    if (!window.confirm('Revoke this invite link? Anyone using it will lose access.')) return;
    try {
      await revokeShare(sessionId, token);
      setShares((prev) => prev.map((s) =>
        s.token === token ? { ...s, revoked_at: new Date().toISOString() } : s
      ));
    } catch (e) {
      setError(e?.message || 'Could not revoke invite link');
    }
  }, [sessionId]);

  const handleCopyLegacy = useCallback(async () => {
    if (!sessionId) return;
    const url = `${window.location.origin}/studio?session=${encodeURIComponent(sessionId)}`;
    await _copy(url);
    setLegacyCopied(true);
    setTimeout(() => setLegacyCopied(false), 2000);
  }, [sessionId]);

  // Close on ESC
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose?.(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const activeShares = useMemo(
    () => shares.filter(_isActive),
    [shares]
  );

  const cardStyle = {
    width: 'min(92vw, 560px)',
    maxHeight: '80vh',
    background: 'var(--hifi-surface)',
    border: '1px solid var(--hifi-rule-strong)',
    borderRadius: 8,
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    color: 'var(--hifi-ink, var(--wb-ink))',
  };

  return (
    <div className="sd-overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose?.(); }}>
      <div role="dialog" aria-label="Share session" style={cardStyle}>
        <div className="sd-overlay-head">
          <strong style={{ fontSize: 13, letterSpacing: 0.4, textTransform: 'uppercase' }}>Share</strong>
          <button
            className="wb-menu__item"
            onClick={() => onClose?.()}
            style={{ fontSize: 16, lineHeight: 1, padding: '0 4px' }}
            aria-label="Close share dialog"
          >×</button>
        </div>

        <div style={{ padding: '16px 18px', overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 18 }}>
          {error && (
            <div style={{
              fontSize: 11,
              color: '#ff5757',
              background: 'rgba(255,87,87,0.08)',
              border: '1px solid rgba(255,87,87,0.3)',
              padding: '6px 10px',
              borderRadius: 4,
            }}>{error}</div>
          )}

          {/* Legacy copy path — visible to everyone */}
          <section>
            <div style={{ fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: 'var(--wb-ink-mute)', marginBottom: 6 }}>
              Copy collab link
            </div>
            <div style={{ fontSize: 11, color: 'var(--wb-ink-soft)', marginBottom: 8 }}>
              Direct session URL — only opens for collaborators who already have access.
            </div>
            <button
              className="wb-menu__item"
              onClick={handleCopyLegacy}
              style={{
                padding: '6px 12px',
                border: '1px solid var(--wb-rule-strong)',
                borderRadius: 4,
                fontSize: 11,
                background: 'transparent',
                cursor: 'pointer',
              }}
            >{legacyCopied ? 'Copied ✓' : 'Copy session URL'}</button>
          </section>

          {/* Owner-only: create + manage invite links */}
          {loading && (
            <div style={{ fontSize: 11, color: 'var(--wb-ink-mute)' }}>Loading share links…</div>
          )}
          {!loading && canManage && (
            <>
              <section>
                <div style={{ fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: 'var(--wb-ink-mute)', marginBottom: 6 }}>
                  Create invite link
                </div>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
                  <label style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 6 }}>
                    Role
                    <select
                      value={role}
                      onChange={(e) => setRole(e.target.value)}
                      style={{ fontSize: 11, padding: '3px 6px', background: 'var(--hifi-bg, var(--wb-bg))', color: 'inherit', border: '1px solid var(--wb-rule)', borderRadius: 3 }}
                    >
                      <option value="view">View only</option>
                      <option value="edit">Can edit</option>
                    </select>
                  </label>
                  <label style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 6 }}>
                    Expires
                    <select
                      value={expiry.label}
                      onChange={(e) => {
                        const opt = EXPIRY_OPTIONS.find((o) => o.label === e.target.value);
                        if (opt) setExpiry(opt);
                      }}
                      style={{ fontSize: 11, padding: '3px 6px', background: 'var(--hifi-bg, var(--wb-bg))', color: 'inherit', border: '1px solid var(--wb-rule)', borderRadius: 3 }}
                    >
                      {EXPIRY_OPTIONS.map((opt) => (
                        <option key={opt.label} value={opt.label}>{opt.label}</option>
                      ))}
                    </select>
                  </label>
                  <button
                    onClick={handleCreate}
                    disabled={creating}
                    className="wb-menu__item"
                    style={{
                      padding: '6px 12px',
                      border: '1px solid var(--wb-rule-strong)',
                      borderRadius: 4,
                      fontSize: 11,
                      background: creating ? 'var(--wb-surface-2, var(--wb-surface))' : 'var(--wb-ink)',
                      color: creating ? 'var(--wb-ink-mute)' : 'var(--wb-bg)',
                      cursor: creating ? 'wait' : 'pointer',
                    }}
                  >{creating ? 'Creating…' : 'Create & copy'}</button>
                </div>
                <div style={{ fontSize: 10, color: 'var(--wb-ink-mute)' }}>
                  Send the copied URL to anyone — they'll get {role === 'edit' ? 'edit' : 'view-only'} access.
                </div>
              </section>

              <section>
                <div style={{ fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: 'var(--wb-ink-mute)', marginBottom: 6 }}>
                  Active invite links ({activeShares.length})
                </div>
                {activeShares.length === 0 && (
                  <div style={{ fontSize: 11, color: 'var(--wb-ink-mute)', fontStyle: 'italic' }}>
                    No active invite links.
                  </div>
                )}
                {activeShares.length > 0 && (
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {activeShares.map((s) => {
                      const url = buildInviteUrl(sessionId, s.token);
                      const justCopied = copiedToken === s.token;
                      return (
                        <li
                          key={s.token}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 8,
                            padding: '6px 8px',
                            border: '1px solid var(--wb-rule)',
                            borderRadius: 4,
                            fontSize: 11,
                          }}
                        >
                          <span style={{
                            padding: '1px 6px',
                            border: '1px solid var(--wb-rule-strong)',
                            borderRadius: 3,
                            fontSize: 9,
                            letterSpacing: 0.4,
                            textTransform: 'uppercase',
                          }}>{s.role}</span>
                          <span style={{ color: 'var(--wb-ink-mute)', fontSize: 10 }}>{_fmtExpiry(s.expires_at)}</span>
                          <span style={{ flex: 1, fontFamily: 'var(--wb-font-mono, monospace)', fontSize: 10, color: 'var(--wb-ink-soft)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {`…${s.token.slice(-12)}`}
                          </span>
                          <button
                            onClick={async () => {
                              await _copy(url);
                              setCopiedToken(s.token);
                              setTimeout(() => setCopiedToken((c) => (c === s.token ? null : c)), 2000);
                            }}
                            className="wb-menu__item"
                            style={{
                              padding: '3px 8px',
                              border: '1px solid var(--wb-rule)',
                              borderRadius: 3,
                              fontSize: 10,
                              background: 'transparent',
                              cursor: 'pointer',
                            }}
                          >{justCopied ? 'Copied ✓' : 'Copy'}</button>
                          <button
                            onClick={() => handleRevoke(s.token)}
                            className="wb-menu__item"
                            style={{
                              padding: '3px 8px',
                              border: '1px solid rgba(255,87,87,0.3)',
                              color: '#ff5757',
                              borderRadius: 3,
                              fontSize: 10,
                              background: 'transparent',
                              cursor: 'pointer',
                            }}
                          >Revoke</button>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </section>
            </>
          )}
          {!loading && !canManage && !error && (
            <div style={{ fontSize: 11, color: 'var(--wb-ink-mute)' }}>
              Only the session owner can mint invite links.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
