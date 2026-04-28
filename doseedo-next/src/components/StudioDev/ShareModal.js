/**
 * ShareModal — one share link, one role dropdown.
 *
 * On open: probe ownership via GET /api/sessions/{sid}/shares. If owner,
 * find or mint a 30-day link for the currently-selected role and show its
 * URL with a Copy button. Switching the role dropdown swaps to the
 * matching existing active link, or mints one if none exists. Non-owners
 * see a single "ask the owner" message.
 *
 * Backend contract: createShare ({ role, expiresInHours }) returns
 * { token, role, expires_at, ... }; buildInviteUrl assembles
 * `${origin}/studio?session=<sid>&share_token=<hex>`.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  listShares,
  createShare,
  buildInviteUrl,
} from '../../services/sharesAPI';

const DEFAULT_EXPIRY_HOURS = 24 * 30;

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
  const [busy, setBusy] = useState(false);
  const [role, setRole] = useState('view');
  const [copied, setCopied] = useState(false);
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

  // Pick the most-recent active link matching the selected role; mint one
  // if none exists. Owners only — non-owners can't create.
  const activeForRole = useMemo(() => {
    return shares
      .filter(_isActive)
      .filter((s) => s.role === role)
      .sort((a, b) => Date.parse(b.created_at || 0) - Date.parse(a.created_at || 0))[0] || null;
  }, [shares, role]);

  useEffect(() => {
    if (!sessionId || loading || !canManage || busy || activeForRole) return;
    let cancelled = false;
    setBusy(true);
    setError(null);
    createShare(sessionId, { role, expiresInHours: DEFAULT_EXPIRY_HOURS })
      .then((row) => { if (!cancelled) setShares((prev) => [row, ...prev]); })
      .catch((e) => { if (!cancelled) setError(e?.message || 'Could not create invite link'); })
      .finally(() => { if (!cancelled) setBusy(false); });
    return () => { cancelled = true; };
  }, [sessionId, loading, canManage, role, activeForRole, busy]);

  const url = activeForRole ? buildInviteUrl(sessionId, activeForRole.token) : '';

  const handleCopy = useCallback(async () => {
    if (!url) return;
    await _copy(url);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [url]);

  // Close on ESC
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose?.(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const cardStyle = {
    width: 'min(92vw, 460px)',
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

        <div style={{ padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 14 }}>
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

          {loading && (
            <div style={{ fontSize: 11, color: 'var(--wb-ink-mute)' }}>Loading…</div>
          )}

          {!loading && canManage && (
            <>
              <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <span style={{ fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: 'var(--wb-ink-mute)' }}>
                  People with this link can:
                </span>
                <select
                  value={role}
                  onChange={(e) => { setRole(e.target.value); setCopied(false); }}
                  style={{
                    fontSize: 12,
                    padding: '6px 10px',
                    background: 'var(--hifi-bg, var(--wb-bg))',
                    color: 'inherit',
                    border: '1px solid var(--wb-rule)',
                    borderRadius: 4,
                  }}
                >
                  <option value="view">View only</option>
                  <option value="edit">Edit</option>
                </select>
              </label>

              <div style={{ display: 'flex', gap: 8, alignItems: 'stretch' }}>
                <input
                  type="text"
                  readOnly
                  value={busy && !url ? 'Creating link…' : (url || '—')}
                  onFocus={(e) => e.target.select()}
                  style={{
                    flex: 1,
                    fontFamily: 'var(--wb-font-mono, monospace)',
                    fontSize: 11,
                    padding: '6px 10px',
                    background: 'var(--hifi-bg, var(--wb-bg))',
                    color: 'var(--wb-ink-soft)',
                    border: '1px solid var(--wb-rule)',
                    borderRadius: 4,
                  }}
                />
                <button
                  onClick={handleCopy}
                  disabled={!url || busy}
                  className="wb-menu__item"
                  style={{
                    padding: '6px 14px',
                    border: '1px solid var(--wb-rule-strong)',
                    borderRadius: 4,
                    fontSize: 11,
                    background: !url || busy ? 'var(--wb-surface-2, var(--wb-surface))' : 'var(--wb-ink)',
                    color: !url || busy ? 'var(--wb-ink-mute)' : 'var(--wb-bg)',
                    cursor: !url || busy ? 'wait' : 'pointer',
                  }}
                >{copied ? 'Copied ✓' : 'Copy link'}</button>
              </div>
            </>
          )}

          {!loading && !canManage && !error && (
            <div style={{ fontSize: 11, color: 'var(--wb-ink-mute)' }}>
              Only the session owner can create a share link.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
