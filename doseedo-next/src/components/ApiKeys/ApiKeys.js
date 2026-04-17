import React, { useState, useEffect, useCallback } from 'react';
import styles from './ApiKeys.module.css';

const API_BASE = 'https://doseedo.com';

const VALID_SCOPES = [
  { value: 'doo', label: 'doo', desc: 'Desktop assistant (full access)' },
  { value: 'studio', label: 'studio', desc: 'Web studio access' },
  { value: 'read:sessions', label: 'read:sessions', desc: 'Read your sessions' },
  { value: 'write:sessions', label: 'write:sessions', desc: 'Create & edit sessions' },
  { value: 'read:profile', label: 'read:profile', desc: 'Read profile info' },
];

async function apiFetch(path, opts = {}) {
  const res = await fetch(API_BASE + path, {
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (res.status === 401) {
    const e = new Error('Not authenticated');
    e.status = 401;
    throw e;
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  if (res.status === 204) return null;
  return res.json();
}

function formatDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  const now = new Date();
  const diff = now - d;
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  if (diff < 7 * 86400000) return `${Math.floor(diff / 86400000)}d ago`;
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

export default function ApiKeys() {
  const [keys, setKeys] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [notAuthed, setNotAuthed] = useState(false);

  // Create form state
  const [keyName, setKeyName] = useState('');
  const [keyExpiry, setKeyExpiry] = useState('');
  const [selectedScopes, setSelectedScopes] = useState(['doo']);
  const [creating, setCreating] = useState(false);

  // Modals
  const [revealKey, setRevealKey] = useState(null); // { id, name, key, scopes }
  const [revokeTarget, setRevokeTarget] = useState(null); // { id, name }
  const [copied, setCopied] = useState(false);
  const [revoking, setRevoking] = useState(false);

  // Toast
  const [toast, setToast] = useState(null);

  const showToast = useCallback((msg, type = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  }, []);

  const loadKeys = useCallback(async () => {
    setLoading(true);
    setError(null);
    setNotAuthed(false);
    try {
      const data = await apiFetch('/api/keys');
      setKeys(data);
    } catch (e) {
      if (e.status === 401) {
        setNotAuthed(true);
      } else {
        setError(e.message);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadKeys();
  }, [loadKeys]);

  const toggleScope = (scope) => {
    setSelectedScopes(prev =>
      prev.includes(scope) ? prev.filter(s => s !== scope) : [...prev, scope]
    );
  };

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!keyName.trim()) { showToast('Give your key a name first', 'error'); return; }
    if (!selectedScopes.length) { showToast('Select at least one scope', 'error'); return; }

    setCreating(true);
    try {
      const body = { name: keyName.trim(), scopes: selectedScopes };
      if (keyExpiry) body.expires_in_days = parseInt(keyExpiry);
      const created = await apiFetch('/api/keys', { method: 'POST', body: JSON.stringify(body) });
      setRevealKey(created);
      setKeyName('');
      setKeyExpiry('');
      setSelectedScopes(['doo']);
      loadKeys();
    } catch (e) {
      showToast(e.message, 'error');
    } finally {
      setCreating(false);
    }
  };

  const handleCopy = async () => {
    if (!revealKey?.key) return;
    await navigator.clipboard.writeText(revealKey.key);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const handleRevoke = async () => {
    if (!revokeTarget) return;
    setRevoking(true);
    try {
      await apiFetch(`/api/keys/${revokeTarget.id}`, { method: 'DELETE' });
      setRevokeTarget(null);
      showToast('Key revoked');
      loadKeys();
    } catch (e) {
      showToast(e.message, 'error');
    } finally {
      setRevoking(false);
    }
  };

  const activeKeys = keys.filter(k => k.is_active && !k.revoked_at);

  if (notAuthed) {
    return (
      <div className={styles.container}>
        <div className={styles.header}>
          <div>
            <h1 className={styles.title}>API Keys</h1>
          </div>
        </div>
        <div className={styles.card} style={{ textAlign: 'center', padding: '3rem 2rem' }}>
          <i className="fa-solid fa-lock" style={{ fontSize: 32, opacity: 0.4, marginBottom: 16, display: 'block' }} />
          <h3 style={{ marginBottom: '0.5rem' }}>Sign in required</h3>
          <p style={{ color: 'rgba(255,255,255,0.5)', marginBottom: '1.5rem' }}>
            Please log in to manage your API keys.
          </p>
          <a href="https://doseedo.com/login" className={styles.btnPrimary} style={{ textDecoration: 'none', display: 'inline-flex' }}>
            <i className="fa-solid fa-arrow-right-to-bracket" /> Go to Login
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>API Keys</h1>
          <p className={styles.subtitle}>
            Manage programmatic access to Doseedo — including the <strong>doo</strong> desktop assistant.
          </p>
        </div>
      </div>

      {/* Info banner */}
      <div className={styles.infoBanner}>
        <i className="fa-solid fa-circle-info" style={{ flexShrink: 0, marginTop: 1 }} />
        <span>
          The full key is shown <strong>once</strong> at creation. Copy it to a safe place — we only store a secure hash.
        </span>
      </div>

      {/* Create form */}
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <span className={styles.cardTitle}>
            <i className="fa-solid fa-plus" />
            Create New Key
          </span>
        </div>
        <form className={styles.createForm} onSubmit={handleCreate}>
          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label className={styles.label}>Key Name</label>
              <input
                className={styles.input}
                type="text"
                placeholder="e.g. doo desktop, home studio…"
                value={keyName}
                onChange={e => setKeyName(e.target.value)}
                maxLength={128}
              />
            </div>
            <div className={styles.formGroup}>
              <label className={styles.label}>Expires In</label>
              <select
                className={styles.input}
                value={keyExpiry}
                onChange={e => setKeyExpiry(e.target.value)}
              >
                <option value="">Never</option>
                <option value="30">30 days</option>
                <option value="90">90 days</option>
                <option value="180">180 days</option>
                <option value="365">1 year</option>
              </select>
            </div>
          </div>

          <div className={styles.formGroup}>
            <label className={styles.label}>Scopes</label>
            <div className={styles.scopeGrid}>
              {VALID_SCOPES.map(s => (
                <label key={s.value} className={styles.scopeOption}>
                  <input
                    type="checkbox"
                    checked={selectedScopes.includes(s.value)}
                    onChange={() => toggleScope(s.value)}
                  />
                  <div className={styles.scopeInfo}>
                    <code className={styles.scopeCode}>{s.label}</code>
                    <span className={styles.scopeDesc}>{s.desc}</span>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <div className={styles.formFooter}>
            {activeKeys.length >= 10 && (
              <span className={styles.limitMsg}>Limit: 10 active keys reached</span>
            )}
            <button
              type="submit"
              className={styles.btnPrimary}
              disabled={creating || activeKeys.length >= 10}
            >
              {creating ? (
                <><span className={styles.spinner} /> Generating…</>
              ) : (
                <><i className="fa-solid fa-key" /> Generate Key</>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Keys list */}
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <span className={styles.cardTitle}>
            <i className="fa-solid fa-key" />
            Your Keys
          </span>
          <span className={styles.keyCount}>
            {activeKeys.length} active / {keys.length} total
          </span>
        </div>

        {loading ? (
          <div className={styles.loadingCenter}>
            <span className={styles.spinner} />
          </div>
        ) : error ? (
          <div className={styles.emptyState}>
            <p style={{ color: '#fc8181' }}>{error}</p>
          </div>
        ) : keys.length === 0 ? (
          <div className={styles.emptyState}>
            <i className="fa-solid fa-key" style={{ fontSize: 28, opacity: 0.25, marginBottom: 12 }} />
            <h3>No API keys yet</h3>
            <p>Create your first key above to get started.</p>
          </div>
        ) : (
          <div className={styles.keysList}>
            {keys.map(key => {
              const isRevoked = !!key.revoked_at;
              const isExpired = key.expires_at && new Date(key.expires_at) < new Date();
              const status = isRevoked ? 'revoked' : isExpired ? 'expired' : 'active';

              return (
                <div key={key.id} className={`${styles.keyRow} ${status !== 'active' ? styles.keyRowDimmed : ''}`}>
                  <div className={styles.keyInfo}>
                    <div className={styles.keyNameRow}>
                      <span className={styles.keyName}>{key.name}</span>
                      <code className={styles.keyPrefix}>{key.key_prefix}…</code>
                      <span className={`${styles.badge} ${styles[`badge_${status}`]}`}>{status}</span>
                    </div>
                    <div className={styles.keyScopes}>
                      {(key.scopes || []).map(s => (
                        <span key={s} className={styles.scopeTag}>{s}</span>
                      ))}
                    </div>
                    <div className={styles.keyMeta}>
                      <span>Created {formatDate(key.created_at)}</span>
                      <span>Last used: {key.last_used_at ? formatDate(key.last_used_at) : 'Never'}</span>
                      <span>Expires: {key.expires_at ? formatDate(key.expires_at) : 'Never'}</span>
                    </div>
                  </div>
                  {status === 'active' && (
                    <button
                      className={styles.btnDanger}
                      onClick={() => setRevokeTarget({ id: key.id, name: key.name })}
                    >
                      Revoke
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Usage */}
      <div className={styles.card} style={{ padding: '1.25rem 1.5rem' }}>
        <div className={styles.cardTitle} style={{ marginBottom: '0.75rem', fontSize: '0.85rem' }}>
          <i className="fa-solid fa-code" />
          Usage
        </div>
        <pre className={styles.codeBlock}>{`# doo desktop — add to ~/.doo/.env
DOO_API_KEY=dsk_live_your_key_here

# API calls — Bearer token
curl https://doseedo.com/api/sessions \\
  -H "Authorization: Bearer dsk_live_your_key_here"

# Or X-API-Key header
curl https://doseedo.com/api/sessions \\
  -H "X-API-Key: dsk_live_your_key_here"`}</pre>
      </div>

      {/* ── Modal: reveal new key ── */}
      {revealKey && (
        <div className={styles.overlay} onClick={() => setRevealKey(null)}>
          <div className={styles.modal} onClick={e => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <div>
                <div className={styles.modalTitle}>🎉 Key created</div>
                <div className={styles.modalSubtitle}>{revealKey.name}</div>
              </div>
              <button className={styles.closeBtn} onClick={() => setRevealKey(null)}>✕</button>
            </div>

            <div className={styles.warningBox}>
              <i className="fa-solid fa-triangle-exclamation" />
              <span>Copy this key now — it won't be shown again. We only store a secure hash.</span>
            </div>

            <div className={styles.keyReveal}>
              <div className={styles.keyRevealLabel}>Your new API key</div>
              <div className={styles.keyRevealValue}>{revealKey.key}</div>
              <div className={styles.keyRevealFooter}>
                <span className={styles.scopesList}>Scopes: {(revealKey.scopes || []).join(', ')}</span>
                <button
                  className={`${styles.copyBtn} ${copied ? styles.copyBtnCopied : ''}`}
                  onClick={handleCopy}
                >
                  {copied ? '✓ Copied!' : 'Copy key'}
                </button>
              </div>
            </div>

            <button
              className={styles.btnGhost}
              style={{ width: '100%', justifyContent: 'center' }}
              onClick={() => setRevealKey(null)}
            >
              I've saved my key
            </button>
          </div>
        </div>
      )}

      {/* ── Modal: confirm revoke ── */}
      {revokeTarget && (
        <div className={styles.overlay} onClick={() => !revoking && setRevokeTarget(null)}>
          <div className={styles.modal} onClick={e => e.stopPropagation()}>
            <div className={styles.confirmIcon}>🗑️</div>
            <div className={styles.modalTitle} style={{ marginBottom: '0.5rem' }}>Revoke API key?</div>
            <p style={{ fontSize: '0.875rem', color: 'rgba(255,255,255,0.6)', marginBottom: '1.5rem' }}>
              "<span style={{ color: '#fff' }}>{revokeTarget.name}</span>" will be immediately invalidated.
              Any services using it will stop working.
            </p>
            <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
              <button className={styles.btnGhost} onClick={() => setRevokeTarget(null)} disabled={revoking}>
                Cancel
              </button>
              <button className={styles.btnDangerSolid} onClick={handleRevoke} disabled={revoking}>
                {revoking ? <><span className={styles.spinner} /> Revoking…</> : 'Revoke key'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Toast */}
      {toast && (
        <div className={`${styles.toast} ${toast.type === 'error' ? styles.toastError : styles.toastSuccess}`}>
          {toast.type === 'error' ? '✕ ' : '✓ '}{toast.msg}
        </div>
      )}
    </div>
  );
}
