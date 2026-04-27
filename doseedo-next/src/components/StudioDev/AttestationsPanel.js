/**
 * AttestationsPanel — request / confirm / dispute UI inside a commit row.
 *
 * Renders inline when a user clicks the "Attest" toggle on a History tab
 * row. Keeps state local so opening attestations on one commit doesn't
 * refetch the whole commit list. The parent passes:
 *   - sessionId
 *   - commit (with server fields incl. attestation_total/confirmed/disputed)
 *   - currentUsername (Clerk user — used to surface confirm/dispute
 *     buttons only on attestations naming this user)
 *   - onChanged() — fired after any state-changing action so the parent
 *     can refetch the commit list (Level pill updates).
 */
import React, { useCallback, useEffect, useState } from 'react';
import {
  listAttestations,
  requestAttestation,
  confirmAttestation,
  disputeAttestation,
} from '../../services/attestationsAPI';

export default function AttestationsPanel({ sessionId, commit, currentUsername, onChanged }) {
  const [rows, setRows] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  // "Add contributor" form state
  const [addOpen, setAddOpen] = useState(false);
  const [newName, setNewName] = useState('');
  const [newRole, setNewRole] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const data = await listAttestations(sessionId, commit.id);
      setRows(data || []);
    } catch (e) {
      setErr(e.message || String(e));
      setRows([]);
    } finally {
      setLoading(false);
    }
  }, [sessionId, commit.id]);

  useEffect(() => { refresh(); }, [refresh]);

  const onAdd = async (e) => {
    e.preventDefault();
    if (!newName.trim()) return;
    setSubmitting(true);
    try {
      await requestAttestation(sessionId, commit.id, newName.trim(), newRole.trim() || null);
      setNewName(''); setNewRole(''); setAddOpen(false);
      await refresh();
      onChanged?.();
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setSubmitting(false);
    }
  };

  const onConfirm = async (att) => {
    setSubmitting(true);
    try { await confirmAttestation(sessionId, att.id); await refresh(); onChanged?.(); }
    catch (e) { setErr(e.message || String(e)); }
    finally { setSubmitting(false); }
  };

  const onDispute = async (att) => {
    const reason = prompt('Why are you disputing this attribution?');
    if (!reason || !reason.trim()) return;
    setSubmitting(true);
    try { await disputeAttestation(sessionId, att.id, reason); await refresh(); onChanged?.(); }
    catch (e) { setErr(e.message || String(e)); }
    finally { setSubmitting(false); }
  };

  const statusFor = (att) => {
    if (att.disputed_at) return { label: 'disputed', cls: 'disputed' };
    if (att.confirmed_at) return { label: 'confirmed', cls: 'confirmed' };
    return { label: 'pending', cls: 'pending' };
  };

  return (
    <div className="sd-attest-panel">
      {loading && <div className="sd-side-sub">Loading attestations…</div>}
      {err && <div className="sd-side-sub" style={{ color: 'tomato' }}>{err}</div>}

      {rows && rows.length === 0 && !loading && (
        <div className="sd-side-sub">No contributors named yet.</div>
      )}

      {rows && rows.length > 0 && (
        <ul className="sd-attest-list">
          {rows.map((att) => {
            const s = statusFor(att);
            const mine = currentUsername && att.contributor_username === currentUsername;
            return (
              <li key={att.id} className={`sd-attest-row ${s.cls}`}>
                <div className="sd-attest-name">
                  <strong>{att.contributor_username}</strong>
                  {att.contributor_role && <span className="sd-attest-role"> · {att.contributor_role}</span>}
                </div>
                <span className={`sd-attest-pill sd-attest-pill-${s.cls}`}>{s.label}</span>
                {att.dispute_reason && (
                  <div className="sd-attest-dispute" title={att.dispute_reason}>
                    {att.dispute_reason.length > 60 ? att.dispute_reason.slice(0, 60) + '…' : att.dispute_reason}
                  </div>
                )}
                {mine && (
                  <div className="sd-attest-actions">
                    {!att.confirmed_at && (
                      <button className="sd-btn ghost" disabled={submitting} onClick={() => onConfirm(att)}>
                        Confirm
                      </button>
                    )}
                    {!att.disputed_at && (
                      <button className="sd-btn ghost" disabled={submitting} onClick={() => onDispute(att)}>
                        Dispute
                      </button>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}

      {!addOpen && (
        <button className="sd-btn ghost" onClick={() => setAddOpen(true)} disabled={submitting}>
          <i className="fa-solid fa-user-plus" style={{ fontSize: 10 }} /> Request attestation
        </button>
      )}
      {addOpen && (
        <form onSubmit={onAdd} className="sd-attest-form">
          <input
            type="text" placeholder="Contributor username" value={newName}
            onChange={(e) => setNewName(e.target.value)} disabled={submitting} autoFocus
          />
          <input
            type="text" placeholder="Role (optional)" value={newRole}
            onChange={(e) => setNewRole(e.target.value)} disabled={submitting}
          />
          <div className="sd-attest-form-actions">
            <button type="submit" className="sd-btn" disabled={submitting || !newName.trim()}>Send request</button>
            <button type="button" className="sd-btn ghost" onClick={() => { setAddOpen(false); setNewName(''); setNewRole(''); }}>Cancel</button>
          </div>
        </form>
      )}
    </div>
  );
}
