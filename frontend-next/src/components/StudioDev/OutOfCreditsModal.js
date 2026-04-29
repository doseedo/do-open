/*
 * OutOfCreditsModal — themed popup that opens when the generation gate
 * returns 429 (daily cap reached). Dispatched via the CustomEvent in
 * services/outOfCreditsSignal.js; mounted once inside StudioDev so every
 * fetch path (separateStemsAuto, separateStems, future endpoints) can
 * surface it without prop drilling.
 *
 * Session work keeps going — the modal is informational. Continue just
 * closes; Upgrade routes to /pricing.
 */
import React, { useEffect, useState } from 'react';
import { onOutOfCredits } from '../../services/outOfCreditsSignal';

function formatResetTime(resetsAt) {
  if (!resetsAt) return 'the next UTC day';
  // Gate returns an ISO timestamp like "2026-04-23T00:00:00Z". Fall back
  // to the raw string if parsing fails so we never render "Invalid Date".
  const d = new Date(resetsAt);
  if (Number.isNaN(d.getTime())) return resetsAt;
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  const timeStr = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  if (sameDay) return `${timeStr} today`;
  // Tomorrow relative to the user's local day — which is what Redis TTL
  // actually resolves to for most free-tier users (UTC midnight ≈ evening
  // in US, morning in EU).
  const tomorrow = new Date(now);
  tomorrow.setDate(tomorrow.getDate() + 1);
  if (d.toDateString() === tomorrow.toDateString()) return `${timeStr} tomorrow`;
  return d.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
}

export default function OutOfCreditsModal() {
  const [open, setOpen] = useState(false);
  const [resetsAt, setResetsAt] = useState(null);

  useEffect(() => {
    return onOutOfCredits(({ resetsAt }) => {
      setResetsAt(resetsAt || null);
      setOpen(true);
    });
  }, []);

  if (!open) return null;

  const close = () => setOpen(false);
  const upgrade = () => { window.location.href = '/pricing'; };

  return (
    <div className="sd-overlay" onClick={close}>
      <div
        className="sd-overlay-card sd-overlay-small"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sd-overlay-head">
          <span className="sd-midi-kv-k">Out of generation credits</span>
          <button className="sd-midi-btn" onClick={close}>✕</button>
        </div>
        <div className="sd-credits-body">
          <p className="sd-credits-lead">
            Don't worry, you can continue working on your session.
          </p>
          <p className="sd-credits-reset">
            Credits reset at <strong>{formatResetTime(resetsAt)}</strong>.
          </p>
          <div className="sd-credits-actions">
            <button className="sd-midi-btn" onClick={close}>Continue</button>
            <button className="sd-midi-btn sd-midi-primary" onClick={upgrade}>Upgrade</button>
          </div>
        </div>
      </div>
    </div>
  );
}
