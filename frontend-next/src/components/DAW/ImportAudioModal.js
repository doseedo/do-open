import React, { useEffect, useState } from 'react';

/**
 * ImportAudioModal — asks the user whether to detect tempo / meter / key
 * from a freshly-uploaded audio file. Yes runs detect-chords on the
 * server and dispatches the results to the timeline (BPM, meter,
 * downbeat offset, chord row).
 *
 * Auto-dismisses after 6s if the user does nothing (defaults to "No").
 * Enter = Yes, Esc = No.
 */
const ImportAudioModal = ({ filename, onYes, onNo }) => {
  const [secondsLeft, setSecondsLeft] = useState(6);

  useEffect(() => {
    const id = setInterval(() => {
      setSecondsLeft((s) => {
        if (s <= 1) {
          clearInterval(id);
          onNo();
          return 0;
        }
        return s - 1;
      });
    }, 1000);
    return () => clearInterval(id);
  }, [onNo]);

  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Enter') { e.preventDefault(); onYes(); }
      if (e.key === 'Escape') { e.preventDefault(); onNo(); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onYes, onNo]);

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 9999,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
    }}>
      <div style={{
        background: 'rgba(20, 20, 24, 0.95)',
        border: '1px solid rgba(139, 127, 240, 0.3)',
        borderRadius: 10,
        padding: '28px 36px',
        minWidth: 380,
        maxWidth: 460,
        boxShadow: '0 12px 48px rgba(0,0,0,0.6)',
        color: '#e8e8f0',
        fontFamily: 'inherit',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
          <i className="fa-solid fa-wand-magic-sparkles" style={{ color: '#8B7FF0', fontSize: 18 }} />
          <h3 style={{ margin: 0, fontSize: 17, fontWeight: 600 }}>Detect tempo &amp; key?</h3>
        </div>
        <p style={{ margin: '6px 0 16px', fontSize: 13, color: '#aaa', lineHeight: 1.5 }}>
          Analyze <span style={{ color: '#8EC8F0' }}>{filename}</span> and apply
          its tempo, meter, key signature, and downbeat alignment to the timeline?
          This is required for accurate meter conversion.
        </p>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={onYes}
            autoFocus
            style={{
              flex: 1, padding: '10px 14px',
              background: 'rgba(139, 127, 240, 0.85)',
              border: 'none', borderRadius: 6, color: '#fff',
              fontSize: 13, fontWeight: 600, cursor: 'pointer',
            }}
          >
            Detect &amp; apply <span style={{ opacity: 0.6, fontSize: 11 }}>(Enter)</span>
          </button>
          <button
            onClick={onNo}
            style={{
              flex: 1, padding: '10px 14px',
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.12)', borderRadius: 6,
              color: '#bbb', fontSize: 13, cursor: 'pointer',
            }}
          >
            Skip <span style={{ opacity: 0.6, fontSize: 11 }}>(Esc · {secondsLeft}s)</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ImportAudioModal;
