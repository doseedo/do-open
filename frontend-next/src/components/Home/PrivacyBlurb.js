import React from 'react';
import { Link } from 'react-router-dom';

/**
 * PrivacyBlurb — landing-page one-paragraph callout that captures the
 * differentiator: stem separation and encoding happen client-side, we
 * keep only latent representations, we don't train on user content.
 *
 * Drop it into Home.js wherever a trust / differentiator slot is
 * needed (below the hero, between feature sections, above the CTA).
 * It links to the full Privacy Policy for the longer version.
 */
const PrivacyBlurb = () => (
  <section
    aria-labelledby="privacy-blurb-title"
    style={{
      maxWidth: 720,
      margin: '48px auto',
      padding: '28px 32px',
      borderRadius: 12,
      background: 'rgba(255, 255, 255, 0.04)',
      border: '1px solid rgba(139, 127, 240, 0.18)',
    }}
  >
    <h2
      id="privacy-blurb-title"
      style={{
        fontSize: 18,
        fontWeight: 600,
        margin: 0,
        color: '#cdbfff',
        letterSpacing: 0.2,
      }}
    >
      Your audio stays on your device.
    </h2>
    <p
      style={{
        marginTop: 12,
        marginBottom: 0,
        fontSize: 15,
        lineHeight: 1.6,
        color: 'rgba(255, 255, 255, 0.8)',
      }}
    >
      Doseedo runs stem separation and waveform encoding in your browser
      via WebGPU. Only a compact latent representation — not your audio —
      is uploaded to our servers for playback and editing. We do not train
      on your uploads or your generations.{' '}
      <Link
        to="/privacy"
        style={{ color: '#cdbfff', textDecoration: 'underline' }}
      >
        Read the full policy →
      </Link>
    </p>
  </section>
);

export default PrivacyBlurb;
