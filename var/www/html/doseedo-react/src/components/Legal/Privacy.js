import React from 'react';
import styles from './Legal.module.css';

const Privacy = () => (
  <div className={styles.legal}>
    <div className={styles.header}>
      <h1 className={styles.title}>Privacy Policy</h1>
      <p className={styles.lastUpdated}>Last updated: April 14, 2026</p>
    </div>

    <div className={styles.content}>
      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Introduction</h2>
        <p className={styles.sectionText}>
          This is the privacy policy for all websites owned and operated by Doseedo Inc. ("Doseedo," "we," or "our"), including doseedo.com (the "Service"). We respect your privacy and are committed to protecting your personal information. This Privacy Policy is subject to the California Consumer Privacy Act (CCPA).
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Information We Collect</h2>
        <p className={styles.sectionText}>
          We collect information at several different points on the Service, including:
        </p>
        <ul className={styles.list}>
          <li><i className="fa-solid fa-circle"></i>Account information: name, email address, and username when you register</li>
          <li><i className="fa-solid fa-circle"></i>Payment information: processed securely through Stripe — we do not store credit card details</li>
          <li><i className="fa-solid fa-circle"></i>Usage data: browser type, operating system, IP address, and Service usage information</li>
        </ul>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          We will not sell, share, or rent personal information to third parties without your explicit permission, except as disclosed in this Privacy Policy.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>How We Handle Your Audio</h2>
        <p className={styles.sectionText}>
          When you upload audio to Doseedo Studio, most of the heavy work happens on your own device — not on our servers. Here is exactly what runs where, and what we keep.
        </p>
        <ul className={styles.list}>
          <li><i className="fa-solid fa-circle"></i>
            <strong>Stem separation and waveform encoding run in your browser.</strong>{' '}
            Your audio is decoded, resampled to 48&nbsp;kHz, and passed through our ONNX models (<code>distill_demucs</code> and the Oobleck VAE encoder) via WebGPU or WASM directly inside the tab. The original audio file does not leave your device for these steps.
          </li>
          <li><i className="fa-solid fa-circle"></i>
            <strong>Only the compact latent is uploaded.</strong>{' '}
            After local encoding, we transmit a small latent representation (about 64&nbsp;floats per 40&nbsp;ms of audio) to our backend via <code>/api/upload-latent</code>. That latent is what downstream tools — stem playback, tempo change, meter edit — read from. The raw waveform bytes stay in your browser.
          </li>
          <li><i className="fa-solid fa-circle"></i>
            <strong>Analysis endpoints are a narrow exception.</strong>{' '}
            Instrument classification and MIDI extraction still require server-side processing in the current release. For those two calls your audio is sent to our GPU backend, held in memory on an ephemeral container, and cleared when the container scales down (typically within 15 minutes of the request). Nothing is persisted to long-term storage, and nothing is used for training.
          </li>
          <li><i className="fa-solid fa-circle"></i>
            <strong>Generated output is saved to your account.</strong>{' '}
            Audio you generate with Doseedo tools (stemphonic, stem repaint, score rendering) is written to a per-user folder in our Google Cloud Storage bucket so you can come back and download it. You can delete any generation from the Studio UI; we honor deletion requests for backing storage within 7 business days.
          </li>
          <li><i className="fa-solid fa-circle"></i>
            <strong>We do not train on your audio.</strong>{' '}
            Nothing you upload, encode, or generate is used to train, fine-tune, or evaluate our models. Our models are trained on licensed and public-domain datasets only, and that set is frozen for each release.
          </li>
          <li><i className="fa-solid fa-circle"></i>
            <strong>Third parties.</strong>{' '}
            The only third-party services that touch your audio pipeline are Modal (GPU inference compute) and Google Cloud Storage (generation output bucket). Neither retains your content for their own use. Payment data is handled by Stripe and never touches this pipeline.
          </li>
        </ul>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          We built this architecture deliberately so that "client-side encoding" is provable, not just marketing. If you open the browser DevTools Network tab during an upload, you will see <code>/api/upload-latent</code> requests carrying latent bytes (content-type <code>application/octet-stream</code>) and no request carrying your original audio payload for stem separation.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>How We Use Your Information</h2>
        <ul className={styles.list}>
          <li><i className="fa-solid fa-circle"></i>Delivering products and services you've purchased</li>
          <li><i className="fa-solid fa-circle"></i>Sending purchase receipts and download links</li>
          <li><i className="fa-solid fa-circle"></i>Providing technical support</li>
          <li><i className="fa-solid fa-circle"></i>Improving and maintaining the Service</li>
          <li><i className="fa-solid fa-circle"></i>Communicating about products and updates (with your consent)</li>
        </ul>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Data Protection</h2>
        <p className={styles.sectionText}>
          We use encryption and authentication tools to protect your information. Access to personal data is restricted to employees who need it to provide services. We use Stripe for payment processing, which is PCI-DSS compliant. However, no data transmission over the Internet can be guaranteed 100% secure.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>California Residents (CCPA)</h2>
        <p className={styles.sectionText}>
          Under the California Consumer Privacy Act, California residents have the right to: (1) know what personal information we collect, use, and may disclose; (2) request deletion of personal information; and (3) be free from discrimination for exercising these rights. To exercise your CCPA rights, please email us at <a href="mailto:support@doseedo.com">support@doseedo.com</a>.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Children's Privacy</h2>
        <p className={styles.sectionText}>
          We do not knowingly collect personally identifiable information from children under the age of 13, in compliance with the Children's Online Privacy Protection Act (COPPA).
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Data Breach Notification</h2>
        <p className={styles.sectionText}>
          In the event of a data breach, we will notify affected users via email within 7 business days and via in-site notification within 7 business days.
        </p>
      </div>

      <div className={styles.contactInfo}>
        <p>Doseedo Inc. — Marina Del Rey, California</p>
        <a href="mailto:support@doseedo.com">support@doseedo.com</a>
      </div>
    </div>
  </div>
);

export default Privacy;
