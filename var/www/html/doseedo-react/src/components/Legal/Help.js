import React from 'react';
import styles from './Legal.module.css';

/**
 * Help page — served at /help.
 *
 * Replaces the dead external link to docs.doseedo.com that used to live in
 * the LeftSidebar "More" dropdown. Kept deliberately lean: a short orientation
 * blurb, pointers to the parts of the app people most often ask about, and a
 * mailto fallback. Full documentation is not in scope here — when a real docs
 * site exists (Docusaurus / Gitbook / whatever), change the sidebar link back
 * to an external <a href="https://docs.doseedo.com"> and retire this file.
 */
const Help = () => (
  <div className={styles.legal}>
    <div className={styles.header}>
      <h1 className={styles.title}>Help</h1>
      <p className={styles.lastUpdated}>Getting started with Doseedo</p>
    </div>

    <div className={styles.content}>
      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Getting Started</h2>
        <p className={styles.sectionText}>
          Doseedo is a browser-based music production environment. Open{' '}
          <strong>Projects</strong> in the sidebar to create a new session, or
          jump into <strong>Plugins</strong> to try our in-browser instruments
          and effects. Sessions are saved locally in your browser — no upload
          required.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Keyboard Shortcuts</h2>
        <ul className={styles.list}>
          <li><i className="fa-solid fa-circle"></i> <span><strong>Space</strong> — play / pause</span></li>
          <li><i className="fa-solid fa-circle"></i> <span><strong>Cmd / Ctrl + S</strong> — save session</span></li>
        </ul>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Common Questions</h2>
        <p className={styles.sectionText}>
          <strong>Where are my sessions stored?</strong>{'\n'}
          In your browser's local storage. Clearing site data will erase them —
          export important sessions before doing so.{'\n'}{'\n'}
          <strong>Can I use Doseedo offline?</strong>{'\n'}
          Partially. Once loaded, most tools run locally in your browser, but
          AI generation and cloud sync require an internet connection.{'\n'}{'\n'}
          <strong>How do I report a bug?</strong>{'\n'}
          Use the Feedback page in the sidebar, or email us directly.
        </p>
      </div>

      <div className={styles.contactInfo}>
        <p>Still stuck? Reach out —</p>
        <a href="mailto:support@doseedo.com">support@doseedo.com</a>
      </div>
    </div>
  </div>
);

export default Help;
