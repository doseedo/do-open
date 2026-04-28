import React, { useState } from 'react';
import styles from './Legal.module.css';

/**
 * Feedback page — served at /feedback.
 *
 * There is no backend feedback endpoint yet, so submission currently opens
 * the user's mail client with the form contents pre-filled via mailto:. This
 * is deliberate — it ships today with zero infra, and the upgrade path is
 * just swapping the onSubmit handler for a fetch() to whatever endpoint we
 * stand up later (likely a new route on the auth Cloud Run service).
 */
const Feedback = () => {
  const [category, setCategory] = useState('general');
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!message.trim()) return;

    const subjectPrefix = {
      general: 'Feedback',
      bug: 'Bug report',
      feature: 'Feature request',
    }[category] || 'Feedback';

    const subject = encodeURIComponent(`[${subjectPrefix}] Doseedo feedback`);
    const body = encodeURIComponent(message);
    window.location.href = `mailto:feedback@doseedo.com?subject=${subject}&body=${body}`;
  };

  return (
    <div className={styles.legal}>
      <div className={styles.header}>
        <h1 className={`${styles.title} page-title`}>Feedback</h1>
        <p className={styles.lastUpdated}>We read everything</p>
      </div>

      <div className={styles.content}>
        <div className={styles.section}>
          <h2 className={styles.sectionTitle}>Tell us what you think</h2>
          <p className={styles.sectionText}>
            Found a bug, have an idea, or just want to say hi? Drop a note
            below and we'll get back to you. Your message opens in your mail
            app so you have a copy of what you sent.
          </p>

          <form onSubmit={handleSubmit} className={styles.feedbackForm}>
            <label className={styles.feedbackLabel}>
              Category
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className={styles.feedbackSelect}
              >
                <option value="general">General feedback</option>
                <option value="bug">Bug report</option>
                <option value="feature">Feature request</option>
              </select>
            </label>

            <label className={styles.feedbackLabel}>
              Message
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                className={styles.feedbackTextarea}
                placeholder="What's on your mind?"
                rows={8}
                required
              />
            </label>

            <button type="submit" className={styles.feedbackButton}>
              Send Feedback
            </button>
          </form>
        </div>

        <div className={styles.contactInfo}>
          <p>Prefer email? Write to us directly at</p>
          <a href="mailto:feedback@doseedo.com">feedback@doseedo.com</a>
        </div>
      </div>
    </div>
  );
};

export default Feedback;
