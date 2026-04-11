import React from 'react';
import styles from './Legal.module.css';

/**
 * Plans / Pricing page — served at /plans.
 *
 * Placeholder content: the project doesn't have a Stripe-backed subscription
 * flow yet, so this page communicates what's free today and teases a future
 * Pro tier without committing to numbers. When real pricing lands, rewrite
 * this to pull tier data from the auth Cloud Run service (the /api/keys
 * endpoint already lives there) rather than hardcoding.
 */
const Plans = () => (
  <div className={styles.legal}>
    <div className={styles.header}>
      <h1 className={styles.title}>Plans</h1>
      <p className={styles.lastUpdated}>Simple pricing. No lock-in.</p>
    </div>

    <div className={styles.content}>
      <div className={styles.plansGrid}>
        {/* Free tier — available now */}
        <div className={styles.planCard}>
          <div className={styles.planBadgeRow}>
            <span className={styles.planTierName}>Free</span>
            <span className={styles.planTierPrice}>$0</span>
          </div>
          <p className={styles.planTagline}>Everything you need to start making music in the browser.</p>
          <ul className={styles.list}>
            <li><i className="fa-solid fa-check"></i> <span>Full access to the DAW</span></li>
            <li><i className="fa-solid fa-check"></i> <span>Browser-based plugin library</span></li>
            <li><i className="fa-solid fa-check"></i> <span>Local session storage</span></li>
            <li><i className="fa-solid fa-check"></i> <span>AI generation (fair-use limits)</span></li>
          </ul>
          <a href="/login" className={styles.planCtaPrimary}>Get started</a>
        </div>

        {/* Pro tier — coming soon */}
        <div className={`${styles.planCard} ${styles.planCardMuted}`}>
          <div className={styles.planBadgeRow}>
            <span className={styles.planTierName}>Pro</span>
            <span className={styles.planTierPriceMuted}>Coming soon</span>
          </div>
          <p className={styles.planTagline}>Higher generation quotas, priority rendering, and cloud-synced sessions.</p>
          <ul className={styles.list}>
            <li><i className="fa-solid fa-circle-dot"></i> <span>Everything in Free</span></li>
            <li><i className="fa-solid fa-circle-dot"></i> <span>Expanded AI generation limits</span></li>
            <li><i className="fa-solid fa-circle-dot"></i> <span>Cloud session sync</span></li>
            <li><i className="fa-solid fa-circle-dot"></i> <span>Priority support</span></li>
          </ul>
          <a href="/feedback" className={styles.planCtaSecondary}>Tell us what you'd pay for</a>
        </div>
      </div>

      <div className={styles.contactInfo}>
        <p>Questions about pricing? We'd rather hear from you than guess —</p>
        <a href="mailto:support@doseedo.com">support@doseedo.com</a>
      </div>
    </div>
  </div>
);

export default Plans;
