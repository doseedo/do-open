import React from 'react';
import styles from './Legal.module.css';

const About = () => (
  <div className={styles.legal}>
    <div className={styles.header}>
      <h1 className={`${styles.title} page-title`}>About Doseedo</h1>
    </div>

    <div className={styles.content}>
      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Our Mission</h2>
        <p className={styles.sectionText}>
          At Doseedo, we build professional audio tools for musicians, producers, and sound designers. Our mission is to make high-quality audio production accessible to everyone — from bedroom producers to professional studios.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>What We Do</h2>
        <p className={styles.sectionText}>
          We develop audio plugins and production software that combine innovative signal processing with intuitive interfaces. Our tools are designed to integrate seamlessly into your existing workflow, whether you use Logic Pro, Ableton Live, FL Studio, or any other major DAW.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Our Commitment</h2>
        <p className={styles.sectionText}>
          We're committed to building reliable, high-performance tools at fair prices. Every plugin we release is crafted with care — rigorously tested across platforms and optimized for low-latency, real-time performance. We value your feedback and invite you to share your thoughts with us as we continue to grow.
        </p>
      </div>

      <div className={styles.contactInfo}>
        <p>Questions or feedback?</p>
        <a href="mailto:support@doseedo.com">support@doseedo.com</a>
      </div>
    </div>
  </div>
);

export default About;
