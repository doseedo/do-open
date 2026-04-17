import React from 'react';
import styles from './Legal.module.css';

const Terms = () => (
  <div className={styles.legal}>
    <div className={styles.header}>
      <h1 className={styles.title}>Terms of Service</h1>
      <p className={styles.lastUpdated}>Last updated: February 16, 2026</p>
    </div>

    <div className={styles.content}>
      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Agreement</h2>
        <p className={styles.sectionText}>
          This is a user agreement ("Agreement") between you and Doseedo Inc. ("Doseedo," "we," or "our") providing the terms and conditions for your use of the Doseedo website and services (the "Service"). By using the Service, you agree to abide by all terms of this Agreement. If you do not agree, please do not use the Service.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Intellectual Property</h2>
        <p className={styles.sectionText}>
          All information, content, services, and software displayed on, transmitted through, or used in connection with the Service is owned by Doseedo and its affiliates, licensors, or suppliers. The Service is protected by copyright, trademark, and other intellectual property laws. You may not decompile, reverse engineer, or disassemble any software accessible through the Service.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Plugin Purchases & Licenses</h2>
        <p className={styles.sectionText}>
          When you purchase a plugin from Doseedo, you receive a personal, non-transferable license to use the plugin. You may install the plugin on your own devices for your personal or professional use. You may not:
        </p>
        <ul className={styles.list}>
          <li><i className="fa-solid fa-circle"></i>Redistribute, resell, or share the plugin files with others</li>
          <li><i className="fa-solid fa-circle"></i>Share download links or access credentials with third parties</li>
          <li><i className="fa-solid fa-circle"></i>Reverse engineer, decompile, or modify the plugin software</li>
          <li><i className="fa-solid fa-circle"></i>Use the plugin in any way that violates applicable laws</li>
        </ul>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Refund Policy</h2>
        <p className={styles.sectionText}>
          Due to the digital nature of our products, all sales are final. If you experience technical issues with a plugin, please contact <a href="mailto:support@doseedo.com">support@doseedo.com</a> and we will work to resolve the issue. Refunds may be considered on a case-by-case basis for verified technical problems that prevent use of the product.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>User Content</h2>
        <p className={styles.sectionText}>
          Any content you submit through the Service is not private. By submitting content, you grant Doseedo a non-exclusive, royalty-free, worldwide license to use, copy, modify, and display such content. You confirm that any content you submit is original or that you have obtained all necessary permissions.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Disclaimer of Warranties</h2>
        <p className={styles.sectionText}>
          THE SERVICE AND ALL CONTENT ARE PROVIDED "AS IS" WITHOUT ANY REPRESENTATION OR WARRANTY OF ANY KIND. DOSEEDO DISCLAIMS ALL WARRANTIES, EXPRESS AND IMPLIED, INCLUDING WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. DOSEEDO DOES NOT WARRANT THAT THE SERVICE WILL BE UNINTERRUPTED OR ERROR-FREE.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Limitation of Liability</h2>
        <p className={styles.sectionText}>
          DOSEEDO SHALL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR EXEMPLARY DAMAGES, INCLUDING DAMAGES FOR LOSS OF PROFITS, GOODWILL, USE, DATA, OR OTHER INTANGIBLE LOSSES. SOME JURISDICTIONS DO NOT ALLOW THE EXCLUSION OF CERTAIN WARRANTIES OR LIMITATION OF LIABILITY, SO SOME OF THE ABOVE LIMITATIONS MAY NOT APPLY TO YOU.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>Governing Law</h2>
        <p className={styles.sectionText}>
          This Agreement shall be governed by and construed in accordance with the laws of the State of California, without regard to its conflict of laws provisions. You agree to submit to personal and exclusive jurisdiction in the state and federal courts located in Los Angeles County, California.
        </p>
      </div>

      <div className={styles.contactInfo}>
        <p>Doseedo Inc. — Marina Del Rey, California</p>
        <a href="mailto:support@doseedo.com">support@doseedo.com</a>
      </div>
    </div>
  </div>
);

export default Terms;
