import React from 'react';
import styles from './Legal.module.css';

const Privacy = () => (
  <div className={styles.legal}>
    <div className={styles.header}>
      <h1 className={styles.title}>Privacy Policy</h1>
      <p className={styles.lastUpdated}>Last updated: April 25, 2026</p>
    </div>

    <div className={styles.content}>
      <div className={styles.section}>
        <p className={styles.sectionText}>
          This Privacy Policy explains how Doseedo, Inc. ("Doseedo," "we," "us," or "our") collects, uses, shares, and protects information in connection with the Doseedo desktop application, web application, mobile applications, AI tools, cloud sync, collaboration features, and related services (collectively, the "Service").
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          We've tried to write this in plain language. Where we use defined terms, they have the same meaning as in our Terms of Service.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>1. Information We Collect</h2>

        <h3 className={styles.subsectionTitle}>1.1 Information You Provide</h3>
        <p className={styles.sectionText}>
          <strong>Account information.</strong> When you register, we collect your name, email address, password (stored hashed), and any optional profile information you provide.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>Payment information.</strong> If you subscribe to a paid plan, we collect billing information through our payment processor. We do not store full payment card numbers; our payment processor handles that.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>Communications.</strong> When you contact us (support, sales, feedback), we keep records of those communications.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>Your Content.</strong> Sessions, audio uploads, MIDI, project files, and other content you upload, create, or store through the Service. See Sections 2 and 3 for how we handle this.
        </p>

        <h3 className={styles.subsectionTitle}>1.2 Information Collected Automatically</h3>
        <p className={styles.sectionText}>
          <strong>Usage data.</strong> How you use the Service: features accessed, actions taken, sessions created, generations performed, errors encountered, performance metrics. We use this to operate and improve the Service.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>Device and connection information.</strong> IP address, device type, operating system, application version, browser type, language settings, time zone, and similar technical information.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>Cookies and similar technologies.</strong> We use cookies, local storage, and similar technologies to authenticate users, remember preferences, and analyze usage. See Section 8 for details.
        </p>

        <h3 className={styles.subsectionTitle}>1.3 Information from Third Parties</h3>
        <p className={styles.sectionText}>
          <strong>Authentication providers.</strong> If you sign in via Apple, Google, or another identity provider, we receive basic profile information from them (typically email address and name).
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>Payment processors.</strong> Our payment processor confirms transactions and shares limited information needed for billing.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>DAW integrations.</strong> When you use Doseedo with third-party DAWs (Logic, Pro Tools, Ableton, etc.), we receive session metadata you choose to sync. We do not receive information from your DAW that you have not authorized us to receive.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>2. How We Handle Your Content</h2>
        <p className={styles.sectionText}>
          Your sessions, audio uploads, generated content, and other creative work require special treatment. Here's how we handle it:
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          <strong>Cloud storage and sync.</strong> When you sync sessions to Doseedo's cloud, we store them on our infrastructure (currently hosted on [provider]) so you can access them from your devices. Your sessions are private by default — only you can access them, except where you have invited specific collaborators.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>Audio-to-session uploads.</strong> When you upload audio for transformation into session data, we process it and return the result. We delete the original audio upload from active processing systems after processing completes. Brief retention may occur for technical reliability (queue redundancy, error recovery) but does not exceed [X hours/days]. We do not retain, archive, or distribute your original audio uploads.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>Generated content.</strong> AI outputs you generate through the Service are stored as part of your session data. They are treated the same as other Your Content.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>Collaboration.</strong> When you invite a collaborator to a session, that collaborator gains access to the session as you authorize. We make the session available to them through the Service. Collaborators can view, edit, and contribute according to the permissions you grant.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>Training data.</strong> We do not use Your Content to train Doseedo's AI models. Your audio uploads, generated outputs, sessions, and other content are not added to Doseedo's training corpus. This is enforced architecturally — Your Content is stored separately from training data and is not accessible to training pipelines.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>Provenance.</strong> The Service tracks provenance of sessions and generations to support the Service's version control and rights-tracking features. This metadata travels with your content and is visible to you and (where applicable) your collaborators.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>3. How We Use Information</h2>
        <p className={styles.sectionText}>We use information to:</p>
        <ul className={styles.list}>
          <li><i className="fa-solid fa-circle"></i><span><strong>Provide the Service.</strong> Authenticate you, sync your sessions, run AI generation, deliver collaboration features, process payments.</span></li>
          <li><i className="fa-solid fa-circle"></i><span><strong>Improve the Service.</strong> Analyze usage patterns to identify bugs, prioritize features, and improve performance. This analysis uses aggregated and de-identified data where possible. We do not analyze Your Content for product improvement except in aggregate forms that do not identify specific creative work.</span></li>
          <li><i className="fa-solid fa-circle"></i><span><strong>Communicate with you.</strong> Send service announcements, security alerts, billing notices, and (with your consent) marketing communications. You can opt out of marketing at any time.</span></li>
          <li><i className="fa-solid fa-circle"></i><span><strong>Provide support.</strong> Respond to your inquiries, troubleshoot issues, and assist you with the Service.</span></li>
          <li><i className="fa-solid fa-circle"></i><span><strong>Ensure safety and integrity.</strong> Detect and prevent fraud, abuse, security incidents, and violations of our Terms.</span></li>
          <li><i className="fa-solid fa-circle"></i><span><strong>Comply with law.</strong> Meet legal obligations, respond to lawful requests, and enforce our agreements.</span></li>
        </ul>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          We do not sell Your Content or your personal information.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>4. How We Share Information</h2>
        <p className={styles.sectionText}>We share information only as described here:</p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          <strong>With your collaborators.</strong> When you invite collaborators, they gain access to the relevant session as you authorize.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>With service providers.</strong> We use third-party service providers to operate the Service, including cloud hosting, payment processing, email delivery, analytics, customer support, and AI infrastructure. These providers process information on our behalf under contracts that limit their use of information to providing services to us.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          Current categories of service providers include:
        </p>
        <ul className={styles.list}>
          <li><i className="fa-solid fa-circle"></i>Cloud infrastructure (currently [provider])</li>
          <li><i className="fa-solid fa-circle"></i>Payment processing (currently [provider])</li>
          <li><i className="fa-solid fa-circle"></i>Email and communications (currently [provider])</li>
          <li><i className="fa-solid fa-circle"></i>Analytics (currently [provider])</li>
          <li><i className="fa-solid fa-circle"></i>Customer support tools (currently [provider])</li>
        </ul>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          <strong>With DAW integrations.</strong> When you use Doseedo with a third-party DAW, the DAW vendor may receive limited information necessary for the integration to function. We do not control how DAW vendors handle their information; consult their privacy policies.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>For legal reasons.</strong> We may disclose information when we have a good-faith belief that disclosure is required or permitted by law, including in response to subpoenas, court orders, or other legal process; to investigate or address violations of our Terms; or to protect the rights, property, or safety of Doseedo, our users, or the public.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>In a corporate transaction.</strong> If Doseedo is involved in a merger, acquisition, financing, or sale of assets, information may be transferred as part of the transaction, subject to commitments consistent with this Privacy Policy.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>With your consent.</strong> We share information for other purposes when you direct or consent.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>5. Your Choices and Rights</h2>
        <p className={styles.sectionText}>
          <strong>Access and correction.</strong> You can access and update most account information through your account settings.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>Export.</strong> You can export your sessions and generated content at any time.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>Deletion.</strong> You can delete sessions, generated content, and your account through account settings or by contacting us. When you delete content, we remove it from active systems; backup retention may persist for [X days] before full deletion.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>Marketing opt-out.</strong> You can opt out of marketing emails through the unsubscribe link in any marketing email or in your account settings.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 12 }}>
          <strong>Cookies.</strong> You can manage cookies through your browser settings. Disabling cookies may limit Service functionality.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          <strong>Regional rights.</strong> Depending on where you live, you may have additional rights:
        </p>
        <ul className={styles.list}>
          <li><i className="fa-solid fa-circle"></i><span><strong>California residents (CCPA/CPRA).</strong> You have rights to know what personal information we collect, request deletion, request correction, opt out of sale or sharing (we do not sell personal information), and not be discriminated against for exercising your rights. To exercise rights, contact us at <a href="mailto:privacy@doseedo.com">privacy@doseedo.com</a> or via our request portal. You may designate an authorized agent.</span></li>
          <li><i className="fa-solid fa-circle"></i><span><strong>EEA, UK, and Switzerland residents (GDPR/UK GDPR).</strong> You have rights of access, rectification, erasure, restriction of processing, data portability, and objection. You can withdraw consent where processing is based on consent. You may lodge complaints with your supervisory authority. Our legal bases for processing are: contract performance (providing the Service), legitimate interests (improving the Service, security, fraud prevention), consent (where required, e.g., marketing), and legal obligation. Our EU/UK representative is [TBD if required by Article 27].</span></li>
          <li><i className="fa-solid fa-circle"></i><span><strong>Other regions.</strong> Where applicable law provides additional rights, we honor them.</span></li>
        </ul>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          To exercise any of these rights, contact <a href="mailto:privacy@doseedo.com">privacy@doseedo.com</a>.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>6. Data Retention</h2>
        <p className={styles.sectionText}>
          We retain information as long as needed to provide the Service and as required or permitted by law. Specifically:
        </p>
        <ul className={styles.list}>
          <li><i className="fa-solid fa-circle"></i><span><strong>Account information:</strong> While your account is active, plus a reasonable period after closure for legal and operational reasons.</span></li>
          <li><i className="fa-solid fa-circle"></i><span><strong>Your Content:</strong> While stored in your account; deleted when you delete it (with backup retention as noted above).</span></li>
          <li><i className="fa-solid fa-circle"></i><span><strong>Audio-to-session uploads:</strong> Deleted after processing as described in Section 2.</span></li>
          <li><i className="fa-solid fa-circle"></i><span><strong>Usage data:</strong> Typically retained in identifiable form for [X months/years], then aggregated or deleted.</span></li>
          <li><i className="fa-solid fa-circle"></i><span><strong>Communications and support records:</strong> Retained for [X years] for service improvement and legal purposes.</span></li>
          <li><i className="fa-solid fa-circle"></i><span><strong>Billing records:</strong> Retained as required by tax and accounting laws (typically 7 years).</span></li>
        </ul>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>7. Security</h2>
        <p className={styles.sectionText}>
          We use technical and organizational measures to protect information, including encryption in transit (TLS), encryption at rest for sensitive data, access controls, audit logging, and secure development practices. No system is perfectly secure; we cannot guarantee absolute security but commit to industry-standard practices and to notifying affected users of security incidents as required by law.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>8. Cookies and Similar Technologies</h2>
        <p className={styles.sectionText}>We use:</p>
        <ul className={styles.list}>
          <li><i className="fa-solid fa-circle"></i><span><strong>Essential cookies.</strong> Required for authentication and core Service functionality. Cannot be disabled.</span></li>
          <li><i className="fa-solid fa-circle"></i><span><strong>Functional cookies.</strong> Remember preferences and settings. Can be disabled but may affect functionality.</span></li>
          <li><i className="fa-solid fa-circle"></i><span><strong>Analytics cookies.</strong> Help us understand usage and improve the Service. Can be disabled in your account settings or browser.</span></li>
        </ul>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          We do not use advertising cookies or third-party tracking for advertising purposes.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>9. International Data Transfers</h2>
        <p className={styles.sectionText}>
          Doseedo is based in the United States. If you are outside the United States, the information we collect may be transferred to and processed in the United States or other countries where we or our service providers operate. These countries may have data protection laws different from those in your country.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          For transfers from the EEA, UK, or Switzerland, we rely on Standard Contractual Clauses or other appropriate safeguards.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>10. Children's Privacy</h2>
        <p className={styles.sectionText}>
          The Service is not directed to children under 13, and we do not knowingly collect personal information from children under 13. If we learn we have collected information from a child under 13, we will delete it. Parents or guardians who believe their child has provided information should contact <a href="mailto:privacy@doseedo.com">privacy@doseedo.com</a>.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>11. Changes to This Privacy Policy</h2>
        <p className={styles.sectionText}>
          We may update this Privacy Policy from time to time. If we make material changes, we will notify you (for example, by email or in-product notice) before the changes take effect.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>12. Contact</h2>
        <p className={styles.sectionText}>
          Questions about this Privacy Policy or our practices? Contact us at:
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          Doseedo, Inc.
          {'\n'}[ADDRESS]
          {'\n'}<a href="mailto:privacy@doseedo.com">privacy@doseedo.com</a>
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          For copyright/DMCA matters, see our Terms of Service or contact <a href="mailto:dmca@doseedo.com">dmca@doseedo.com</a>.
        </p>
      </div>

      <div className={styles.contactInfo}>
        <p>Doseedo, Inc. — Marina Del Rey, California</p>
        <a href="mailto:privacy@doseedo.com">privacy@doseedo.com</a>
      </div>
    </div>
  </div>
);

export default Privacy;
