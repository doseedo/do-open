import React from 'react';
import styles from './Legal.module.css';

const Terms = () => (
  <div className={styles.legal}>
    <div className={styles.header}>
      <h1 className={styles.title}>Terms of Service</h1>
      <p className={styles.lastUpdated}>Last updated: April 25, 2026</p>
    </div>

    <div className={styles.content}>
      <div className={styles.section}>
        <p className={styles.sectionText}>
          These Terms of Service ("Terms") govern your access to and use of the services, software, and websites provided by Doseedo, Inc. ("Doseedo," "we," "us," or "our"), including the Doseedo desktop application, web application, mobile applications, AI tools, cloud sync, collaboration features, and related services (collectively, the "Service").
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          By accessing or using the Service, you agree to be bound by these Terms. If you do not agree, do not use the Service.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>1. Eligibility and Account Registration</h2>
        <p className={styles.sectionText}>
          You must be at least 18 years old, or the age of legal majority in your jurisdiction, to use the Service. If you are between 13 and 18, you may use the Service only with the consent and supervision of a parent or legal guardian who agrees to be bound by these Terms on your behalf.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          You must register an account to access most Service features. You agree to provide accurate, current, and complete information during registration and to keep your account information updated. You are responsible for maintaining the security of your account credentials and for all activity that occurs under your account.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>2. The Service</h2>
        <p className={styles.sectionText}>
          Doseedo provides creative tools for music producers, composers, and other audio professionals. The Service includes, among other things:
        </p>
        <ul className={styles.list}>
          <li><i className="fa-solid fa-circle"></i>AI-assisted stem and instrument generation conditioned on user-provided MIDI, structure, and timbre inputs</li>
          <li><i className="fa-solid fa-circle"></i>Audio-to-session conversion that transforms user-provided audio into editable session data</li>
          <li><i className="fa-solid fa-circle"></i>Cloud synchronization of user sessions across the user's own devices</li>
          <li><i className="fa-solid fa-circle"></i>Invite-based collaboration features allowing users to grant specific other users access to specific sessions</li>
          <li><i className="fa-solid fa-circle"></i>Integration with third-party digital audio workstations including Logic Pro, Pro Tools, Ableton Live, and others</li>
          <li><i className="fa-solid fa-circle"></i>Provenance tracking for sessions and generated content</li>
          <li><i className="fa-solid fa-circle"></i>A web-based digital audio workstation</li>
        </ul>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          The Service is a creator tool. It is designed for active music production by users. It is not designed for, and you may not use it as, a passive consumption or streaming service.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>3. Your Content</h2>

        <h3 className={styles.subsectionTitle}>3.1 What "Your Content" Means</h3>
        <p className={styles.sectionText}>
          "Your Content" means any audio, MIDI, project files, session data, text, images, or other materials that you upload, submit, create, or store through the Service, including content created with AI assistance through the Service.
        </p>

        <h3 className={styles.subsectionTitle}>3.2 Ownership</h3>
        <p className={styles.sectionText}>
          You retain all ownership rights in Your Content, subject to the rights you grant to Doseedo in these Terms. Doseedo does not claim ownership of Your Content.
        </p>

        <h3 className={styles.subsectionTitle}>3.3 Your Representations About Your Content</h3>
        <p className={styles.sectionText}>You represent and warrant that:</p>
        <ul className={styles.list}>
          <li><i className="fa-solid fa-circle"></i>(a) You own Your Content, or you have all necessary rights, licenses, consents, and permissions to upload, store, process, and share Your Content through the Service;</li>
          <li><i className="fa-solid fa-circle"></i>(b) Your Content does not and will not infringe, misappropriate, or violate any third party's intellectual property rights, rights of publicity or privacy, or any applicable law or regulation;</li>
          <li><i className="fa-solid fa-circle"></i>(c) Any audio you upload that incorporates compositions or sound recordings owned by third parties has been licensed by you for the uses you make of it through the Service, or your use is otherwise lawful (for example, fair use);</li>
          <li><i className="fa-solid fa-circle"></i>(d) Any persons whose voices, performances, or likenesses appear in Your Content have consented to such use; and</li>
          <li><i className="fa-solid fa-circle"></i>(e) You will not use the Service to upload, generate, or share content that you know or should know infringes the rights of others.</li>
        </ul>

        <h3 className={styles.subsectionTitle}>3.4 License to Doseedo</h3>
        <p className={styles.sectionText}>
          Solely to provide the Service to you, you grant Doseedo a limited, worldwide, non-exclusive, royalty-free license to host, store, copy, transmit, process, display, and (where you have used collaboration or sharing features) make Your Content available to the specific other users you have authorized. This license is limited to operational use of the Service and terminates when you delete Your Content or close your account, except for content you have shared with others (which the recipients may retain) and except as needed for backup retention, legal compliance, or dispute resolution.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          We do not use Your Content to train Doseedo's AI models. Your audio uploads, generated outputs, sessions, and other content are not added to Doseedo's training corpus. This commitment is architectural: Your Content is stored separately from training data and is not accessible to training pipelines.
        </p>

        <h3 className={styles.subsectionTitle}>3.5 Audio-to-Session Processing</h3>
        <p className={styles.sectionText}>
          When you use audio-to-session features, you upload audio to the Service for transformation into editable session data (MIDI, stems, structural information). The Service processes your upload, returns the transformed result, and deletes the original audio upload from active processing systems. Brief retention may occur for technical reliability (queue redundancy, error recovery) but does not exceed [X hours/days]. We do not retain, archive, or distribute your original audio uploads.
        </p>

        <h3 className={styles.subsectionTitle}>3.6 Your Responsibility for Your Content</h3>
        <p className={styles.sectionText}>
          You are solely responsible for Your Content and the consequences of uploading, generating, sharing, or distributing it. Doseedo does not pre-screen Your Content and is not responsible for it.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>4. Acceptable Use</h2>
        <p className={styles.sectionText}>You agree not to use the Service to:</p>
        <ul className={styles.list}>
          <li><i className="fa-solid fa-circle"></i>Infringe, misappropriate, or violate any third party's intellectual property rights, including by uploading copyrighted audio you do not have rights to use, generating content intended to infringe a specific copyrighted work, or distributing infringing content through the Service's collaboration features;</li>
          <li><i className="fa-solid fa-circle"></i>Upload, generate, or share content that is unlawful, defamatory, fraudulent, deceptive, harassing, threatening, hateful, or that incites violence or harm;</li>
          <li><i className="fa-solid fa-circle"></i>Upload, generate, or share sexually explicit content involving minors, non-consensual intimate imagery, or content depicting real-world violence in graphic detail;</li>
          <li><i className="fa-solid fa-circle"></i>Impersonate another person or entity, or misrepresent your affiliation with any person or entity;</li>
          <li><i className="fa-solid fa-circle"></i>Use the Service to develop a competing product, reverse engineer the Service, or extract Doseedo's proprietary models, training data, or trade secrets;</li>
          <li><i className="fa-solid fa-circle"></i>Attempt to gain unauthorized access to the Service, other users' accounts, or Doseedo's systems;</li>
          <li><i className="fa-solid fa-circle"></i>Interfere with the Service's operation, including by transmitting malware, conducting denial-of-service attacks, or scraping content at volumes that burden Service infrastructure;</li>
          <li><i className="fa-solid fa-circle"></i>Resell, sublicense, or redistribute the Service or access to it without our written permission;</li>
          <li><i className="fa-solid fa-circle"></i>Use the Service in violation of applicable export controls, sanctions, or other laws;</li>
          <li><i className="fa-solid fa-circle"></i>Use the Service to operate as a music streaming service, music distribution platform, or other consumption-oriented service for end listeners.</li>
        </ul>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          We may suspend or terminate accounts that violate this Acceptable Use Policy, with or without notice depending on severity.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>5. Copyright Infringement; DMCA</h2>

        <h3 className={styles.subsectionTitle}>5.1 Reporting Infringement</h3>
        <p className={styles.sectionText}>
          Doseedo respects intellectual property rights and responds to clear notices of alleged copyright infringement. If you believe content on the Service infringes your copyright, please send a notice to our Designated Agent containing:
        </p>
        <ul className={styles.list}>
          <li><i className="fa-solid fa-circle"></i>(a) A physical or electronic signature of the copyright owner or authorized agent;</li>
          <li><i className="fa-solid fa-circle"></i>(b) Identification of the copyrighted work claimed to be infringed;</li>
          <li><i className="fa-solid fa-circle"></i>(c) Identification of the material claimed to be infringing, with information sufficient for us to locate it (such as URL or session ID);</li>
          <li><i className="fa-solid fa-circle"></i>(d) Your contact information;</li>
          <li><i className="fa-solid fa-circle"></i>(e) A statement that you have a good-faith belief that the use is not authorized by the copyright owner, its agent, or the law;</li>
          <li><i className="fa-solid fa-circle"></i>(f) A statement, under penalty of perjury, that the information in the notice is accurate and that you are authorized to act on behalf of the copyright owner.</li>
        </ul>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          <strong>Designated Agent for Notice of Claims of Copyright Infringement:</strong>
          {'\n'}[NAME]
          {'\n'}[ADDRESS]
          {'\n'}<a href="mailto:dmca@doseedo.com">dmca@doseedo.com</a>
          {'\n'}[PHONE]
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          Our Designated Agent is registered with the U.S. Copyright Office.
        </p>

        <h3 className={styles.subsectionTitle}>5.2 Counter-Notice</h3>
        <p className={styles.sectionText}>
          If you believe your content was removed or disabled by mistake or misidentification, you may submit a counter-notice containing the elements required by 17 U.S.C. § 512(g)(3). Submit counter-notices to the Designated Agent above.
        </p>

        <h3 className={styles.subsectionTitle}>5.3 Repeat Infringer Policy</h3>
        <p className={styles.sectionText}>
          We will terminate the accounts of users who are determined to be repeat infringers. Generally, three valid takedown notices within a 12-month period will result in termination, though we may terminate sooner in cases of egregious infringement.
        </p>

        <h3 className={styles.subsectionTitle}>5.4 Knowingly Material Misrepresentations</h3>
        <p className={styles.sectionText}>
          Under 17 U.S.C. § 512(f), any person who knowingly materially misrepresents that material is infringing, or that material was removed or disabled by mistake, may be liable for damages.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>6. AI-Generated Content</h2>

        <h3 className={styles.subsectionTitle}>6.1 How AI Generation Works</h3>
        <p className={styles.sectionText}>
          The Service includes AI features that generate audio stems, MIDI, instrument parts, and other creative content based on inputs you provide (including MIDI conditioning, timbre selections, structural directions, and chatbot instructions). You direct the generation; the Service executes your creative direction.
        </p>

        <h3 className={styles.subsectionTitle}>6.2 Training Data</h3>
        <p className={styles.sectionText}>
          Doseedo's AI models are trained on a proprietary corpus of multitrack sessions commissioned and licensed for this purpose, with rights documented through work-for-hire agreements and equivalent legal instruments. We do not train on user-uploaded content. We do not train on copyrighted material we do not have rights to use.
        </p>

        <h3 className={styles.subsectionTitle}>6.3 Ownership of Generated Content</h3>
        <p className={styles.sectionText}>
          To the extent permitted by law, you own AI-generated content you create through the Service, subject to the rights of others (for example, if your inputs include copyrighted material that influences the output). You are responsible for ensuring your use of generated content complies with applicable law.
        </p>

        <h3 className={styles.subsectionTitle}>6.4 No Guarantee of Originality</h3>
        <p className={styles.sectionText}>
          While we design our generation systems to produce original output, we cannot guarantee that any specific generated content is non-infringing of all third-party rights. You are responsible for evaluating generated content before using or distributing it.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>7. Collaboration Features</h2>

        <h3 className={styles.subsectionTitle}>7.1 How Collaboration Works</h3>
        <p className={styles.sectionText}>
          The Service allows you to invite specific other users to collaborate on specific sessions. Collaboration is by invite only and limited to identified users. The Service does not provide public sharing, public discovery, or broadcast distribution of user sessions.
        </p>

        <h3 className={styles.subsectionTitle}>7.2 Your Responsibilities When Collaborating</h3>
        <p className={styles.sectionText}>
          When you invite a collaborator, you grant them access to the relevant session and authorize the Service to make the session available to them. You represent that you have the right to share the session's contents with the collaborator. You remain responsible for content in shared sessions, including any content uploaded or added by collaborators that you control or could reasonably control.
        </p>

        <h3 className={styles.subsectionTitle}>7.3 Collaborators' Rights</h3>
        <p className={styles.sectionText}>
          Users you invite as collaborators may view, edit, and contribute to the session as you authorize. Collaborators do not gain ownership of your underlying content but may own their own contributions.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>8. Subscriptions and Payment</h2>
        <p className={styles.sectionText}>
          [Subscription terms, payment, refunds, auto-renewal, price changes, and tax handling — to be drafted with your billing system specifics. Standard SaaS subscription language. Should include: subscription tiers, billing cadence, refund policy, auto-renewal disclosure (required by law in many states), price change notice provisions, and tax handling.]
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>9. Third-Party Software and DAW Integration</h2>
        <p className={styles.sectionText}>
          The Service integrates with third-party digital audio workstations and other software. You are responsible for complying with the terms of any third-party software you use in connection with the Service. Doseedo is not affiliated with, endorsed by, or sponsored by Apple Inc., Avid Technology, Inc., Ableton AG, or other DAW vendors. Trademarks belong to their respective owners.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          The Service's interoperability with third-party DAWs is provided for the purpose of enabling you to work with your own session data across the tools you use. You are responsible for your own licenses to third-party software.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>10. Intellectual Property</h2>

        <h3 className={styles.subsectionTitle}>10.1 Doseedo's IP</h3>
        <p className={styles.sectionText}>
          The Service, including its software, models, designs, branding, and documentation, is owned by Doseedo and protected by copyright, trademark, and other laws. We grant you a limited, non-exclusive, non-transferable, revocable license to use the Service in accordance with these Terms.
        </p>

        <h3 className={styles.subsectionTitle}>10.2 Feedback</h3>
        <p className={styles.sectionText}>
          If you provide feedback, suggestions, or ideas about the Service, you grant Doseedo a perpetual, irrevocable, royalty-free license to use them without obligation.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>11. Privacy</h2>
        <p className={styles.sectionText}>
          Our Privacy Policy describes how we collect, use, and share information about you. By using the Service, you consent to the practices described in the Privacy Policy.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>12. Termination</h2>
        <p className={styles.sectionText}>
          You may terminate your account at any time through your account settings. We may suspend or terminate your access to the Service at any time, with or without cause and with or without notice, including for violation of these Terms.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          Upon termination, your right to use the Service ends immediately. We may delete Your Content after termination, except as needed for legal compliance, dispute resolution, or backup retention. You are responsible for exporting Your Content before termination if you wish to retain it.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          Sections that by their nature should survive termination (including ownership, disclaimers, indemnification, limitations of liability, and dispute resolution) will survive.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>13. Disclaimers</h2>
        <p className={styles.sectionText}>
          THE SERVICE IS PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, AND ANY WARRANTIES ARISING FROM COURSE OF DEALING OR USAGE OF TRADE.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          DOSEEDO DOES NOT WARRANT THAT THE SERVICE WILL BE UNINTERRUPTED, ERROR-FREE, OR SECURE; THAT GENERATED CONTENT WILL BE ORIGINAL OR NON-INFRINGING; OR THAT THE SERVICE WILL MEET YOUR REQUIREMENTS.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>14. Limitation of Liability</h2>
        <p className={styles.sectionText}>
          TO THE MAXIMUM EXTENT PERMITTED BY LAW, DOSEEDO AND ITS AFFILIATES, OFFICERS, DIRECTORS, EMPLOYEES, AND AGENTS WILL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING LOSS OF PROFITS, DATA, GOODWILL, OR BUSINESS OPPORTUNITY, ARISING FROM OR RELATED TO THESE TERMS OR THE SERVICE, REGARDLESS OF THE THEORY OF LIABILITY.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          DOSEEDO'S TOTAL LIABILITY ARISING FROM OR RELATED TO THESE TERMS OR THE SERVICE WILL NOT EXCEED THE GREATER OF (A) THE AMOUNTS YOU PAID DOSEEDO IN THE TWELVE MONTHS BEFORE THE CLAIM AROSE OR (B) ONE HUNDRED U.S. DOLLARS.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          These limitations apply even if any remedy fails its essential purpose. Some jurisdictions do not allow limitations on certain damages; in those jurisdictions, the limitations apply to the fullest extent permitted.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>15. Indemnification</h2>
        <p className={styles.sectionText}>
          You will defend, indemnify, and hold harmless Doseedo and its affiliates, officers, directors, employees, and agents from and against any claims, liabilities, damages, losses, and expenses (including reasonable attorneys' fees) arising from or related to:
        </p>
        <ul className={styles.list}>
          <li><i className="fa-solid fa-circle"></i>(a) Your Content, including claims that Your Content infringes a third party's rights;</li>
          <li><i className="fa-solid fa-circle"></i>(b) Your use of the Service;</li>
          <li><i className="fa-solid fa-circle"></i>(c) Your violation of these Terms;</li>
          <li><i className="fa-solid fa-circle"></i>(d) Your violation of any applicable law or third-party right.</li>
        </ul>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          Doseedo will provide you reasonable notice of any claim and reasonable cooperation in defense.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>16. Governing Law and Dispute Resolution</h2>
        <p className={styles.sectionText}>
          These Terms are governed by the laws of the State of Delaware, without regard to conflict of laws principles.
        </p>
        <p className={styles.sectionText} style={{ marginTop: 16 }}>
          [Arbitration / class action waiver / venue clauses — these are jurisdiction-sensitive and require lawyer attention. Consider AAA arbitration, class action waiver, small claims carveout, opt-out window. State-by-state enforceability varies.]
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>17. Changes to These Terms</h2>
        <p className={styles.sectionText}>
          We may update these Terms from time to time. If we make material changes, we will notify you (for example, by email or in-product notice) before the changes take effect. Your continued use of the Service after the changes take effect constitutes acceptance.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>18. General</h2>
        <p className={styles.sectionText}>
          These Terms, together with our Privacy Policy and any other agreements we link to, are the entire agreement between you and Doseedo regarding the Service. If any provision is unenforceable, the rest remains in effect. Our failure to enforce any provision is not a waiver. You may not assign these Terms without our consent. We may assign them in connection with a corporate transaction.
        </p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>19. Contact</h2>
        <p className={styles.sectionText}>
          Questions about these Terms? Contact us at <a href="mailto:legal@doseedo.com">legal@doseedo.com</a>.
        </p>
      </div>

      <div className={styles.contactInfo}>
        <p>Doseedo, Inc. — Marina Del Rey, California</p>
        <a href="mailto:legal@doseedo.com">legal@doseedo.com</a>
      </div>
    </div>
  </div>
);

export default Terms;
