import React from 'react';

/**
 * PageEyebrow — small uppercase mono label rendered just above a
 * subpage's title. Matches the inline eyebrow already used at the top
 * of Tools.js (`§ Tools · Standalone AI utilities`) so every dashboard
 * subpage gets the same hierarchy: PageTopbar → eyebrow → page-title.
 *
 * Usage:
 *   <PageEyebrow section="Tools" description="Standalone AI utilities" />
 */
export default function PageEyebrow({ section, description }) {
  return (
    <div
      style={{
        fontFamily: 'var(--wb-font-mono, ui-monospace, SFMono-Regular, Menlo, monospace)',
        fontSize: 10,
        letterSpacing: 0.8,
        textTransform: 'uppercase',
        color: 'var(--wb-ink-mute, rgba(255,255,255,0.5))',
        marginBottom: 12,
      }}
    >
      § {section} · {description}
    </div>
  );
}
