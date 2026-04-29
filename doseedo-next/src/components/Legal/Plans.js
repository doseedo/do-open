import React, { useEffect, useState } from 'react';

/**
 * Plans / Pricing page — served at /plans.
 *
 * Paper/technical workbench aesthetic matching the dashboard mock at
 * daw/plans.jsx (Inter / JetBrains Mono / Lora on a warm off-white).
 * App.js wraps this component with <LeftSidebar/> so we only render the
 * content region (hero + plans grid + free strip + matrix + FAQ + closing).
 *
 * Provision buttons POST to /api/billing/checkout, which is rewritten by
 * next.config.js to the Fly auth-service. The auth-service owns the
 * Stripe SDK + webhook + tier table; this client just sends the chosen
 * { tier, billing } pair and follows the returned Checkout URL.
 */

async function startCheckout({ tier, billing }) {
  const r = await fetch('/api/billing/checkout', {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tier, billing }),
  });
  if (r.status === 401) {
    // Not signed in — send through Clerk and come back to /plans.
    window.location.href = `/sign-in?redirect_url=${encodeURIComponent('/plans')}`;
    return;
  }
  if (!r.ok) {
    let msg = `Checkout failed (${r.status})`;
    try {
      const body = await r.json();
      if (body?.error) msg = body.error;
    } catch {}
    // eslint-disable-next-line no-alert
    alert(msg);
    return;
  }
  const { url } = await r.json();
  if (url) window.location.href = url;
}

const C = {
  bg: '#e8e6e1',
  surface: '#f2f0ea',
  surface2: '#dcd9d1',
  ink: '#15181c',
  inkSoft: 'rgba(21,24,28,0.66)',
  inkMute: 'rgba(21,24,28,0.40)',
  inkFaint: 'rgba(21,24,28,0.22)',
  rule: 'rgba(21,24,28,0.14)',
  ruleStrong: 'rgba(21,24,28,0.30)',
  accent: '#1d4c7a',
  warm: '#c94f2c',
  purple: '#AAB0EE',
  sans: '"Inter",system-ui,sans-serif',
  mono: '"JetBrains Mono",ui-monospace,Menlo,monospace',
  head: '"Lora",Georgia,serif',
};

const arrowPath = 'M5 12h14 M13 6l6 6-6 6';

const Arrow = ({ size = 13, stroke = 1.8, color = 'currentColor' }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke={color}
    strokeWidth={stroke}
    strokeLinecap="round"
    strokeLinejoin="round"
    style={{ flexShrink: 0 }}
  >
    <path d={arrowPath} />
  </svg>
);

function Topbar({ billing, setBilling }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 14,
        padding: '0 36px',
        height: 48,
        borderBottom: `1px solid ${C.rule}`,
        background: C.surface,
        fontFamily: C.mono,
        fontSize: 10,
        letterSpacing: 0.6,
        textTransform: 'uppercase',
        color: C.inkMute,
        flexWrap: 'wrap',
      }}
    >
      <span>Dashboard</span>
      <span style={{ color: C.inkFaint }}>/</span>
      <span style={{ color: C.inkSoft }}>Info</span>
      <span style={{ color: C.inkFaint }}>/</span>
      <span>
        <strong style={{ color: C.inkSoft, fontWeight: 500 }}>Plans</strong>
      </span>
      <span style={{ color: C.inkFaint }}>·</span>
      <span>3 tiers · free tier available</span>
      <div style={{ flex: 1 }} />
      <div style={{ display: 'inline-flex', border: `1px solid ${C.rule}`, background: C.bg }}>
        {[
          ['monthly', 'Monthly'],
          ['yearly', 'Yearly − 20%'],
        ].map(([v, l]) => (
          <button
            key={v}
            type="button"
            onClick={() => setBilling(v)}
            style={{
              padding: '4px 10px',
              fontFamily: C.mono,
              fontSize: 10,
              letterSpacing: 0.6,
              textTransform: 'uppercase',
              background: billing === v ? C.ink : 'transparent',
              color: billing === v ? C.bg : C.inkSoft,
              borderRight: v === 'monthly' ? `1px solid ${C.rule}` : 'none',
              borderTop: 'none',
              borderLeft: 'none',
              borderBottom: 'none',
              cursor: 'pointer',
            }}
          >
            {l}
          </button>
        ))}
      </div>
      <span style={{ color: C.inkFaint }}>·</span>
      <span>
        billing · <strong style={{ color: C.inkSoft, fontWeight: 500 }}>stripe</strong>
      </span>
    </div>
  );
}

function SectHead({ title, count, right }) {
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, marginBottom: 18, flexWrap: 'wrap' }}>
      <h2 style={{ fontFamily: C.head, fontSize: 20, fontWeight: 600, letterSpacing: -0.3, margin: 0 }}>{title}</h2>
      {count && (
        <span style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: C.inkMute }}>
          {count}
        </span>
      )}
      <div style={{ flex: 1 }} />
      {right}
    </div>
  );
}

function Hero() {
  return (
    <section style={{ marginBottom: 40, paddingBottom: 28, borderBottom: `1px solid ${C.rule}` }}>
      <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.8, textTransform: 'uppercase', color: C.inkMute, marginBottom: 12 }}>
        § Plans &middot; Choose a tier
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1.3fr) minmax(0,1fr)', gap: 60, alignItems: 'end' }}>
        <h1 className="page-title">
          Three plans for producers who want&nbsp;the <span style={{ color: C.accent }}>full studio.</span>
        </h1>
        <div style={{ fontFamily: C.sans, fontSize: 13, color: C.inkSoft, lineHeight: 1.6, paddingBottom: 4, maxWidth: 420 }}>
          One credit covers a generation, a chat session, or an attestation. Sessions, devices, and commercial rights scale with the tier. Upgrade, downgrade, or cancel from the terminal at any time — prorated to the minute.
        </div>
      </div>
    </section>
  );
}

function PlanCard({ plan, billing, featured }) {
  const price = billing === 'yearly' ? Math.round(plan.price * 0.8) : plan.price;
  const yearTotal = price * 12;
  const savings = (plan.price - price) * 12;
  const [submitting, setSubmitting] = useState(false);
  const onProvision = async () => {
    if (submitting) return;
    setSubmitting(true);
    try {
      await startCheckout({ tier: plan.name.toLowerCase(), billing });
    } finally {
      setSubmitting(false);
    }
  };
  return (
    <div
      style={{
        background: featured ? C.ink : C.surface,
        color: featured ? C.bg : C.ink,
        border: `1px solid ${featured ? C.ink : C.rule}`,
        padding: '20px 22px 22px',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: -1,
          right: -1,
          background: featured ? C.purple : C.ink,
          color: featured ? C.ink : C.purple,
          fontFamily: C.mono,
          fontSize: 10,
          fontWeight: 600,
          letterSpacing: 0.5,
          padding: '3px 8px',
        }}
      >
        {plan.sku}
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 14, flexWrap: 'wrap' }}>
        <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.8, textTransform: 'uppercase', color: featured ? 'rgba(232,230,225,.55)' : C.inkMute }}>
          Tier · {plan.tier}
        </div>
        {featured && (
          <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.8, textTransform: 'uppercase', color: C.purple, background: 'rgba(170,176,238,.12)', padding: '2px 6px', border: `1px solid ${C.purple}55` }}>
            recommended
          </div>
        )}
      </div>
      <div style={{ fontFamily: C.head, fontSize: 30, fontWeight: 600, letterSpacing: -0.6, lineHeight: 1, marginBottom: 8 }}>{plan.name}</div>
      <div style={{ fontFamily: C.sans, fontSize: 13, color: featured ? 'rgba(232,230,225,.7)' : C.inkSoft, lineHeight: 1.5, marginBottom: 22, maxWidth: 360 }}>
        {plan.tagline}
      </div>

      <div
        style={{
          background: featured ? '#0a0c0f' : C.bg,
          border: `1px solid ${featured ? 'rgba(255,255,255,.1)' : C.rule}`,
          padding: '16px 18px',
          display: 'flex',
          alignItems: 'baseline',
          gap: 8,
          marginBottom: 6,
          flexWrap: 'wrap',
        }}
      >
        <div style={{ fontFamily: C.mono, fontWeight: 500, fontSize: 48, letterSpacing: -2, lineHeight: 1, color: featured ? C.purple : C.ink, fontFeatureSettings: '"tnum"' }}>
          ${price}
        </div>
        <div style={{ fontFamily: C.mono, fontSize: 11, letterSpacing: 0.4, color: featured ? 'rgba(232,230,225,.5)' : C.inkMute }}>USD / mo</div>
        <div style={{ flex: 1 }} />
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontFamily: C.mono, fontSize: 9, letterSpacing: 0.6, textTransform: 'uppercase', color: featured ? 'rgba(232,230,225,.5)' : C.inkMute }}>
            {billing === 'yearly' ? 'billed yearly' : 'billed monthly'}
          </div>
          <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.3, color: featured ? 'rgba(232,230,225,.66)' : C.inkSoft, marginTop: 2, fontFeatureSettings: '"tnum"' }}>
            {billing === 'yearly' ? `$${yearTotal}/yr` : `$${price * 12}/yr`}
            {billing === 'yearly' && savings > 0 && (
              <span style={{ color: featured ? C.purple : C.warm, marginLeft: 6 }}>−${savings}</span>
            )}
          </div>
        </div>
      </div>

      <button
        type="button"
        onClick={onProvision}
        disabled={submitting}
        style={{
          background: featured ? C.purple : C.ink,
          color: featured ? C.ink : C.bg,
          border: 0,
          padding: '12px 16px',
          marginTop: 16,
          marginBottom: 22,
          fontFamily: C.mono,
          fontSize: 11,
          fontWeight: 600,
          letterSpacing: 0.8,
          textTransform: 'uppercase',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: submitting ? 'wait' : 'pointer',
          opacity: submitting ? 0.6 : 1,
          gap: 12,
        }}
      >
        <span>{submitting ? 'Connecting Stripe…' : plan.cta}</span>
        <Arrow size={13} color={featured ? C.ink : C.bg} stroke={1.8} />
      </button>

      <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.8, textTransform: 'uppercase', color: featured ? 'rgba(232,230,225,.5)' : C.inkMute, marginBottom: 4 }}>
          Spec sheet
        </div>
        {plan.specs.map((s, i) => (
          <div
            key={i}
            style={{
              display: 'grid',
              gridTemplateColumns: '110px 1fr',
              gap: 14,
              padding: '10px 0',
              borderTop: `1px dashed ${featured ? 'rgba(232,230,225,.18)' : C.rule}`,
              fontSize: 12,
              lineHeight: 1.45,
              alignItems: 'baseline',
            }}
          >
            <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: featured ? 'rgba(232,230,225,.55)' : C.inkMute, whiteSpace: 'nowrap' }}>
              {s.k}
            </div>
            <div
              style={{
                color: s.muted ? (featured ? 'rgba(232,230,225,.45)' : C.inkMute) : (featured ? C.bg : C.ink),
                fontStyle: s.muted ? 'italic' : 'normal',
                fontFamily: C.sans,
                fontSize: 13,
                fontWeight: s.bold ? 500 : 400,
              }}
            >
              {s.v}
            </div>
          </div>
        ))}
      </div>
      <div
        style={{
          marginTop: 18,
          paddingTop: 14,
          borderTop: `1px solid ${featured ? 'rgba(232,230,225,.18)' : C.rule}`,
          display: 'flex',
          justifyContent: 'space-between',
          fontFamily: C.mono,
          fontSize: 10,
          letterSpacing: 0.5,
          textTransform: 'uppercase',
          color: featured ? 'rgba(232,230,225,.5)' : C.inkMute,
        }}
      >
        <span>rev. {plan.rev}</span>
        <span>commercial license ✓</span>
      </div>
    </div>
  );
}

const PLANS = [
  {
    sku: 'DSD-PRO-012',
    tier: 'I',
    rev: '2026.04',
    name: 'Pro',
    tagline: 'For songwriters and producers working on a song or three at a time. Everything you need to ship, nothing you don’t.',
    price: 12,
    cta: 'Provision Pro',
    specs: [
      { k: 'Credits', v: '800 / month · 2× rollover', bold: true },
      { k: 'Sessions', v: 'Unlimited · every take kept' },
      { k: 'Storage', v: '5 GB · Opus 128k stems' },
      { k: 'Generation', v: 'Standard queue' },
      { k: 'Instruments', v: 'Curated instrument set' },
      { k: 'Stem export', v: '48 kHz · 24-bit WAV · stems + mixdown' },
      { k: 'Stem split', v: '— not included', muted: true },
      { k: 'Score-to-pic', v: '— not included', muted: true },
      { k: 'Collab seats', v: '1 (you)' },
      { k: 'Devices', v: '2 simultaneous' },
      { k: 'Support', v: 'Email · 48h SLA' },
    ],
  },
  {
    sku: 'DSD-STD-029',
    tier: 'II',
    rev: '2026.04',
    name: 'Studio',
    tagline: 'For working producers and small studios. Stem split, score-to-picture, and priority compute on the floor when you need it.',
    price: 29,
    cta: 'Provision Studio',
    specs: [
      { k: 'Credits', v: '3,000 / month · 2× rollover', bold: true },
      { k: 'Sessions', v: 'Unlimited · every take kept' },
      { k: 'Storage', v: '25 GB · Opus 128k stems' },
      { k: 'Generation', v: 'Priority · 2× faster', bold: true },
      { k: 'Instruments', v: 'Full set + early access models' },
      { k: 'Stem export', v: '48 / 96 kHz · 24-bit · stems + mix + master' },
      { k: 'Stem split', v: 'Included · unlimited splits', bold: true },
      { k: 'Score-to-pic', v: 'Frame-accurate cues · video import', bold: true },
      { k: 'Collab seats', v: 'Up to 5 · shared sessions, comments' },
      { k: 'Devices', v: 'Unlimited' },
      { k: 'Support', v: 'Studio hours · 8h SLA' },
    ],
  },
  {
    sku: 'DSD-PWR-079',
    tier: 'III',
    rev: '2026.04',
    name: 'Power',
    tagline: 'For agencies, labels, and producers who live inside doseedo. Unlimited seats, API access, and per-credit overage so the studio never sleeps.',
    price: 79,
    cta: 'Provision Power',
    specs: [
      { k: 'Credits', v: '10,000 / month · 2× rollover', bold: true },
      { k: 'Overage', v: '$0.011 / credit · usage-billed' },
      { k: 'Sessions', v: 'Unlimited · every take kept' },
      { k: 'Storage', v: '100 GB · Opus 128k stems' },
      { k: 'Generation', v: 'Priority · 4× faster · queue jump', bold: true },
      { k: 'API access', v: 'Programmatic gens + stem split', bold: true },
      { k: 'Stem export', v: '96 kHz · 24-bit · stems + mix + master' },
      { k: 'Stem split', v: 'Unlimited · batch via API', bold: true },
      { k: 'Score-to-pic', v: 'Frame-accurate · video import' },
      { k: 'Collab seats', v: 'Unlimited' },
      { k: 'Devices', v: 'Unlimited' },
      { k: 'Support', v: 'Dedicated channel · 4h SLA' },
    ],
  },
];

function PlansGrid({ billing }) {
  return (
    <section style={{ marginBottom: 40 }}>
      <SectHead
        title="Available tiers"
        count="2 plans"
        right={
          <span style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: C.inkMute }}>
            prices · USD · ex. tax
          </span>
        }
      />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 14 }}>
        <PlanCard plan={PLANS[0]} billing={billing} featured={false} />
        <PlanCard plan={PLANS[1]} billing={billing} featured />
        <PlanCard plan={PLANS[2]} billing={billing} featured={false} />
      </div>
    </section>
  );
}

function FreeStrip() {
  return (
    <section style={{ marginBottom: 40 }}>
      <div
        style={{
          background: C.surface,
          border: `1px dashed ${C.ruleStrong}`,
          padding: '18px 22px',
          display: 'grid',
          gridTemplateColumns: 'auto minmax(0,1fr) auto auto',
          gap: 24,
          alignItems: 'center',
        }}
      >
        <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.8, textTransform: 'uppercase', color: C.inkMute, whiteSpace: 'nowrap' }}>
          DSD-FREE-000
        </div>
        <div>
          <div style={{ fontFamily: C.sans, fontSize: 15, fontWeight: 600, letterSpacing: -0.1, marginBottom: 2 }}>
            Free tier · no card required
          </div>
          <div style={{ fontFamily: C.sans, fontSize: 12, color: C.inkSoft, lineHeight: 1.5 }}>
            100 credits / month · 200 MB storage · 30-day rolling retention · 1 device · watermarked exports. Upgrade anytime — your sessions come with you.
          </div>
        </div>
        <div style={{ fontFamily: C.mono, fontSize: 11, letterSpacing: 0.5, color: C.inkMute, textAlign: 'right' }}>
          $0<span style={{ fontSize: 9, marginLeft: 4 }}>/MO</span>
        </div>
        <button
          type="button"
          style={{
            padding: '8px 12px',
            fontFamily: C.mono,
            fontSize: 10,
            letterSpacing: 0.8,
            textTransform: 'uppercase',
            background: C.bg,
            border: `1px solid ${C.rule}`,
            color: C.ink,
            cursor: 'pointer',
            whiteSpace: 'nowrap',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <span>Open free room</span>
          <Arrow size={11} stroke={1.8} />
        </button>
      </div>
    </section>
  );
}

function Matrix() {
  // [feature, Pro, Studio, Power]
  const rows = [
    ['Credits / month', '800', '3,000', '10,000'],
    ['Rollover', '2× cap', '2× cap', '2× cap'],
    ['Overage', '—', '—', '$0.011 / credit'],
    ['Storage', '5 GB', '25 GB', '100 GB'],
    ['Generation speed', 'Standard', 'Priority · 2×', 'Priority · 4× + jump'],
    ['Instruments', 'Curated', 'Full + early', 'Full + early'],
    ['Stem separation', '—', 'Unlimited', 'Unlimited + API'],
    ['Score-to-picture', '—', 'Included', 'Included'],
    ['Sample rate', '48 kHz', '48 / 96 kHz', '96 kHz'],
    ['API access', '—', '—', 'Included'],
    ['Collaborators', '1', 'Up to 5', 'Unlimited'],
    ['Devices', '2', 'Unlimited', 'Unlimited'],
    ['Commercial use', 'Yes', 'Yes', 'Yes'],
    ['Support SLA', '48 h', '8 h · studio', '4 h · dedicated'],
  ];
  return (
    <section style={{ marginBottom: 40 }}>
      <SectHead
        title="Spec matrix"
        count="feature · tier · value"
        right={
          <span style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: C.inkMute }}>
            ref. DSD-PRICE-02
          </span>
        }
      />
      <div style={{ background: C.surface, border: `1px solid ${C.rule}`, overflowX: 'auto' }}>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1.4fr 1fr 1fr 1fr',
            background: C.surface2,
            borderBottom: `1px solid ${C.rule}`,
            minWidth: 720,
          }}
        >
          <div style={{ padding: '10px 16px', fontFamily: C.mono, fontSize: 10, letterSpacing: 0.7, textTransform: 'uppercase', color: C.inkMute }}>Feature</div>
          <div style={{ padding: '10px 16px', fontFamily: C.mono, fontSize: 10, letterSpacing: 0.7, textTransform: 'uppercase', color: C.ink, borderLeft: `1px solid ${C.rule}` }}>Pro · $12</div>
          <div style={{ padding: '10px 16px', fontFamily: C.mono, fontSize: 10, letterSpacing: 0.7, textTransform: 'uppercase', color: C.ink, borderLeft: `1px solid ${C.rule}`, background: 'rgba(170,176,238,.16)' }}>
            Studio · $29 <span style={{ color: C.purple }}>●</span>
          </div>
          <div style={{ padding: '10px 16px', fontFamily: C.mono, fontSize: 10, letterSpacing: 0.7, textTransform: 'uppercase', color: C.ink, borderLeft: `1px solid ${C.rule}` }}>Power · $79</div>
        </div>
        {rows.map((r, i) => (
          <div
            key={i}
            style={{
              display: 'grid',
              gridTemplateColumns: '1.4fr 1fr 1fr 1fr',
              borderBottom: i === rows.length - 1 ? 'none' : `1px solid ${C.rule}`,
              minWidth: 720,
            }}
          >
            <div style={{ padding: '11px 16px', fontFamily: C.sans, fontSize: 13, color: C.ink }}>{r[0]}</div>
            <div style={{ padding: '11px 16px', fontFamily: C.mono, fontSize: 11, letterSpacing: 0.2, color: r[1] === '—' ? C.inkMute : C.inkSoft, borderLeft: `1px solid ${C.rule}` }}>{r[1]}</div>
            <div style={{ padding: '11px 16px', fontFamily: C.mono, fontSize: 11, letterSpacing: 0.2, color: r[2] === '—' ? C.inkMute : C.ink, borderLeft: `1px solid ${C.rule}`, background: 'rgba(170,176,238,.08)', fontWeight: 500 }}>{r[2]}</div>
            <div style={{ padding: '11px 16px', fontFamily: C.mono, fontSize: 11, letterSpacing: 0.2, color: r[3] === '—' ? C.inkMute : C.inkSoft, borderLeft: `1px solid ${C.rule}` }}>{r[3]}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

function FAQ() {
  const items = [
    { q: 'How do credits work?', a: 'One credit covers one audio generation, five chat turns, or one Polygon attestation. Credits refill at the start of each billing cycle. Unused credits roll over up to 2× your monthly allotment — anything beyond that resets, so the studio never builds up infinite back-stock.' },
    { q: 'What happens if I run out of credits?', a: 'Pro and Studio pause new generations until the next cycle (or you upgrade). Power keeps running and bills overage at $0.011 per credit on your usage line. Existing sessions, exports, and playback are never gated by credits — only new generation work.' },
    { q: 'Can I switch tiers mid-cycle?', a: 'Yes. Upgrades are prorated to the minute and billed immediately, and your credit balance is topped up to the new tier. Downgrades take effect at the start of the next cycle, so you never lose credits you’ve already paid for.' },
    { q: 'What happens to my sessions if I cancel?', a: 'Sessions move to read-only for 12 months. You can export stems, resume on the free tier, or come back to a paid plan — nothing is deleted without explicit confirmation.' },
    { q: 'Is the license really commercial?', a: 'Yes, on every paid tier. Release on DSPs, sync to film, publish — no per-track fees, no surprise carve-outs. The exported audio is yours.' },
    { q: 'Are there student / education rates?', a: 'Pro and Studio are 50% off with a verified .edu address or institutional email. Power is flat rate — the compute cost is the compute cost.' },
    { q: 'Do I have to install anything?', a: 'No. doseedo runs in the browser. A desktop build (macOS / Windows) is available on every paid tier and syncs to the same cloud sessions.' },
  ];
  const [open, setOpen] = useState(0);
  return (
    <section style={{ marginBottom: 40 }}>
      <SectHead title="Frequently asked" count="7 · signed by billing team" />
      <div style={{ background: C.surface, border: `1px solid ${C.rule}` }}>
        {items.map((it, i) => (
          <div key={i} style={{ borderTop: i > 0 ? `1px solid ${C.rule}` : 'none' }}>
            <button
              type="button"
              onClick={() => setOpen(open === i ? -1 : i)}
              style={{
                width: '100%',
                display: 'grid',
                gridTemplateColumns: 'auto 1fr auto',
                gap: 16,
                alignItems: 'baseline',
                padding: '14px 18px',
                cursor: 'pointer',
                textAlign: 'left',
                background: 'transparent',
                border: 'none',
                color: 'inherit',
              }}
            >
              <span style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.6, color: C.inkMute }}>
                Q.{String(i + 1).padStart(2, '0')}
              </span>
              <span style={{ fontFamily: C.sans, fontSize: 14, fontWeight: 500, color: C.ink }}>{it.q}</span>
              <span
                style={{
                  fontFamily: C.mono,
                  fontSize: 13,
                  color: C.inkSoft,
                  transform: open === i ? 'rotate(45deg)' : 'none',
                  transition: 'transform .15s',
                }}
              >
                +
              </span>
            </button>
            {open === i && (
              <div style={{ padding: '0 18px 16px', display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 16 }}>
                <span style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.6, color: C.inkMute }}>
                  A.{String(i + 1).padStart(2, '0')}
                </span>
                <span style={{ fontFamily: C.sans, fontSize: 13, color: C.inkSoft, lineHeight: 1.6, maxWidth: 720 }}>
                  {it.a}
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function Closing() {
  return (
    <section style={{ marginBottom: 20 }}>
      <div
        style={{
          background: C.ink,
          color: C.bg,
          padding: '28px 32px',
          display: 'grid',
          gridTemplateColumns: 'minmax(0,1fr) auto',
          gap: 32,
          alignItems: 'center',
        }}
      >
        <div>
          <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.8, textTransform: 'uppercase', color: 'rgba(232,230,225,.5)', marginBottom: 10 }}>
            § Provision
          </div>
          <div style={{ fontFamily: C.head, fontSize: 28, fontWeight: 600, letterSpacing: -0.6, lineHeight: 1.15, maxWidth: 640 }}>
            Pick a tier. <span style={{ color: C.purple }}>The studio is already warm.</span>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <button
            type="button"
            style={{
              padding: '12px 18px',
              fontFamily: C.mono,
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: 0.8,
              textTransform: 'uppercase',
              background: 'transparent',
              border: '1px solid rgba(232,230,225,.3)',
              color: C.bg,
              cursor: 'pointer',
            }}
          >
            Provision Power · $79
          </button>
          <button
            type="button"
            style={{
              padding: '12px 18px',
              fontFamily: C.mono,
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: 0.8,
              textTransform: 'uppercase',
              background: C.purple,
              border: 'none',
              color: C.ink,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 10,
            }}
          >
            <span>Provision Studio · $29</span>
            <Arrow size={13} stroke={1.8} color={C.ink} />
          </button>
        </div>
      </div>
      <div
        style={{
          marginTop: 10,
          display: 'flex',
          justifyContent: 'space-between',
          fontFamily: C.mono,
          fontSize: 10,
          letterSpacing: 0.5,
          textTransform: 'uppercase',
          color: C.inkMute,
          flexWrap: 'wrap',
          gap: 8,
        }}
      >
        <span>doseedo billing · stripe · 2026</span>
        <span>cancel any time · prorated · no dark patterns</span>
      </div>
    </section>
  );
}

// Inject the workbench Google Fonts once — Inter / JetBrains Mono / Lora
// aren't bundled globally by Next, and this page leans on them hard.
function useWorkbenchFonts() {
  useEffect(() => {
    const href =
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&family=Lora:wght@500;600;700&display=swap';
    if (typeof document === 'undefined') return;
    if (document.querySelector(`link[href="${href}"]`)) return;
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = href;
    document.head.appendChild(link);
  }, []);
}

const Plans = () => {
  const [billing, setBilling] = useState('monthly');
  useWorkbenchFonts();

  return (
    <main
      style={{
        // 220px clears the fixed LeftSidebar on dashboard routes; without
        // this the sidebar overlays the topbar + first content column.
        marginLeft: 220,
        minWidth: 0,
        display: 'flex',
        flexDirection: 'column',
        background: C.bg,
        color: C.ink,
        fontFamily: C.sans,
        fontSize: 13,
        minHeight: '100vh',
        flex: 1,
      }}
    >
      <Topbar billing={billing} setBilling={setBilling} />
      <div style={{ flex: 1, overflow: 'auto', padding: '36px 40px 80px', maxWidth: 1200, width: '100%', margin: '0 auto', boxSizing: 'border-box' }}>
        <Hero />
        <PlansGrid billing={billing} />
        <FreeStrip />
        <Matrix />
        <FAQ />
        <Closing />
      </div>
    </main>
  );
};

export default Plans;
