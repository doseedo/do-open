import React, { useEffect, useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

/**
 * Docs page — served at /docs and /docs/<slug>.
 *
 * Single component, internal section routing. The left TOC is a sticky
 * navigation rail; the right column renders the active subpage. URLs are
 * the source of truth (so deep-linking + browser back/forward work) — we
 * read location.pathname and switch sections on it. App.js applies the
 * workbench cream body class for /docs paths and wraps this with the
 * single LeftSidebar instance.
 *
 * The subpages here are intentionally sketches, not final copy: each one
 * carries enough orientation to be useful, with TODO blocks where deeper
 * walkthroughs need to land. Treat this file as the doc skeleton — fill
 * in the bodies as features stabilise.
 */

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
  ok: '#2f6b4e',
  sans: '"Inter",system-ui,sans-serif',
  mono: '"JetBrains Mono",ui-monospace,Menlo,monospace',
  head: '"Lora",Georgia,serif',
};

// ---- Section catalog ---------------------------------------------------
// Each entry is {slug, title, blurb}. SECTIONS groups them for the TOC.
// Adding a page = add the slug to SECTIONS and a renderer in PAGES.

const SECTIONS = [
  {
    label: 'Getting Started',
    items: [
      { slug: 'introduction', title: 'Introduction',  blurb: 'What Doseedo is and isn’t.' },
      { slug: 'quickstart',   title: 'Quickstart',    blurb: 'From sign-in to first stem.' },
      { slug: 'account',      title: 'Account & Auth',blurb: 'Sign-in, devices, sessions.' },
    ],
  },
  {
    label: 'Workspace',
    items: [
      { slug: 'studio',     title: 'Studio (DAW)',     blurb: 'Layout, transport, mixer.' },
      { slug: 'sessions',   title: 'Sessions & Projects', blurb: 'Autosave, sharing, retention.' },
      { slug: 'navigation', title: 'Navigation',       blurb: 'Sidebar, search, bookmarks.' },
      { slug: 'shortcuts',  title: 'Keyboard Shortcuts', blurb: 'Full key reference.' },
    ],
  },
  {
    label: 'Creation',
    items: [
      { slug: 'tracks',     title: 'Tracks & Mixer',    blurb: 'Buses, gain, pan, sends.' },
      { slug: 'midi',       title: 'MIDI Editing',      blurb: 'Notes, automation, export.' },
      { slug: 'audio',      title: 'Audio & Stems',     blurb: 'Waveform, regions, comping.' },
      { slug: 'plugins',    title: 'Plugins & Instruments', blurb: 'Browse, load, chain.' },
      { slug: 'models',     title: 'Models Catalog',    blurb: 'Which model for which job.' },
    ],
  },
  {
    label: 'AI Features',
    items: [
      { slug: 'generation',    title: 'Generation',        blurb: 'Prompts, presets, queues.' },
      { slug: 'polypitch',     title: 'Polypitch (Stem Split)', blurb: 'Mask-based stem isolation.' },
      { slug: 'video-scoring', title: 'Video Scoring',     blurb: 'Score-to-picture cues.' },
      { slug: 'chat',          title: 'Chat Assistant',    blurb: 'Qwen3-14B via Modal vLLM.' },
    ],
  },
  {
    label: 'Sharing',
    items: [
      { slug: 'search',   title: 'Search',          blurb: 'MIDI + content discovery.' },
      { slug: 'profiles', title: 'Public Profiles', blurb: 'Your /profile page.' },
      { slug: 'sharing',  title: 'Sharing & Export',blurb: 'Opus stems, mixdowns, links.' },
    ],
  },
  {
    label: 'Account & Reference',
    items: [
      { slug: 'plans',   title: 'Plans & Billing', blurb: 'Tiers, credits, overage.' },
      { slug: 'privacy', title: 'Privacy & Data',  blurb: 'What we store and where.' },
      { slug: 'api',     title: 'API Access',      blurb: 'Programmatic gens (Power).' },
      { slug: 'faq',     title: 'FAQ',             blurb: 'Common questions.' },
      { slug: 'contact', title: 'Contact & Support', blurb: 'How to reach us.' },
    ],
  },
];

// Flat slug → {section, item} lookup.
const FLAT = SECTIONS.flatMap((s) => s.items.map((it) => ({ ...it, section: s.label })));
const SLUG_INDEX = Object.fromEntries(FLAT.map((it, i) => [it.slug, i]));

// ---- Shared atoms -----------------------------------------------------

function Topbar({ section, title }) {
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
      <span>Docs</span>
      {section && (
        <>
          <span style={{ color: C.inkFaint }}>/</span>
          <span style={{ color: C.inkSoft }}>{section}</span>
        </>
      )}
      {title && (
        <>
          <span style={{ color: C.inkFaint }}>/</span>
          <strong style={{ color: C.inkSoft, fontWeight: 500 }}>{title}</strong>
        </>
      )}
      <div style={{ flex: 1 }} />
      <span>v0.3 · last revised 2026-04-28</span>
    </div>
  );
}

function PageHead({ kicker, title, lede }) {
  return (
    <header style={{ marginBottom: 32, paddingBottom: 22, borderBottom: `1px solid ${C.rule}` }}>
      <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.8, textTransform: 'uppercase', color: C.inkMute, marginBottom: 10 }}>
        § {kicker}
      </div>
      <h1 className="page-title">
        {title}
      </h1>
      {lede && (
        <p style={{ fontFamily: C.sans, fontSize: 15, color: C.inkSoft, lineHeight: 1.6, marginTop: 14, marginBottom: 0, maxWidth: 720 }}>
          {lede}
        </p>
      )}
    </header>
  );
}

function H2({ children }) {
  return (
    <h2 style={{
      fontFamily: C.head, fontSize: 22, fontWeight: 600, letterSpacing: -0.3,
      margin: '32px 0 14px', color: C.ink,
    }}>
      {children}
    </h2>
  );
}

function P({ children }) {
  return (
    <p style={{ fontFamily: C.sans, fontSize: 14, color: C.inkSoft, lineHeight: 1.7, margin: '0 0 14px', maxWidth: 720 }}>
      {children}
    </p>
  );
}

function Bullets({ items }) {
  return (
    <ul style={{ listStyle: 'none', padding: 0, margin: '6px 0 18px', display: 'flex', flexDirection: 'column', gap: 8, maxWidth: 720 }}>
      {items.map((t, i) => (
        <li key={i} style={{
          display: 'grid', gridTemplateColumns: '14px 1fr', gap: 10,
          fontFamily: C.sans, fontSize: 14, color: C.inkSoft, lineHeight: 1.6,
        }}>
          <span style={{ fontFamily: C.mono, color: C.inkMute, fontSize: 11, lineHeight: '22px' }}>·</span>
          <span>{t}</span>
        </li>
      ))}
    </ul>
  );
}

function Note({ kind = 'info', title, children }) {
  const map = {
    info: { fg: C.accent, label: 'Note' },
    warn: { fg: C.warm,   label: 'Heads up' },
    ok:   { fg: C.ok,     label: 'Tip' },
  }[kind] || { fg: C.accent, label: 'Note' };
  return (
    <div style={{
      background: C.surface,
      border: `1px solid ${C.rule}`,
      borderLeft: `3px solid ${map.fg}`,
      padding: '12px 16px',
      margin: '12px 0 18px',
      maxWidth: 720,
    }}>
      <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: map.fg, marginBottom: 6 }}>
        {title || map.label}
      </div>
      <div style={{ fontFamily: C.sans, fontSize: 13, color: C.inkSoft, lineHeight: 1.6 }}>
        {children}
      </div>
    </div>
  );
}

function Code({ children }) {
  return (
    <code style={{
      fontFamily: C.mono, fontSize: 12, background: C.surface,
      border: `1px solid ${C.rule}`, padding: '1px 6px',
      color: C.ink,
    }}>
      {children}
    </code>
  );
}

function NextPrev({ slug }) {
  const navigate = useNavigate();
  const idx = SLUG_INDEX[slug];
  const prev = idx > 0 ? FLAT[idx - 1] : null;
  const next = idx < FLAT.length - 1 ? FLAT[idx + 1] : null;
  return (
    <div style={{
      marginTop: 40, paddingTop: 18, borderTop: `1px solid ${C.rule}`,
      display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12,
    }}>
      {prev ? (
        <button type="button" onClick={() => navigate(`/docs/${prev.slug}`)} style={navButton('left')}>
          <span style={{ fontFamily: C.mono, fontSize: 9, letterSpacing: 0.8, textTransform: 'uppercase', color: C.inkMute }}>
            ← {prev.section}
          </span>
          <span style={{ fontFamily: C.sans, fontSize: 14, fontWeight: 500, color: C.ink, marginTop: 4 }}>
            {prev.title}
          </span>
        </button>
      ) : <div />}
      {next ? (
        <button type="button" onClick={() => navigate(`/docs/${next.slug}`)} style={navButton('right')}>
          <span style={{ fontFamily: C.mono, fontSize: 9, letterSpacing: 0.8, textTransform: 'uppercase', color: C.inkMute }}>
            {next.section} →
          </span>
          <span style={{ fontFamily: C.sans, fontSize: 14, fontWeight: 500, color: C.ink, marginTop: 4 }}>
            {next.title}
          </span>
        </button>
      ) : <div />}
    </div>
  );
}

function navButton(align) {
  return {
    background: C.surface,
    border: `1px solid ${C.rule}`,
    padding: '12px 16px',
    cursor: 'pointer',
    textAlign: align,
    display: 'flex',
    flexDirection: 'column',
    alignItems: align === 'right' ? 'flex-end' : 'flex-start',
    gap: 0,
    fontFamily: 'inherit',
    color: 'inherit',
  };
}

// ---- TOC --------------------------------------------------------------

function TOC({ activeSlug }) {
  const navigate = useNavigate();
  return (
    <aside style={{
      width: 260, flexShrink: 0,
      borderRight: `1px solid ${C.rule}`,
      background: C.surface,
      overflowY: 'auto',
      position: 'sticky',
      top: 48,
      alignSelf: 'flex-start',
      maxHeight: 'calc(100vh - 48px)',
      padding: '28px 0',
    }}>
      <div style={{
        padding: '0 22px 16px', borderBottom: `1px solid ${C.rule}`,
        marginBottom: 12,
      }}>
        <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 0.8, textTransform: 'uppercase', color: C.inkMute, marginBottom: 4 }}>
          Documentation
        </div>
        <button type="button" onClick={() => navigate('/docs')} style={{
          background: 'transparent', border: 'none', padding: 0, cursor: 'pointer',
          fontFamily: C.head, fontSize: 18, fontWeight: 600, color: C.ink, letterSpacing: -0.3,
          textAlign: 'left',
        }}>
          Doseedo Docs
        </button>
      </div>
      {SECTIONS.map((s) => (
        <div key={s.label} style={{ padding: '8px 22px 14px' }}>
          <div style={{
            fontFamily: C.mono, fontSize: 9, letterSpacing: 0.8, textTransform: 'uppercase',
            color: C.inkMute, marginBottom: 8,
          }}>
            {s.label}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {s.items.map((it) => {
              const active = it.slug === activeSlug;
              return (
                <button
                  key={it.slug}
                  type="button"
                  onClick={() => navigate(`/docs/${it.slug}`)}
                  style={{
                    background: active ? C.ink : 'transparent',
                    color: active ? C.bg : C.ink,
                    border: 'none',
                    padding: '6px 10px',
                    margin: '0 -10px',
                    cursor: 'pointer',
                    textAlign: 'left',
                    fontFamily: C.sans,
                    fontSize: 13,
                    fontWeight: active ? 500 : 400,
                    letterSpacing: -0.1,
                    lineHeight: 1.4,
                  }}
                >
                  {it.title}
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </aside>
  );
}

// ---- Subpage renderers ------------------------------------------------
// Each PAGES[slug] returns the content for that doc subpage. Add a new
// slug to SECTIONS above and register its renderer here.

const PAGES = {
  introduction: () => (
    <>
      <PageHead
        kicker="Introduction"
        title="Doseedo is a browser-based studio for music + AI."
        lede="Open a tab, sign in, drop in a track. The DAW, the instruments, the AI generators, and the chat assistant all run together — no plugin install, no project files to lose."
      />
      <H2>What's in the box</H2>
      <Bullets items={[
        'A multi-track DAW (the Studio) with MIDI editing, audio comping, automation, and a mixer.',
        'A library of in-browser instruments and effects, plus a chain editor to wire them up.',
        'AI generators: melody, accompaniment, stems, score-to-picture cues.',
        'Polypitch — phase-preserving stem isolation that re-pitches and re-times audio with no smearing.',
        'A chat assistant trained on production context, served from Modal vLLM.',
      ]} />
      <H2>What it isn't</H2>
      <P>
        Doseedo isn't a sample marketplace, a podcast editor, or a DAW replacement for full mixing-and-mastering workflows. It's a fast lane from idea to finished stem, with the AI tools right where you need them.
      </P>
      <Note kind="info" title="Where to start">
        New here? Head to <strong>Quickstart</strong> for the 3-minute path from sign-in to first stem.
      </Note>
    </>
  ),

  quickstart: () => (
    <>
      <PageHead
        kicker="Getting started"
        title="Quickstart: from sign-in to first stem in 3 minutes."
        lede="The shortest path through the studio. Sign in, open the Studio, generate a melody, render to Opus stems."
      />
      <H2>1. Sign in</H2>
      <P>
        Hit <Code>/dashboard</Code> — the top of the workspace. If you're not signed in, you'll be bounced to the marketing site to create an account. The free tier doesn't ask for a card.
      </P>
      <H2>2. Open the Studio</H2>
      <P>
        From the sidebar, click <strong>New Session</strong> (or pick a template from the dashboard slides). The Studio opens with one empty track and the transport at zero.
      </P>
      <H2>3. Generate something</H2>
      <P>
        Click the <strong>wand</strong> icon in the collapsed sidebar (or <strong>Instruments</strong> when expanded). Pick a preset, type a short prompt, hit Generate. The first run takes ~10s; subsequent runs are warm.
      </P>
      <H2>4. Render</H2>
      <P>
        Cmd/Ctrl-S saves the session. Use the export menu in the top bar to render Opus 128k stems — they land in your Downloads.
      </P>
      <Note kind="ok" title="Tip">
        Sessions auto-save every change. You can close the tab mid-take and pick up exactly where you left off.
      </Note>
    </>
  ),

  account: () => (
    <>
      <PageHead
        kicker="Getting started"
        title="Accounts, devices, and authentication."
        lede="Doseedo accounts are managed by Clerk. You can sign in with email, Google, or GitHub; sessions are issued as cookies and verified by the auth service."
      />
      <H2>Sign-in flow</H2>
      <Bullets items={[
        'Email magic link — fastest. We email a one-time link.',
        'Google / GitHub OAuth — recommended for shared machines.',
        'Sessions live as HTTP-only cookies and refresh in the background.',
      ]} />
      <H2>Device limits</H2>
      <P>
        Free and Pro have device caps (1 and 2 simultaneous). Studio and Power are unlimited. If you hit the cap, the oldest session is signed out — your work is safe, just the session token is rotated.
      </P>
      <H2>Resetting / signing out</H2>
      <P>
        Use the user chip in the bottom of the sidebar → Sign out. To clear local session data (cached projects), the Settings page has a <em>Clear local cache</em> action.
      </P>
      <Note kind="warn" title="Heads up">
        Clearing local cache wipes any sessions that haven't synced to the cloud. Export anything you can't reproduce first.
      </Note>
    </>
  ),

  studio: () => (
    <>
      <PageHead
        kicker="Workspace"
        title="The Studio: a tour of the DAW."
        lede="The Studio is a multi-track environment with MIDI, audio, automation, and a built-in mixer. Everything runs in the browser; the AI tools live one button away in the sidebar."
      />
      <H2>Layout</H2>
      <Bullets items={[
        'Top: transport (play/stop/loop, tempo, time signature, master meter).',
        'Left: navigation sidebar — collapses to a tool rail with quick toggles for Instruments, Browse, Saved, and Chat.',
        'Center: the track stack. MIDI tracks show a piano roll; audio tracks show a waveform.',
        'Right: track inspector / sidebar — gain, pan, sends, plugin chain, automation lanes.',
        'Bottom: the mixer — strips for every track and bus.',
      ]} />
      <H2>Transport</H2>
      <P>
        Spacebar toggles play/pause everywhere. Click the timeline to set the playhead; shift-click to set a loop point. Tempo and time signature are session-wide and survive across reloads.
      </P>
      <H2>Tracks</H2>
      <P>
        Drag audio onto the workspace to create an audio track. Click <Code>+</Code> to add a MIDI track. Each track has its own plugin chain, automation lanes, and routing. See <strong>Tracks &amp; Mixer</strong> for the routing model.
      </P>
      <Note kind="info" title="Mid-playback safety">
        Plugin chain mutations are now mapped via the PluginAdapter and apply mid-playback without a click — see the recent <Code>plugin-sync</Code> commits for the producer/diff details.
      </Note>
    </>
  ),

  sessions: () => (
    <>
      <PageHead
        kicker="Workspace"
        title="Sessions, projects, and autosave."
        lede="Every change is autosaved. Sessions live in the browser AND in cloud storage on paid tiers, so opening on a new device picks up where you left off."
      />
      <H2>Where sessions live</H2>
      <Bullets items={[
        'Local: your browser\'s local storage (always).',
        'Cloud: synced to Neon Postgres + R2 audio storage on paid tiers.',
        'Stems & latents: object storage with sha256 dedup, served as Opus 128k by default.',
      ]} />
      <H2>Autosave</H2>
      <P>
        The session sync hook polls for drift and writes deltas back. There's no Save button — Cmd/Ctrl-S is a hint that flushes pending writes immediately.
      </P>
      <H2>Retention</H2>
      <P>
        Free: 30-day rolling. Pro+: every take kept. Cancelled paid tiers move to read-only for 12 months — exports stay accessible.
      </P>
      <Note kind="warn" title="Local-only sessions">
        On the free tier, clearing browser data deletes local-only sessions. Export the ones that matter.
      </Note>
    </>
  ),

  navigation: () => (
    <>
      <PageHead
        kicker="Workspace"
        title="Navigation: the sidebar, search, and bookmarks."
        lede="The left sidebar is the spine of the app. It collapses to a tool rail in the Studio and expands on every other route."
      />
      <H2>Sidebar groups</H2>
      <Bullets items={[
        'Home / Search — the dashboard slides and the global search.',
        'Create — Projects, New Session, Tools, Plugins.',
        'Info — Plans, What\'s New, Downloads, and the More dropdown (Help, About, Docs, Feedback).',
        'User chip — your tier and account access at the bottom.',
      ]} />
      <H2>Tool rail (collapsed)</H2>
      <P>
        In the Studio, the sidebar collapses to a 4-button rail: Saved, Instruments, Browse, Chat. These swap the right-hand panel inside the Studio without leaving the route.
      </P>
      <H2>Bookmarks</H2>
      <P>
        Bookmarks (the panel behind the Saved icon) save creations across sessions. Useful for keeping reference renders next to the active project.
      </P>
    </>
  ),

  shortcuts: () => (
    <>
      <PageHead
        kicker="Workspace"
        title="Keyboard shortcuts."
        lede="Doseedo's shortcuts are DAW-conventional where possible. Modifier keys: Cmd on macOS, Ctrl elsewhere."
      />
      <H2>Transport</H2>
      <Bullets items={[
        'Space — play / pause',
        'Return — return to start',
        'L — toggle loop',
        'M — toggle metronome',
      ]} />
      <H2>Editing</H2>
      <Bullets items={[
        'Cmd/Ctrl + S — flush autosave',
        'Cmd/Ctrl + Z — undo',
        'Cmd/Ctrl + Shift + Z — redo',
        'Cmd/Ctrl + D — duplicate selection',
        'Backspace — delete selection',
      ]} />
      <H2>Navigation</H2>
      <Bullets items={[
        'G — toggle Generation panel',
        'B — toggle Browse',
        'C — toggle Chat',
        '/ — focus global search',
      ]} />
      <Note kind="info" title="Sketch">
        This list is a starter. The full shortcut surface is being audited — see issue tracker for the canonical version.
      </Note>
    </>
  ),

  tracks: () => (
    <>
      <PageHead
        kicker="Creation"
        title="Tracks, buses, and the mixer."
        lede="Doseedo tracks route through buses to a master. Each track has gain, pan, sends, a plugin chain, and automation lanes."
      />
      <H2>Track types</H2>
      <Bullets items={[
        'MIDI — piano roll, instrument plugin, automation.',
        'Audio — waveform, regions, fades, comping.',
        'Bus — sums tracks, hosts effect chains. Useful for parallel processing.',
        'Master — the final output bus.',
      ]} />
      <H2>Mixer</H2>
      <P>
        The mixer at the bottom of the Studio shows a strip per track + bus. Drag faders for gain, knobs for pan; double-click to type a value.
      </P>
      <H2>Plugin chains</H2>
      <P>
        Each track has its own chain. The PluginAdapter applies chain mutations mid-playback without a click — preset swaps, reorders, and parameter snapshots all stay in sync.
      </P>
      <Note kind="info" title="Chain diffing">
        Tier-2 plugin sync uses a reorder-aware chain diff so non-destructive reorders don't trigger a teardown.
      </Note>
    </>
  ),

  midi: () => (
    <>
      <PageHead
        kicker="Creation"
        title="MIDI editing."
        lede="The MIDI chart is a piano roll. Click to draw notes, drag to move, alt-drag to copy. Velocity and CC lanes sit underneath."
      />
      <H2>Drawing notes</H2>
      <Bullets items={[
        'Pencil tool: click empty space to draw a note at the active duration.',
        'Select tool: drag to marquee-select; arrow keys nudge.',
        'Velocity: drag the bottom of a note vertically or paint in the velocity lane.',
      ]} />
      <H2>Automation</H2>
      <P>
        Each MIDI parameter (CC, plugin param, gain, pan) gets its own lane. Click the lane header to add/remove lanes; click points to add automation.
      </P>
      <H2>Export</H2>
      <P>
        MIDI tracks export as <Code>.mid</Code> files via the track context menu. Sessions also export combined MIDI bundles.
      </P>
    </>
  ),

  audio: () => (
    <>
      <PageHead
        kicker="Creation"
        title="Audio waveform, regions, and stems."
        lede="Audio tracks render as waveforms. Slice, crossfade, comp, and route them through the mixer like any DAW. AI-derived stems carry extra metadata."
      />
      <H2>Regions and edits</H2>
      <Bullets items={[
        'Drag region edges to trim. Drag the body to move.',
        'Cmd/Ctrl + click to slice at the playhead.',
        'Drag two regions over each other for an automatic crossfade.',
      ]} />
      <H2>Stems</H2>
      <P>
        Stems generated by Polypitch synthesise from the master track in a worklet — they don't store an independent audio blob. The <Code>polypitchRendered</Code> metadata flag tells the player when to route to a baked audio URL instead of the worklet.
      </P>
      <Note kind="warn" title="Why this matters">
        Treating polypitch stems as a normal audio swap will fail silently — the playback path checks the flag and routes accordingly. See the audio service for the routing logic.
      </Note>
    </>
  ),

  plugins: () => (
    <>
      <PageHead
        kicker="Creation"
        title="Plugins and instruments."
        lede="The plugin browser shows every in-browser instrument and effect available to your tier. Drag onto a track to insert, double-click to open the editor."
      />
      <H2>Browse</H2>
      <P>
        Visit <Code>/plugins</Code> from the sidebar. Filter by category (synth, sampler, FX, utility). Hover for a preview tone.
      </P>
      <H2>Chains</H2>
      <P>
        A plugin chain is the ordered list of plugins on one track. The PluginAdapter syncs chain edits mid-playback — including reorders and preset swaps.
      </P>
      <H2>Presets</H2>
      <P>
        Each plugin ships with factory presets. Save your own with the <strong>Save Preset</strong> button in the editor; presets sync across sessions.
      </P>
      <Note kind="info" title="Custom plugins">
        Custom plugin creation lives behind <Code>/plugins/create</Code> — gated to early-access accounts.
      </Note>
    </>
  ),

  models: () => (
    <>
      <PageHead
        kicker="Creation"
        title="The Models catalog."
        lede="Doseedo's generators are not one-size-fits-all. The Models page lets you pick which model handles a given generation, and shows what each one is best at."
      />
      <H2>Categories</H2>
      <Bullets items={[
        'Melodic — short tonal phrases, hooks, motifs.',
        'Harmonic — pads, beds, chord progressions.',
        'Percussive — drums, percussion, grooves.',
        'Vocal — vocal phrasing and adlibs (where licensed).',
        'Stem-aware — models that condition on existing stems.',
      ]} />
      <H2>Picking a model</H2>
      <P>
        Each model card shows its strengths, latency band, and credit cost. Studio and Power tiers see early-access models that aren't yet in the curated set.
      </P>
      <Note kind="info" title="Where models run">
        Generators run on Modal — A100/H100 GPUs depending on the model's spec. Renders are deduped by sha256 so re-running the same prompt+seed costs zero credits.
      </Note>
    </>
  ),

  generation: () => (
    <>
      <PageHead
        kicker="AI features"
        title="Generation: prompts, presets, and queues."
        lede="The generation panel sits inside the Instruments sidebar. Pick a preset, write a prompt, generate. Output drops onto the active track."
      />
      <H2>Prompts</H2>
      <P>
        Short prompts beat long ones. Three to seven words that name the timbre, the mood, and the function (lead / pad / bass) is the sweet spot.
      </P>
      <H2>Presets</H2>
      <P>
        Presets bundle a model, sampling parameters, and a prompt skeleton. Every instrument exposes its own preset list; the embedded Generate form runs the active preset on the current track.
      </P>
      <H2>Queue</H2>
      <P>
        Standard-tier generations queue behind priority traffic. Studio is 2× faster, Power is 4× plus queue jump. The queue indicator in the panel header shows current depth.
      </P>
    </>
  ),

  polypitch: () => (
    <>
      <PageHead
        kicker="AI features"
        title="Polypitch: phase-preserving stem isolation."
        lede="Polypitch decomposes a track into stems by predicting per-stem masks against the master, not by re-synthesising audio. That keeps phase intact and avoids the smearing typical of generative stem split."
      />
      <H2>How it works</H2>
      <Bullets items={[
        'A learned model predicts a soft mask per stem at every frame.',
        'A WebAudio worklet multiplies the master spectrum by the mask in real time.',
        'No new audio is synthesised — the result is the master, gated.',
        'Pitch and time edits ride on top via a phase-preserving vocoder.',
      ]} />
      <H2>The metadata flag</H2>
      <Note kind="warn" title="metadata.polypitchRendered">
        Polypitch stems normally play through the worklet, even when a stem URL is present. The <Code>polypitchRendered</Code> metadata flag tells the player to route via the baked audio URL instead — set it after a final render.
      </Note>
      <H2>Limitations</H2>
      <P>
        Polypitch is great for melodic + harmonic separation. For dense percussion or clipped masters it can leak; use the model selector to fall back to a synthesised split when needed.
      </P>
    </>
  ),

  'video-scoring': () => (
    <>
      <PageHead
        kicker="AI features"
        title="Score-to-picture: video scoring."
        lede="Drop a video onto the Studio's drop-zone and Doseedo extracts cues, scene boundaries, and tempo hints. Score frame-accurately, render against picture."
      />
      <H2>Workflow</H2>
      <Bullets items={[
        'Drag a video file onto the workspace. The CPU-only video-scoring app extracts cues.',
        'Cue markers appear on the timeline — scene cuts, motion peaks, dialogue gaps.',
        'Snap notes and regions to cues for frame-accurate hits.',
        'Export bounces in sync; cue markers ride along in the project file.',
      ]} />
      <H2>Tier availability</H2>
      <P>
        Score-to-picture is included on Studio and Power. Pro and Free can preview the cues but not export against picture.
      </P>
    </>
  ),

  chat: () => (
    <>
      <PageHead
        kicker="AI features"
        title="The chat assistant."
        lede="The chat window (the speech-bubble icon in the sidebar) is a music-production-aware assistant. It can read your session, suggest edits, and wire up plugin chains on request."
      />
      <H2>Under the hood</H2>
      <Bullets items={[
        'Model: Qwen3-14B-AWQ.',
        'Served from Modal vLLM with a server-side proxy.',
        'Auth via session cookie + service-side API key (never exposed to the client).',
        'Context: the active session\'s metadata (tracks, plugins, tempo) is included.',
      ]} />
      <H2>What you can ask</H2>
      <P>
        Mix advice, plugin chain suggestions, theory questions, prompt rewrites. The assistant doesn't render audio — it suggests, you accept.
      </P>
      <Note kind="info" title="Credits">
        Chat costs ~1 credit per 5 turns. Long context windows count as one turn each.
      </Note>
    </>
  ),

  search: () => (
    <>
      <PageHead
        kicker="Sharing"
        title="Search."
        lede="Global search across MIDI, presets, public creations, and your own sessions. Hit / from anywhere to focus the search bar."
      />
      <H2>Scopes</H2>
      <Bullets items={[
        'MIDI — community + curated MIDI files, browseable as patterns.',
        'Presets — instrument and effect presets, public + private.',
        'Creations — public renders by other users.',
        'Yours — your own sessions and bookmarks.',
      ]} />
      <H2>Filters</H2>
      <P>
        Each scope exposes its own facets — key, BPM, time signature, tags. URL-driven, so you can share a filtered search.
      </P>
    </>
  ),

  profiles: () => (
    <>
      <PageHead
        kicker="Sharing"
        title="Public profiles."
        lede="Every account has a public profile at /profile/<username>. Profiles show pinned creations, public bookmarks, and a bio."
      />
      <H2>What's public</H2>
      <Bullets items={[
        'Pinned creations and their stems (if marked public).',
        'Public bookmarks.',
        'Display name, bio, and a link.',
      ]} />
      <H2>What's private by default</H2>
      <Bullets items={[
        'Sessions in progress.',
        'Bookmarks marked private.',
        'Account email, billing, devices.',
      ]} />
      <Note kind="info" title="Vanity URLs">
        Username = the URL slug. Change it from Settings; the previous slug 301-redirects for 30 days.
      </Note>
    </>
  ),

  sharing: () => (
    <>
      <PageHead
        kicker="Sharing"
        title="Exporting and sharing."
        lede="Doseedo exports stems and mixdowns as Opus 128k by default; paid tiers can render WAV at 48 / 96 kHz."
      />
      <H2>Formats</H2>
      <Bullets items={[
        'Free / Pro: Opus 128k stems, WAV mixdown at 48 kHz.',
        'Studio: WAV stems + mixdown + master at 48 / 96 kHz.',
        'Power: same as Studio + programmatic export via API.',
      ]} />
      <H2>Share links</H2>
      <P>
        Every render has a shareable URL. Share links can be public, unlisted, or password-gated. Recipients don't need an account to listen.
      </P>
      <Note kind="warn" title="Free-tier watermark">
        Free-tier exports carry an audible watermark. See <strong>Privacy &amp; Data</strong> for what the watermark contains and how attestation works.
      </Note>
    </>
  ),

  plans: () => (
    <>
      <PageHead
        kicker="Account"
        title="Plans, credits, and billing."
        lede="Three paid tiers plus a free room. Credits cover generation and chat; sessions, exports, and playback are never gated by credits."
      />
      <H2>Tiers</H2>
      <Bullets items={[
        'Free — 100 credits / mo, 200 MB, watermarked exports.',
        'Pro ($12) — 800 credits, 5 GB, commercial use, 2 devices.',
        'Studio ($29) — 3,000 credits, 25 GB, stem split + score-to-picture.',
        'Power ($79) — 10,000 credits + overage, API access, unlimited seats.',
      ]} />
      <H2>Credits</H2>
      <P>
        One credit ≈ one generation, five chat turns, or one attestation. Unused credits roll over up to 2× the monthly cap. Overage on Power bills at $0.011/credit.
      </P>
      <Note kind="info" title="Live pricing">
        Visit <Code>/plans</Code> for the current pricing matrix and per-feature comparisons.
      </Note>
    </>
  ),

  privacy: () => (
    <>
      <PageHead
        kicker="Account"
        title="Privacy and data."
        lede="A short version of the privacy story. The full policy lives at /privacy."
      />
      <H2>What we store</H2>
      <Bullets items={[
        'Account: email, display name, OAuth identity, device list (Clerk).',
        'Sessions: project metadata in Neon Postgres; audio + latents in R2.',
        'Renders: deduped by sha256 — identical inputs = same R2 object.',
        'Telemetry: route page views (Google Analytics).',
      ]} />
      <H2>Watermark + attestation</H2>
      <P>
        Free-tier exports carry an inaudible watermark. Paid exports can be optionally attested via the attestation batcher — a signed claim that this audio was generated on Doseedo at a given timestamp.
      </P>
      <H2>Deletion</H2>
      <P>
        Account deletion is self-service from Settings. Cancelled paid plans go read-only for 12 months before deletion; you can re-activate within that window with everything intact.
      </P>
    </>
  ),

  api: () => (
    <>
      <PageHead
        kicker="Reference"
        title="API access (Power tier)."
        lede="Power-tier accounts can drive generation, stem split, and export programmatically. The API mirrors the in-app workflows."
      />
      <H2>Auth</H2>
      <P>
        Issue an API key from Settings → API. Keys are scoped (read, generate, export) and revocable per-key.
      </P>
      <H2>Endpoints (sketch)</H2>
      <Bullets items={[
        'POST /v1/generate — submit a prompt + model + preset, get a job id.',
        'GET /v1/jobs/{id} — poll job status; get URLs on completion.',
        'POST /v1/stem-split — submit an audio URL, get stem URLs back.',
        'GET /v1/sessions/{id}/export — render an export bundle.',
      ]} />
      <Note kind="warn" title="Sketch">
        API surface is in flight. Treat the endpoint list as illustrative until the public reference lands.
      </Note>
    </>
  ),

  faq: () => (
    <>
      <PageHead
        kicker="Reference"
        title="FAQ."
        lede="Quick answers to the questions support gets most often."
      />
      <H2>Where are my sessions?</H2>
      <P>Local storage on free; Neon + R2 on paid. Both are auto-loaded on the next sign-in.</P>
      <H2>Can I work offline?</H2>
      <P>Partially. The Studio runs locally once loaded. Generation and chat require connectivity.</P>
      <H2>Why does the first generation take longer?</H2>
      <P>Cold starts on Modal. Subsequent runs hit a warm container and complete in seconds.</P>
      <H2>How do I report a bug?</H2>
      <P>Use the Feedback page in the More menu, or email support directly.</P>
      <H2>Is the license really commercial?</H2>
      <P>Yes, on every paid tier. See <Code>/plans</Code> for the per-tier license summary.</P>
    </>
  ),

  contact: () => (
    <>
      <PageHead
        kicker="Reference"
        title="Contact and support."
        lede="The fastest channels to get unstuck."
      />
      <H2>Support</H2>
      <P>
        Email <a href="mailto:support@doseedo.com" style={{ color: C.accent }}>support@doseedo.com</a>. Response time is tier-dependent — see <Code>/plans</Code>.
      </P>
      <H2>Feedback</H2>
      <P>
        Product feedback lives at <Code>/feedback</Code>. Feature requests, bug reports, copy fixes — all welcome.
      </P>
      <H2>Status</H2>
      <P>
        Live status board: <Code>status.doseedo.com</Code>. Subscribe for incident notifications.
      </P>
    </>
  ),
};

// ---- Overview (when no slug is set) -----------------------------------

function Overview() {
  const navigate = useNavigate();
  return (
    <>
      <PageHead
        kicker="Documentation"
        title="Doseedo Docs."
        lede="Everything you need to get from sign-in to first stem, and the reference material for going deeper. Start with Quickstart, or jump straight to a section."
      />
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: 14,
        marginTop: 8,
      }}>
        {SECTIONS.map((s) => (
          <div
            key={s.label}
            style={{
              background: C.surface,
              border: `1px solid ${C.rule}`,
              padding: '18px 20px 8px',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <div style={{
              fontFamily: C.mono, fontSize: 10, letterSpacing: 0.8, textTransform: 'uppercase',
              color: C.inkMute, marginBottom: 6,
            }}>
              {s.label}
            </div>
            <div style={{
              fontFamily: C.head, fontSize: 18, fontWeight: 600, color: C.ink,
              letterSpacing: -0.3, marginBottom: 12,
            }}>
              {s.items.length} pages
            </div>
            {s.items.map((it) => (
              <button
                key={it.slug}
                type="button"
                onClick={() => navigate(`/docs/${it.slug}`)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  borderTop: `1px dashed ${C.rule}`,
                  padding: '10px 0',
                  cursor: 'pointer',
                  textAlign: 'left',
                  display: 'grid',
                  gridTemplateColumns: '1fr auto',
                  gap: 12,
                  alignItems: 'baseline',
                  color: 'inherit',
                  fontFamily: 'inherit',
                }}
              >
                <span style={{
                  fontFamily: C.sans, fontSize: 13, fontWeight: 500, color: C.ink,
                }}>
                  {it.title}
                </span>
                <span style={{
                  fontFamily: C.mono, fontSize: 9, letterSpacing: 0.6, textTransform: 'uppercase',
                  color: C.inkMute,
                }}>
                  →
                </span>
              </button>
            ))}
          </div>
        ))}
      </div>
    </>
  );
}

// Inject the workbench Google Fonts (Inter / JetBrains Mono / Lora).
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

const Docs = () => {
  useWorkbenchFonts();
  const location = useLocation();
  const slug = useMemo(() => {
    const parts = location.pathname.split('/').filter(Boolean);
    return parts[1] || null;
  }, [location.pathname]);

  // Scroll content area to top on subpage change.
  useEffect(() => {
    const el = document.getElementById('docs-content');
    if (el) el.scrollTop = 0;
  }, [slug]);

  const meta = slug ? FLAT.find((it) => it.slug === slug) : null;
  const Renderer = slug ? PAGES[slug] : null;

  return (
    <main
      style={{
        // 220px clears the fixed LeftSidebar on dashboard routes.
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
      <Topbar section={meta?.section} title={meta?.title} />
      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        <TOC activeSlug={slug} />
        <div
          id="docs-content"
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '36px 48px 80px',
            maxWidth: 980,
            width: '100%',
            boxSizing: 'border-box',
          }}
        >
          {Renderer ? (
            <>
              <Renderer />
              <NextPrev slug={slug} />
            </>
          ) : slug ? (
            <PageHead
              kicker="404"
              title="That page hasn't been written yet."
              lede="Pick a topic from the left rail — or head back to the docs index."
            />
          ) : (
            <Overview />
          )}
        </div>
      </div>
    </main>
  );
};

export default Docs;
