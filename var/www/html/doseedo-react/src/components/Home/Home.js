import React, { useState, useRef, useEffect } from 'react';
import { Slide } from 'react-slideshow-image';
import { motion, AnimatePresence } from 'framer-motion';
import 'react-slideshow-image/dist/styles.css';
import * as authService from '../../services/authService';
import styles from './Home.module.css';

// Animated slide components (framer-motion, ported from /home/arlo/do2/slides).
// Each component renders an 800×500 canvas. The slides USED to render their
// own headline + copy overlay inside the canvas, but that meant the text
// slid horizontally with the canvas during transitions. The overlay was
// pulled out into Home.js — see SLIDE_LABELS + the .slideTextOverlay block
// in the JSX below. SlideFrame still accepts headline/copy props for
// backwards compat but ignores them (slides/shared.tsx).
import Slide1 from './slides/slide1';
import Slide2 from './slides/slide2';
import Slide3 from './slides/slide3';
import Slide4 from './slides/slide4';
import Slide5 from './slides/slide5';

const SLIDE_W = 800;
const SLIDE_H = 500;
const SLIDES = [Slide1, Slide2, Slide3, Slide4, Slide5];

// Per-slide animation cycle in ms. The slideshow library used to use a
// single hardcoded `duration: 8000` for all slides, which (a) cut some
// slides off mid-animation (slide2 = 10s, slide4/5 = 9s) and (b) let
// other slides loop their animation a partial second time before
// advancing (slide1 = 6s would loop ~2s into cycle 2 before transition).
//
// Index N matches SLIDES[N]. Sourced directly from the CYCLE / TOTAL_CYCLE
// constants in each slide file:
//   slide1 TOTAL_CYCLE  = 6000
//   slide2 CYCLE        = 10000
//   slide3 CYCLE        = 8000
//   slide4 CYCLE        = 9000
//   slide5 CYCLE        = 9000
// If you change any of those constants in the slide file, update the
// matching entry here too.
const SLIDE_DURATIONS = [6000, 10000, 8000, 9000, 9000];

// Headline + copy for each slide, kept in lockstep with SLIDES above.
// Index N here corresponds to SLIDES[N]. Single source of truth — the
// strings used to live inside each slide component (passed to SlideFrame
// as `headline` / `copy` props for slide1/3/4/5, or hardcoded inline in
// slide2). They were moved here so the text can fade-swap in place
// without sliding when the slideshow library transitions canvases.
const SLIDE_LABELS = [
  {
    headline: 'Turn your songs back into sessions.',
    copy: 'Our state of the art source separation and reverse FX models allow instant stems with equivalent dry recordings through extracted FX chains, from only a master recording.',
  },
  {
    headline: 'Track-aware generation',
    copy: 'Our stem generation models have session awareness, providing musically accurate results reliably.',
  },
  {
    headline: 'Make literally any sound possible.',
    copy: 'Our timbre shaping models allow you to generate and shape sounds with incredible control.',
  },
  {
    headline: 'Personalized, adaptive music',
    copy: 'Adapt existing music to picture, or generate new scores tailored for your visual media. With track level key framing, you can polish to perfection with ease.',
  },
  {
    headline: 'Everything stays editable.',
    copy: 'Nothing is ever frozen. Come back and reshape it anytime.',
  },
];

// ── Mock data for dashboard sections ────────────────────────────
// These placeholders will be replaced with real API calls once the
// backend endpoints are wired up. Structure mirrors the expected
// response shape so the swap is mechanical.

const MOCK_SESSIONS = [
  { id: 'sess-1', name: 'Sunrise', daw: 'Logic', time: '2h ago', collabs: ['Maya', 'Alex'] },
  { id: 'sess-2', name: 'Project Nebula', daw: 'Ableton', time: '5h ago', collabs: ['Perro'] },
  { id: 'sess-3', name: 'Bass V2 Rework', daw: 'Pro Tools', time: 'Yesterday', collabs: [] },
  { id: 'sess-4', name: 'Score — Ep. 3', daw: 'Logic', time: '2 days ago', collabs: ['Sam', 'Jules', 'Ari'] },
  { id: 'sess-5', name: 'Ambient Sketch', daw: 'Ableton', time: '3 days ago', collabs: [] },
  { id: 'sess-6', name: 'Funk Stems', daw: 'Pro Tools', time: '4 days ago', collabs: ['Maya'] },
];

const MOCK_ACTIVITY = [
  { who: 'Maya', action: "forked your stem 'Bass V2'", time: '12m ago' },
  { who: 'Perro', action: 'left 3 comments on Session 17', time: '1h ago' },
  { who: 'Alex', action: 'pushed new takes to Project Nebula', time: '2h ago' },
  { who: 'Doseedo', action: "Your session 'Sunrise' hit 200 plays", time: '4h ago' },
  { who: 'Sam', action: "forked 'Score — Ep. 3' and added strings", time: '6h ago' },
  { who: 'Jules', action: 'started following you', time: '8h ago' },
];

const MOCK_LIVE = [
  { who: 'Maya', session: 'Bass V2 Rework', sessionId: 'sess-3', joinable: true },
  { who: 'Perro', session: 'Funk Stems', sessionId: 'sess-6', joinable: true },
  { who: 'Alex', session: 'Private session', sessionId: null, joinable: false },
];

const MOCK_TRENDING = [
  { id: 'tr-1', name: 'Lo-fi Rainfall', creator: 'waveform.koi', tags: ['Lo-fi', 'Ambient'], plays: '12.4k', forks: 342, stems: 8, daw: 'Ableton' },
  { id: 'tr-2', name: 'Brass Section Redo', creator: 'horns.daily', tags: ['Jazz', 'Orchestral'], plays: '8.1k', forks: 189, stems: 12, daw: 'Logic' },
  { id: 'tr-3', name: 'Trap Kitchen', creator: 'beatmode', tags: ['Trap', 'Hip-Hop'], plays: '23.7k', forks: 891, stems: 6, daw: 'Pro Tools' },
  { id: 'tr-4', name: 'Synth Waves 80s', creator: 'retrosound', tags: ['Synthwave', 'Electronic'], plays: '6.3k', forks: 154, stems: 10, daw: 'Ableton' },
  { id: 'tr-5', name: 'Acoustic Fireside', creator: 'stringtheory', tags: ['Acoustic', 'Folk'], plays: '4.9k', forks: 97, stems: 5, daw: 'Logic' },
  { id: 'tr-6', name: 'Drill Edit Pack', creator: 'ukbassweight', tags: ['Drill', 'UK Bass'], plays: '15.2k', forks: 623, stems: 9, daw: 'Ableton' },
];

const MOCK_MADE_FOR_YOU = [
  { id: 'mfy-1', name: 'Velvet Pads', reason: 'Similar timbre to your recent work' },
  { id: 'mfy-2', name: 'Orchestral Layers', reason: 'Matches your Score sessions' },
  { id: 'mfy-3', name: 'Granular Textures', reason: 'Popular with producers you follow' },
  { id: 'mfy-4', name: 'Keys & Rhodes', reason: 'Stems that fit Project Nebula' },
];

const MOCK_WEEK_STATS = [
  { value: '7', label: 'Sessions edited' },
  { value: '23', label: 'Stems generated' },
  { value: '4', label: 'Collaborators active' },
  { value: '3h 42m', label: 'Time in studio' },
];

/**
 * Home Component
 * Dashboard with feature slideshow + workspace sections
 */
const Home = () => {
  const user = authService.getCurrentUser();
  const [activeSlide, setActiveSlide] = useState(0);
  // One restart counter per slide. Bumped each time a slide becomes
  // active so its key changes and React remounts the SlideComponent,
  // resetting its useEffect-driven animation chain back to t=0. Without
  // this, slide N would have been running its setTimeout cycle since
  // page mount, and by the time it became visible the animation would
  // be mid-cycle (or several cycles in for slides at the end).
  const [restartCounters, setRestartCounters] = useState(() =>
    SLIDES.map(() => 0)
  );
  // Imperative handle to the slideshow library so we can call goNext()
  // on a per-slide-duration timer below (library autoplay only takes
  // one duration value for all slides).
  const slideshowRef = useRef(null);

  // Per-slide advance timer. Whenever activeSlide changes, schedule a
  // goNext() call after that slide's specific SLIDE_DURATIONS entry
  // has elapsed. The cleanup clears the pending timer if the user
  // hand-navigates with the arrows or if Home unmounts.
  useEffect(() => {
    const ms = SLIDE_DURATIONS[activeSlide] || SLIDE_DURATIONS[0];
    const timer = setTimeout(() => {
      slideshowRef.current?.goNext();
    }, ms);
    return () => clearTimeout(timer);
  }, [activeSlide]);

  const slideProperties = {
    // Library autoplay disabled — we drive advances manually via the
    // useEffect above so each slide gets its own animation-aligned
    // duration.
    autoplay: false,
    transitionDuration: 500,
    arrows: true,
    infinite: true,
    easing: 'ease',
    indicators: true,
    // Fires when the library STARTS transitioning to slide `to`. Two
    // jobs: (1) update activeSlide so the headline/copy overlay
    // fade-swaps in sync with the canvas motion, (2) bump that slide's
    // restart counter so its component remounts and the animation
    // restarts from t=0 — see restartCounters above.
    onStartChange: (_from, to) => {
      setActiveSlide(to);
      setRestartCounters((prev) => {
        const next = [...prev];
        next[to] = (next[to] || 0) + 1;
        return next;
      });
    },
  };

  const currentLabel = SLIDE_LABELS[activeSlide] || SLIDE_LABELS[0];

  return (
    <div className={styles.homeContainer}>
      <div className={styles.slideshow}>
        <Slide ref={slideshowRef} {...slideProperties}>
          {SLIDES.map((SlideComponent, index) => (
            <div key={index} className={styles.slide}>
              <div className={styles.slideFrameWrap}>
                {/* Inner key includes the per-slide restart counter
                    so React remounts this specific SlideComponent
                    each time it becomes active, restarting the
                    animation. The wrapper div above keeps a stable
                    key={index} so react-slideshow-image's children
                    list doesn't change shape. */}
                <SlideComponent
                  key={`${index}-${restartCounters[index] || 0}`}
                  width={SLIDE_W}
                  height={SLIDE_H}
                />
              </div>
            </div>
          ))}
        </Slide>

        {/* Headline + copy that stays put while the canvas slides.
            AnimatePresence with mode="wait" cross-fades the text on
            slide change. The overlay's position is locked to the bottom
            of the centered 800×500 slide canvas via .slideTextOverlay
            (see Home.module.css). */}
        <div className={styles.slideTextOverlay}>
          <AnimatePresence mode="wait">
            <motion.div
              key={activeSlide}
              className={styles.slideTextOverlayInner}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.35, ease: [0.2, 0.8, 0.2, 1] }}
            >
              <h2 className={styles.slideHeadline}>{currentLabel.headline}</h2>
              <p className={styles.slideCopy}>{currentLabel.copy}</p>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      {/* ── 2. Jump Back In ─────────────────────────────────────────
          Horizontal scroll of recent sessions. Utility anchor — the
          user's own work sits directly below the slideshow so the
          dashboard feels like a workspace, not a marketing page.
          Each card: waveform thumbnail, DAW badge, timestamp,
          collaborator avatars, one-click resume. */}
      {user && (
        <section className={styles.jumpBackInSection}>
          <div className={styles.sectionHeaderRow}>
            <h2 className={styles.dashSectionTitle}>Jump Back In</h2>
            <a href="/projects" className={styles.seeAllLink}>
              See all <i className="fa-solid fa-arrow-right"></i>
            </a>
          </div>
          <div className={styles.sessionScroll}>
            {MOCK_SESSIONS.map((s, i) => (
              <a key={i} href={`/studio/${s.id}`} className={styles.sessionCard}>
                <div className={styles.sessionThumb}>
                  <div className={styles.sessionWaveform} />
                  <div className={styles.dawBadge} data-daw={s.daw}>
                    {s.daw}
                  </div>
                </div>
                <div className={styles.sessionInfo}>
                  <span className={styles.sessionName}>{s.name}</span>
                  <span className={styles.sessionTime}>{s.time}</span>
                </div>
                {s.collabs.length > 0 && (
                  <div className={styles.sessionCollabs}>
                    {s.collabs.map((c, j) => (
                      <span key={j} className={styles.collab} title={c}>
                        {c[0]}
                      </span>
                    ))}
                  </div>
                )}
                <span className={styles.resumeBtn}>Resume</span>
              </a>
            ))}
          </div>
        </section>
      )}

      {/* ── 3. Activity + Presence (two-column split) ───────────────
          Left 2/3: Activity Feed — social network layer. GitHub-style
          event stream for sessions/stems/forks/comments.
          Right 1/3: Live Now — Figma-style presence, who's online,
          what session, Join button. */}
      {user && (
        <section className={styles.activityPresenceSection}>
          <div className={styles.activityFeed}>
            <h3 className={styles.dashSubTitle}>Activity</h3>
            <div className={styles.feedList}>
              {MOCK_ACTIVITY.map((ev, i) => (
                <div key={i} className={styles.feedItem}>
                  <div className={styles.feedAvatar}>{ev.who[0]}</div>
                  <div className={styles.feedBody}>
                    <span className={styles.feedText}>
                      <strong>{ev.who}</strong> {ev.action}
                    </span>
                    <span className={styles.feedTime}>{ev.time}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className={styles.liveNow}>
            <h3 className={styles.dashSubTitle}>
              <span className={styles.liveIndicator} /> Live Now
            </h3>
            <div className={styles.liveList}>
              {MOCK_LIVE.map((p, i) => (
                <div key={i} className={styles.liveItem}>
                  <div className={styles.liveAvatar}>{p.who[0]}</div>
                  <div className={styles.liveInfo}>
                    <span className={styles.liveName}>{p.who}</span>
                    <span className={styles.liveSession}>{p.session}</span>
                  </div>
                  {p.joinable && (
                    <a href={`/studio/${p.sessionId}`} className={styles.joinBtn}>
                      Join
                    </a>
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* ── 4. Trending in Doseedo ──────────────────────────────────
          Grid of public sessions gaining traction. Waveform preview,
          creator, genre tags, stats. Filter tabs: Trending / New /
          Most Forked / Your Genre. The SoundCloud-for-editable-
          sessions moment — forks are the social currency. */}
      <section className={styles.trendingSection}>
        <div className={styles.sectionHeaderRow}>
          <h2 className={styles.dashSectionTitle}>Trending in Doseedo</h2>
        </div>
        <div className={styles.trendingTabs}>
          {['Trending', 'New', 'Most Forked', 'Your Genre'].map((tab, i) => (
            <button
              key={tab}
              className={`${styles.trendingTab} ${i === 0 ? styles.trendingTabActive : ''}`}
            >
              {tab}
            </button>
          ))}
        </div>
        <div className={styles.trendingGrid}>
          {MOCK_TRENDING.map((t, i) => (
            <a key={i} href={`/session/${t.id}`} className={styles.trendingCard}>
              <div className={styles.trendingThumb}>
                <div className={styles.trendingWaveform} />
              </div>
              <div className={styles.trendingMeta}>
                <span className={styles.trendingName}>{t.name}</span>
                <span className={styles.trendingCreator}>{t.creator}</span>
                <div className={styles.trendingTags}>
                  {t.tags.map((tag) => (
                    <span key={tag} className={styles.genreTag}>{tag}</span>
                  ))}
                </div>
                <div className={styles.trendingStats}>
                  <span><i className="fa-solid fa-play"></i> {t.plays}</span>
                  <span><i className="fa-solid fa-code-fork"></i> {t.forks}</span>
                  <span><i className="fa-solid fa-layer-group"></i> {t.stems}</span>
                </div>
              </div>
              <div className={styles.trendingDaw} data-daw={t.daw}>{t.daw}</div>
            </a>
          ))}
        </div>
      </section>

      {/* ── 5. Made for You ─────────────────────────────────────────
          Personalized discovery: sessions similar to your timbre
          profile, creators to follow, stems that match your current
          projects. Leverages SAMI/timbre space silently. */}
      {user && (
        <section className={styles.madeForYouSection}>
          <div className={styles.sectionHeaderRow}>
            <h2 className={styles.dashSectionTitle}>Made for You</h2>
          </div>
          <div className={styles.mfyGrid}>
            {MOCK_MADE_FOR_YOU.map((m, i) => (
              <a key={i} href={`/session/${m.id}`} className={styles.mfyCard}>
                <div className={styles.mfyThumb}>
                  <div className={styles.mfyWaveform} />
                </div>
                <span className={styles.mfyName}>{m.name}</span>
                <span className={styles.mfyReason}>{m.reason}</span>
              </a>
            ))}
          </div>
        </section>
      )}

      {/* ── 6. Your Week (footer stats strip) ───────────────────────
          Lightweight stats: sessions edited, stems generated,
          collaborators active, minutes in studio. Makes the space
          feel alive and gives a reason to come back. */}
      {user && (
        <section className={styles.yourWeekSection}>
          <h3 className={styles.yourWeekTitle}>Your Week</h3>
          <div className={styles.weekStats}>
            {MOCK_WEEK_STATS.map((stat, i) => (
              <div key={i} className={styles.weekStat}>
                <span className={styles.weekStatValue}>{stat.value}</span>
                <span className={styles.weekStatLabel}>{stat.label}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── CTA card ────────────────────────────────────────────────
          Guests get a prominent "Create an account" pitch. Logged-in
          users don't need it — the dashboard IS the product now. */}
      {!user && (
        <section className={styles.ctaSection}>
          <div className={styles.ctaCard}>
            <h2 className={styles.ctaTitle}>Create your account.</h2>
            <p className={styles.ctaLead}>
              Free forever. No credit card. Save your sessions,
              sync across devices, and get the full plugin library.
            </p>
            <div className={styles.ctaButtons}>
              <a href="/login" className={styles.ctaPrimary}>
                <i className="fa-solid fa-user-plus"></i>
                Create an Account
              </a>
              <a href="/plans" className={styles.ctaSecondary}>
                <i className="fa-solid fa-arrow-up-right-dots"></i>
                View Plans
              </a>
            </div>
          </div>
        </section>
      )}
    </div>
  );
};

export default Home;
