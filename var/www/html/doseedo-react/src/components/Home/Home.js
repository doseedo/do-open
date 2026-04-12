import React, { useState, useRef, useEffect } from 'react';
import { Slide } from 'react-slideshow-image';
import { motion, AnimatePresence } from 'framer-motion';
import 'react-slideshow-image/dist/styles.css';
import * as authService from '../../services/authService';
import * as dashboard from '../../services/dashboardService';
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

// Trending tab labels and their corresponding filter keys passed
// to dashboardService.getTrending().
const TRENDING_TABS = [
  { label: 'Trending', key: 'trending' },
  { label: 'New', key: 'new' },
  { label: 'Most Forked', key: 'most_forked' },
  { label: 'Your Genre', key: 'your_genre' },
];

/**
 * Home Component
 * Dashboard with feature slideshow + workspace sections.
 * All data below the slideshow is fetched from real APIs / localStorage
 * via dashboardService, with graceful fallbacks.
 */
const Home = () => {
  const user = authService.getCurrentUser();
  const username = user?.username; // stable string for effect deps

  // ── Dashboard section state ──
  const [recentSessions, setRecentSessions] = useState([]);
  const [activityItems, setActivityItems] = useState([]);
  const [liveUsers, setLiveUsers] = useState([]);
  const [trendingItems, setTrendingItems] = useState([]);
  const [trendingFilter, setTrendingFilter] = useState('trending');
  const [madeForYouItems, setMadeForYouItems] = useState([]);
  const [weekStats, setWeekStats] = useState([]);
  const [loading, setLoading] = useState({
    sessions: true, activity: true, trending: true, mfy: true,
  });

  // Fetch user-specific dashboard data once on mount (or when user changes).
  useEffect(() => {
    if (!username) {
      setLoading({ sessions: false, activity: false, trending: false, mfy: false });
      return;
    }

    dashboard.getRecentSessions().then(s => {
      setRecentSessions(s);
      setLoading(prev => ({ ...prev, sessions: false }));
    });

    dashboard.getActivityFeed().then(items => {
      setActivityItems(items);
      setLoading(prev => ({ ...prev, activity: false }));
    });

    dashboard.getLiveUsers().then(setLiveUsers);

    dashboard.getMadeForYou().then(items => {
      setMadeForYouItems(items);
      setLoading(prev => ({ ...prev, mfy: false }));
    });

    setWeekStats(dashboard.getWeekStats());
  }, [username]);

  // Trending: refetch when filter tab changes (also runs on mount).
  useEffect(() => {
    setLoading(prev => ({ ...prev, trending: true }));
    dashboard.getTrending(trendingFilter).then(items => {
      setTrendingItems(items);
      setLoading(prev => ({ ...prev, trending: false }));
    });
  }, [trendingFilter]);

  // ── Slideshow state ──
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

      {/* ── 2. Jump Back In ───────────────────────────────────────── */}
      {user && (
        <section className={styles.jumpBackInSection}>
          <div className={styles.sectionHeaderRow}>
            <h2 className={styles.dashSectionTitle}>Jump Back In</h2>
            <a href="/projects" className={styles.seeAllLink}>
              See all <i className="fa-solid fa-arrow-right"></i>
            </a>
          </div>

          {loading.sessions ? (
            <div className={styles.sessionScroll}>
              {[0,1,2,3].map(i => (
                <div key={i} className={`${styles.sessionCard} ${styles.skeleton}`} />
              ))}
            </div>
          ) : recentSessions.length === 0 ? (
            <div className={styles.emptyState}>
              <i className="fa-solid fa-plus-circle"></i>
              <span>No sessions yet. Open the <a href="/studio">Studio</a> to create your first one.</span>
            </div>
          ) : (
            <div className={styles.sessionScroll}>
              {recentSessions.map((s) => (
                <a key={s.id} href={`/studio?project=${encodeURIComponent(s.name)}`} className={styles.sessionCard}>
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
                  <span className={styles.resumeBtn}>
                    Resume{s.trackCount ? ` · ${s.trackCount} tracks` : ''}
                  </span>
                </a>
              ))}
            </div>
          )}
        </section>
      )}

      {/* ── 3. Activity + Presence (two-column split) ───────────── */}
      {user && (
        <section className={styles.activityPresenceSection}>
          <div className={styles.activityFeed}>
            <h3 className={styles.dashSubTitle}>Activity</h3>
            {loading.activity ? (
              <div className={styles.feedList}>
                {[0,1,2,3].map(i => (
                  <div key={i} className={`${styles.feedItem} ${styles.skeleton}`} style={{height: 48}} />
                ))}
              </div>
            ) : activityItems.length === 0 ? (
              <div className={styles.emptyState}>
                <span>No community activity yet. <a href="/community">Publish a creation</a> to get started.</span>
              </div>
            ) : (
              <div className={styles.feedList}>
                {activityItems.map((ev, i) => (
                  <div key={ev.id || i} className={styles.feedItem}>
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
            )}
          </div>
          <div className={styles.liveNow}>
            <h3 className={styles.dashSubTitle}>
              <span className={styles.liveIndicator} /> Live Now
            </h3>
            {liveUsers.length === 0 ? (
              <div className={styles.emptyStateSmall}>
                <span>No collaborators online right now.</span>
              </div>
            ) : (
              <div className={styles.liveList}>
                {liveUsers.map((p, i) => (
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
            )}
          </div>
        </section>
      )}

      {/* ── 4. Trending in Doseedo ────────────────────────────────── */}
      <section className={styles.trendingSection}>
        <div className={styles.sectionHeaderRow}>
          <h2 className={styles.dashSectionTitle}>Trending in Doseedo</h2>
        </div>
        <div className={styles.trendingTabs}>
          {TRENDING_TABS.map((tab) => (
            <button
              key={tab.key}
              className={`${styles.trendingTab} ${trendingFilter === tab.key ? styles.trendingTabActive : ''}`}
              onClick={() => setTrendingFilter(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {loading.trending ? (
          <div className={styles.trendingGrid}>
            {[0,1,2].map(i => (
              <div key={i} className={`${styles.trendingCard} ${styles.skeleton}`} style={{height: 180}} />
            ))}
          </div>
        ) : trendingItems.length === 0 ? (
          <div className={styles.emptyState}>
            <span>No published sessions yet. Be the first to <a href="/studio">create</a> and share one.</span>
          </div>
        ) : (
          <div className={styles.trendingGrid}>
            {trendingItems.map((t) => (
              <a key={t.id} href={`/session/${t.id}`} className={styles.trendingCard}>
                <div className={styles.trendingThumb}>
                  <div className={styles.trendingWaveform} />
                </div>
                <div className={styles.trendingMeta}>
                  <span className={styles.trendingName}>{t.name}</span>
                  <span className={styles.trendingCreator}>{t.creator}</span>
                  {t.tags.length > 0 && (
                    <div className={styles.trendingTags}>
                      {t.tags.map((tag) => (
                        <span key={tag} className={styles.genreTag}>{tag}</span>
                      ))}
                    </div>
                  )}
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
        )}
      </section>

      {/* ── 5. Made for You ───────────────────────────────────────── */}
      {user && madeForYouItems.length > 0 && (
        <section className={styles.madeForYouSection}>
          <div className={styles.sectionHeaderRow}>
            <h2 className={styles.dashSectionTitle}>Made for You</h2>
          </div>
          <div className={styles.mfyGrid}>
            {madeForYouItems.map((m) => (
              <a key={m.id} href={`/session/${m.id}`} className={styles.mfyCard}>
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

      {/* ── 6. Your Week (footer stats strip) ─────────────────────── */}
      {user && weekStats.length > 0 && (
        <section className={styles.yourWeekSection}>
          <h3 className={styles.yourWeekTitle}>Your Week</h3>
          <div className={styles.weekStats}>
            {weekStats.map((stat, i) => (
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
