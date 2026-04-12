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

/**
 * Home Component
 * Landing page with feature slideshow
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

      {/* ── Feature grid ────────────────────────────────────────────
          Below the slideshow: a six-card preview of what the studio
          actually does. The cards intentionally echo the slide
          headlines so a guest can see the slideshow, then look down
          and see the same features broken out for them — slideshow
          is the demo, feature grid is the receipts. */}
      <section className={styles.featureSection}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>
            A studio that listens.
          </h2>
          <p className={styles.sectionLead}>
            Doseedo is a browser-based music studio designed to feel
            like a collaborator, not a generator. Built by composers
            and producers. Trained on rights-cleared recordings.
            Editable end-to-end.
          </p>
        </div>

        <div className={styles.featureGrid}>
          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>
              <i className="fa-solid fa-layer-group"></i>
            </div>
            <h3 className={styles.featureTitle}>Songs back into sessions</h3>
            <p className={styles.featureCopy}>
              State-of-the-art source separation and reverse-FX models
              recover dry stems from a single master recording.
            </p>
          </div>

          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>
              <i className="fa-solid fa-wave-square"></i>
            </div>
            <h3 className={styles.featureTitle}>Track-aware generation</h3>
            <p className={styles.featureCopy}>
              New stems that listen to what's already in the session.
              Generated parts sit in the mix instead of fighting it.
            </p>
          </div>

          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>
              <i className="fa-solid fa-sliders"></i>
            </div>
            <h3 className={styles.featureTitle}>Sculpt any sound</h3>
            <p className={styles.featureCopy}>
              Timbre-shaping models give you knob-level control over
              generated audio. Tune the result, don't reroll the prompt.
            </p>
          </div>

          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>
              <i className="fa-solid fa-film"></i>
            </div>
            <h3 className={styles.featureTitle}>Score to picture</h3>
            <p className={styles.featureCopy}>
              Adapt existing music to a scene or generate a new score
              from scratch, with track-level keyframing for the polish
              pass.
            </p>
          </div>

          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>
              <i className="fa-solid fa-pen-ruler"></i>
            </div>
            <h3 className={styles.featureTitle}>Nothing is ever frozen</h3>
            <p className={styles.featureCopy}>
              Every generation stays editable. Come back tomorrow,
              next week, or next year and reshape it.
            </p>
          </div>

          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>
              <i className="fa-solid fa-globe"></i>
            </div>
            <h3 className={styles.featureTitle}>Lives in your browser</h3>
            <p className={styles.featureCopy}>
              No installs. Sessions save locally and sync when you're
              ready. Share a link with a collaborator and they're in.
            </p>
          </div>
        </div>
      </section>

      {/* ── CTA card ────────────────────────────────────────────────
          Two paths: guests get a prominent "Create an account" pitch
          (they're previewing the dashboard right now and we want them
          to convert). Logged-in users get a quieter "you're already
          in" surface that points back to their work. */}
      <section className={styles.ctaSection}>
        <div className={styles.ctaCard}>
          {user ? (
            <>
              <h2 className={styles.ctaTitle}>
                Welcome back, {user.username || 'friend'}.
              </h2>
              <p className={styles.ctaLead}>
                Pick up where you left off, or start something new.
              </p>
              <div className={styles.ctaButtons}>
                <a href="/projects" className={styles.ctaPrimary}>
                  <i className="fa-solid fa-folder-open"></i>
                  Open Projects
                </a>
                <a href="/plugins" className={styles.ctaSecondary}>
                  <i className="fa-solid fa-puzzle-piece"></i>
                  Browse Plugins
                </a>
              </div>
            </>
          ) : (
            <>
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
            </>
          )}
        </div>
      </section>

      {/* ── Mission footer ──────────────────────────────────────────
          Closes the page with the same voice the Framer marketing
          site uses. Same words, same posture: musicians making tools
          for musicians, AI as collaborator. */}
      <section className={styles.missionSection}>
        <p className={styles.missionQuote}>
          "AI is the next chapter of our story."
        </p>
        <p className={styles.missionCopy}>
          Built by composers and producers. Trained on rights-cleared
          recordings. Designed to feel like a fellow musician in the
          room — not a replacement for one.
        </p>
      </section>
    </div>
  );
};

export default Home;
