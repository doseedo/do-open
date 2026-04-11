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

      <div className={styles.ctaSection}>
        <h2 className={styles.ctaTitle}>Ready to Create?</h2>
        <p className={styles.ctaDescription}>
          Start making professional music with powerful tools
        </p>
        <div className={styles.ctaButtons}>
          {user ? (
            <a href="/plugins" className={styles.primaryBtn}>
              <i className="fa-solid fa-flask"></i>
              Sign Up for Open Beta
            </a>
          ) : (
            /* /login hosts both signin AND registration — the toggle is inside
               login.html. There is no standalone /register SPA route. */
            <a href="/login" className={styles.primaryBtn}>
              <i className="fa-solid fa-user-plus"></i>
              Create an Account
            </a>
          )}
          <a href="/plans" className={styles.secondaryBtn}>
            <i className="fa-solid fa-arrow-up-right-dots"></i>
            View Plans
          </a>
        </div>
      </div>
    </div>
  );
};

export default Home;
