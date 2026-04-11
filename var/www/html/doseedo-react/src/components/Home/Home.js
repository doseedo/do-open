import React from 'react';
import { Slide } from 'react-slideshow-image';
import 'react-slideshow-image/dist/styles.css';
import * as authService from '../../services/authService';
import styles from './Home.module.css';

// Animated slide components (framer-motion, ported from /home/arlo/do2/slides).
// Each component renders an 800×500 canvas with its own SlideFrame that already
// contains the headline + copy overlay, so Home.js only has to center them.
import Slide1 from './slides/slide1';
import Slide2 from './slides/slide2';
import Slide3 from './slides/slide3';
import Slide4 from './slides/slide4';
import Slide5 from './slides/slide5';

const SLIDE_W = 800;
const SLIDE_H = 500;
const SLIDES = [Slide1, Slide2, Slide3, Slide4, Slide5];

/**
 * Home Component
 * Landing page with feature slideshow
 */
const Home = () => {
  const user = authService.getCurrentUser();

  const slideProperties = {
    duration: 8000,
    autoplay: true,
    transitionDuration: 500,
    arrows: true,
    infinite: true,
    easing: 'ease',
    indicators: true,
  };

  return (
    <div className={styles.homeContainer}>
      <div className={styles.slideshow}>
        <Slide {...slideProperties}>
          {SLIDES.map((SlideComponent, index) => (
            <div key={index} className={styles.slide}>
              <div className={styles.slideFrameWrap}>
                <SlideComponent width={SLIDE_W} height={SLIDE_H} />
              </div>
            </div>
          ))}
        </Slide>
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
