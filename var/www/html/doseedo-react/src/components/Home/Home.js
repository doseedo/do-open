import React from 'react';
import { Slide } from 'react-slideshow-image';
import 'react-slideshow-image/dist/styles.css';
import * as authService from '../../services/authService';
import styles from './Home.module.css';

/**
 * Home Component
 * Landing page with feature slideshow
 */
const Home = () => {
  const user = authService.getCurrentUser();

  const slides = [
    {
      title: 'Professional DAW Interface',
      description: 'Edit and arrange your tracks with our powerful timeline and mixing tools',
      gradient: 'linear-gradient(135deg, rgba(186, 156, 255, 0.9), rgba(156, 130, 200, 0.9))',
      icon: 'fa-sliders'
    },
    {
      title: 'Multi-Track Recording',
      description: 'Layer multiple instruments and vocals with real-time audio processing',
      gradient: 'linear-gradient(135deg, rgba(118, 75, 162, 0.9), rgba(102, 126, 234, 0.9))',
      icon: 'fa-microphone-lines'
    },
    {
      title: 'Orchestra & Instrument Library',
      description: 'Access a vast library of orchestral instruments and synthesizers',
      gradient: 'linear-gradient(135deg, rgba(156, 130, 200, 0.9), rgba(186, 156, 255, 0.9))',
      icon: 'fa-music'
    },
    {
      title: 'Video to Music',
      description: 'Upload your videos and generate perfectly synced soundtracks',
      gradient: 'linear-gradient(135deg, rgba(102, 126, 234, 0.9), rgba(186, 156, 255, 0.9))',
      icon: 'fa-video'
    }
  ];

  const slideProperties = {
    duration: 5000,
    autoplay: true,
    transitionDuration: 500,
    arrows: true,
    infinite: true,
    easing: 'ease',
    indicators: true
  };

  return (
    <div className={styles.homeContainer}>
      <div className={styles.slideshow}>
        <Slide {...slideProperties}>
          {slides.map((slide, index) => (
            <div key={index} className={styles.slide}>
              <div
                className={styles.slideContent}
                style={{ background: slide.gradient }}
              >
                <div className={styles.slideIcon}>
                  <i className={`fa-solid ${slide.icon}`}></i>
                </div>
                <h1 className={styles.slideTitle}>{slide.title}</h1>
                <p className={styles.slideDescription}>{slide.description}</p>
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
            <a href="/register" className={styles.primaryBtn}>
              <i className="fa-solid fa-user-plus"></i>
              Create an Account
            </a>
          )}
          <a href="/plans.html" className={styles.secondaryBtn}>
            <i className="fa-solid fa-arrow-up-right-dots"></i>
            View Plans
          </a>
        </div>
      </div>
    </div>
  );
};

export default Home;
