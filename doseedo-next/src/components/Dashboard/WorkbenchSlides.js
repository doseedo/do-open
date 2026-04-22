import React, { useEffect, useState, useCallback } from 'react';
import styles from './WorkbenchSlides.module.css';

// Required so the bundler pulls in the module.css even though the selectors
// are all `:global`. styles is otherwise unused.
if (styles) { /* side-effect */ }

/* =====================================================================
 * Slide text blocks — lifted verbatim from /Users/hydroadmin/Downloads/Do/home/stage2.
 * Only .slide__eyebrow / .slide__h / .slide__copy / .slide__meta are kept;
 * the mock DAW stage, brand strip, slide marker, and foot bar are dropped
 * per product direction (text-only slideshow).
 * ===================================================================== */
const SLIDE_TEXTS = [
  "<div class=\"slide__eyebrow\"><span class=\"bar\"></span>Source separation</div>\n<h1 class=\"slide__h\">Turn your songs back into <em>sessions.</em></h1>\n<p class=\"slide__copy\">State-of-the-art source separation and reverse-FX models produce instant stems with equivalent dry recordings and extracted FX chains \u2014 from nothing but a master recording.</p>\n<dl class=\"slide__meta\">\n      <div><dt>Input</dt><dd>Stereo master \u00b7 44.1k</dd></div>\n      <div><dt>Output</dt><dd>4 stems <span class=\"accent\">+ FX</span></dd></div>\n      <div><dt>Latency</dt><dd>\u2248 8s / 3-min track</dd></div>\n      <div><dt>Engine</dt><dd>Workbench <span class=\"warm\">v0.7</span></dd></div>\n    </dl>",
  "<div class=\"slide__eyebrow\"><span class=\"bar\"></span>Session-aware synthesis</div>\n<h1 class=\"slide__h\">Generate that <em>fits the room.</em></h1>\n<p class=\"slide__copy\">Stem generation models listen to the whole session \u2014 tempo, key, neighbors \u2014 so new parts arrive musically accurate, reliably, every time.</p>\n<dl class=\"slide__meta\">\n      <div><dt>Conditioning</dt><dd>Session + prompt</dd></div>\n      <div><dt>Selection</dt><dd>Bars 09 \u2013 20</dd></div>\n      <div><dt>Prompt</dt><dd><span class=\"accent\">Jazz</span></dd></div>\n      <div><dt>Context</dt><dd>4 tracks</dd></div>\n    </dl>",
  "<div class=\"slide__eyebrow\"><span class=\"bar\"></span>Timbre shaping</div>\n<h1 class=\"slide__h\">Shape <em>any sound</em> imaginable.</h1>\n<p class=\"slide__copy\">Timbre-shaping models let you dial in tone with the precision of a hardware channel strip \u2014 space, clarity, density, brightness \u2014 all continuous, all non-destructive.</p>\n<dl class=\"slide__meta\">\n      <div><dt>Knobs</dt><dd>05 params</dd></div>\n      <div><dt>Range</dt><dd>Continuous</dd></div>\n      <div><dt>Preview</dt><dd><span class=\"accent\">Live</span></dd></div>\n      <div><dt>Undo</dt><dd>\u221e history</dd></div>\n    </dl>",
  "<div class=\"slide__eyebrow\"><span class=\"bar\"></span>Score to picture</div>\n<h1 class=\"slide__h\">Personalized, <em>adaptive</em> music.</h1>\n<p class=\"slide__copy\">Adapt existing music to picture, or generate a fresh score tailored to your visuals. Track-level keyframing lets you polish scene by scene with precision.</p>\n<dl class=\"slide__meta\">\n      <div><dt>Input</dt><dd>1080p \u00b7 24fps</dd></div>\n      <div><dt>Scenes</dt><dd>04 detected</dd></div>\n      <div><dt>Score</dt><dd><span class=\"accent\">4 stems</span></dd></div>\n      <div><dt>Sync</dt><dd>Frame-accurate</dd></div>\n    </dl>",
  "<div class=\"slide__eyebrow\"><span class=\"bar\"></span>Non-destructive editing</div>\n<h1 class=\"slide__h\">Everything stays <em>editable.</em></h1>\n<p class=\"slide__copy\">Nothing is ever frozen. Come back days, weeks or months later \u2014 reshape any part, re-render any stem, export fresh. The session is the source of truth.</p>",
];

const SLIDE_ADVANCE_MS = 7000;

const WorkbenchSlides = () => {
  const [active, setActive] = useState(0);

  useEffect(() => {
    const t = setTimeout(
      () => setActive((i) => (i + 1) % SLIDE_TEXTS.length),
      SLIDE_ADVANCE_MS
    );
    return () => clearTimeout(t);
  }, [active]);

  const goTo = useCallback((i) => setActive(i), []);

  return (
    <div className="wbDeck">
      <div className="wbDeckStage">
        {SLIDE_TEXTS.map((html, i) => (
          <div
            key={i}
            className={'wbDeckSlideTxt' + (i === active ? ' wbDeckSlideTxtActive' : '')}
            dangerouslySetInnerHTML={{ __html: html }}
          />
        ))}
      </div>

      <div className="wbDeckDots">
        {SLIDE_TEXTS.map((_, i) => (
          <button
            key={i}
            type="button"
            className={'wbDeckDot' + (i === active ? ' wbDeckDotActive' : '')}
            onClick={() => goTo(i)}
            aria-label={'Go to slide ' + (i + 1)}
          />
        ))}
      </div>
    </div>
  );
};

export default WorkbenchSlides;
