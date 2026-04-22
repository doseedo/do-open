import React, { useEffect, useRef, useState, useCallback } from 'react';
import styles from './WorkbenchSlides.module.css';

// Required so the module.css bundle is included; the `styles` object
// is otherwise unused (selectors are all :global).
if (styles) { /* side-effect import */ }

// Five workbench slides extracted from /Users/hydroadmin/Downloads/Workbench Slides.html.
// The animation JS directly below manipulates DOM by ID (`s1-body`, `s2-ticks`, ...),
// so we render all five markup strings at once and simply toggle which is visible.
const SLIDES_HTML = [
  "<section class=\"slide\" data-screen-label=\"01 Stems\">\n  <div class=\"slide__brand\">\n    <div class=\"mark\">D</div>\n    <strong>doseedo</strong>\n    <span class=\"sep\">/</span>\n    <span>Workbench \u00b7 Capabilities</span>\n  </div>\n  <div class=\"slide__marker\"><span class=\"dot\"></span>01 \u00b7 05</div>\n\n  <div class=\"slide__txt\">\n    <div class=\"slide__eyebrow\"><span class=\"bar\"></span>Source separation</div>\n    <h1 class=\"slide__h\">Turn your songs back into <em>sessions.</em></h1>\n    <p class=\"slide__copy\">State-of-the-art source separation and reverse-FX models produce instant stems with equivalent dry recordings and extracted FX chains \u2014 from nothing but a master recording.</p>\n    <dl class=\"slide__meta\">\n      <div><dt>Input</dt><dd>Stereo master \u00b7 44.1k</dd></div>\n      <div><dt>Output</dt><dd>4 stems <span class=\"accent\">+ FX</span></dd></div>\n      <div><dt>Latency</dt><dd>\u2248 8s / 3-min track</dd></div>\n      <div><dt>Engine</dt><dd>Workbench <span class=\"warm\">v0.7</span></dd></div>\n    </dl>\n  </div>\n\n  <div class=\"slide__stage\">\n    <div class=\"stage-label\"><span class=\"ix\">FIG 01</span> Full Mix \u2192 Stems <span class=\"sp\"></span><span>Ch. 01\u201304</span></div>\n    <div class=\"daw\" id=\"s1-daw\" style=\"--label-w:130px;--row-h:92px\">\n      <div class=\"daw__bar\">\n        <div class=\"daw__btn\"><svg width=\"10\" height=\"10\" viewBox=\"0 0 10 10\" fill=\"currentColor\"><polygon points=\"1,0 9,5 1,10\"></polygon></svg></div>\n        <div class=\"daw__btn\"><svg width=\"10\" height=\"10\" viewBox=\"0 0 10 10\" fill=\"currentColor\"><rect width=\"10\" height=\"10\"></rect></svg></div>\n        <div class=\"daw__lcd\">00:00:12:04</div>\n        <div class=\"daw__tag\">120 BPM</div>\n        <div class=\"daw__tag\">4/4</div>\n        <div class=\"daw__sep\"></div>\n        <div class=\"daw__tag\">Demo.wav</div>\n        <div class=\"right\">\n          <div class=\"daw__tag\">Separate</div>\n        </div>\n      </div>\n      <div class=\"daw__ruler\">\n        <div class=\"pad\">Track</div>\n        <div class=\"ticks\" id=\"s1-ticks\"></div>\n      </div>\n      <div class=\"daw__body\" id=\"s1-body\"></div>\n      <div class=\"status-badge\" id=\"s1-badge\" style=\"display:none\">\n        <div class=\"dot\"></div>\n        <span id=\"s1-badge-text\">Analyzing stems\u2026</span>\n      </div>\n    </div>\n  </div>\n\n  <div class=\"slide__foot\">\n    <div class=\"left\"><strong>Ch. 01</strong> Source separation</div>\n    <div>doseedo / Workbench</div>\n  </div>\n</section>",
  "<section class=\"slide\" data-screen-label=\"02 Generate\">\n  <div class=\"slide__brand\">\n    <div class=\"mark\">D</div>\n    <strong>doseedo</strong>\n    <span class=\"sep\">/</span>\n    <span>Workbench \u00b7 Capabilities</span>\n  </div>\n  <div class=\"slide__marker\"><span class=\"dot\"></span>02 \u00b7 05</div>\n\n  <div class=\"slide__txt\">\n    <div class=\"slide__eyebrow\"><span class=\"bar\"></span>Session-aware synthesis</div>\n    <h1 class=\"slide__h\">Generate that <em>fits the room.</em></h1>\n    <p class=\"slide__copy\">Stem generation models listen to the whole session \u2014 tempo, key, neighbors \u2014 so new parts arrive musically accurate, reliably, every time.</p>\n    <dl class=\"slide__meta\">\n      <div><dt>Conditioning</dt><dd>Session + prompt</dd></div>\n      <div><dt>Selection</dt><dd>Bars 09 \u2013 20</dd></div>\n      <div><dt>Prompt</dt><dd><span class=\"accent\">Jazz</span></dd></div>\n      <div><dt>Context</dt><dd>4 tracks</dd></div>\n    </dl>\n  </div>\n\n  <div class=\"slide__stage\">\n    <div class=\"stage-label\"><span class=\"ix\">FIG 02</span> Generate into selection <span class=\"sp\"></span><span>Ch. 09\u201320</span></div>\n    <div class=\"daw\" id=\"s2-daw\" style=\"--label-w:130px;--row-h:74px\">\n      <div class=\"daw__bar\">\n        <div class=\"daw__btn on\"><svg width=\"10\" height=\"10\" viewBox=\"0 0 10 10\" fill=\"currentColor\"><rect x=\"0\" y=\"0\" width=\"3\" height=\"10\"></rect><rect x=\"7\" y=\"0\" width=\"3\" height=\"10\"></rect></svg></div>\n        <div class=\"daw__btn\"><svg width=\"10\" height=\"10\" viewBox=\"0 0 10 10\" fill=\"currentColor\"><rect width=\"10\" height=\"10\"></rect></svg></div>\n        <div class=\"daw__lcd\" id=\"s2-lcd\">00:00:08:15</div>\n        <div class=\"daw__tag\">112 BPM</div>\n        <div class=\"daw__tag\">Am</div>\n        <div class=\"daw__sep\"></div>\n        <div class=\"daw__tag\">Session \u00b7 04 tracks</div>\n        <div class=\"right\">\n          <div class=\"daw__tag\" style=\"color:var(--wb-accent);border-color:var(--wb-accent)\">\u25cf Jazz</div>\n        </div>\n      </div>\n      <div class=\"daw__ruler\">\n        <div class=\"pad\">Bars</div>\n        <div class=\"ticks\" id=\"s2-ticks\"></div>\n      </div>\n      <div class=\"daw__body\" id=\"s2-body\"></div>\n      <div class=\"status-badge\" id=\"s2-badge\" style=\"display:none\">\n        <div class=\"dot\"></div>\n        <span>Generating \u201cJazz\u201d\u2026</span>\n      </div>\n    </div>\n  </div>\n\n  <div class=\"slide__foot\">\n    <div class=\"left\"><strong>Ch. 02</strong> Track-aware generation</div>\n    <div>doseedo / Workbench</div>\n  </div>\n</section>",
  "<section class=\"slide\" data-screen-label=\"03 Shape\">\n  <div class=\"slide__brand\">\n    <div class=\"mark\">D</div>\n    <strong>doseedo</strong>\n    <span class=\"sep\">/</span>\n    <span>Workbench \u00b7 Capabilities</span>\n  </div>\n  <div class=\"slide__marker\"><span class=\"dot\"></span>03 \u00b7 05</div>\n\n  <div class=\"slide__txt\">\n    <div class=\"slide__eyebrow\"><span class=\"bar\"></span>Timbre shaping</div>\n    <h1 class=\"slide__h\">Shape <em>any sound</em> imaginable.</h1>\n    <p class=\"slide__copy\">Timbre-shaping models let you dial in tone with the precision of a hardware channel strip \u2014 space, clarity, density, brightness \u2014 all continuous, all non-destructive.</p>\n    <dl class=\"slide__meta\">\n      <div><dt>Knobs</dt><dd>05 params</dd></div>\n      <div><dt>Range</dt><dd>Continuous</dd></div>\n      <div><dt>Preview</dt><dd><span class=\"accent\">Live</span></dd></div>\n      <div><dt>Undo</dt><dd>\u221e history</dd></div>\n    </dl>\n  </div>\n\n  <div class=\"slide__stage\">\n    <div class=\"stage-label\"><span class=\"ix\">FIG 03</span> Channel Strip \u2192 Waveform <span class=\"sp\"></span><span>Ch. A</span></div>\n    <div class=\"params\" id=\"s3-params\">\n      <!-- params panel -->\n      <div class=\"panel\" style=\"display:flex;flex-direction:column\">\n        <div class=\"panel__head\">\n          <span class=\"ico\"><svg viewBox=\"0 0 14 14\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\"><line x1=\"2\" y1=\"4\" x2=\"12\" y2=\"4\"></line><circle cx=\"5\" cy=\"4\" r=\"1.5\" fill=\"currentColor\"></circle><line x1=\"2\" y1=\"10\" x2=\"12\" y2=\"10\"></line><circle cx=\"9\" cy=\"10\" r=\"1.5\" fill=\"currentColor\"></circle></svg></span>\n          Channel Strip\n          <span class=\"badge\">A</span>\n        </div>\n        <div class=\"panel__body\" style=\"display:flex;flex-direction:column\">\n          <div class=\"param\" data-name=\"space\"><div class=\"param__row\"><span class=\"param__label\">Space</span><span class=\"param__val\">72%</span></div><div class=\"param__track\"><div class=\"param__fill\" style=\"width:72%\"></div><div class=\"param__knob\" style=\"left:72%\"></div></div></div>\n          <div class=\"param\" data-name=\"clarity\"><div class=\"param__row\"><span class=\"param__label\">Clarity</span><span class=\"param__val\">45%</span></div><div class=\"param__track\"><div class=\"param__fill\" style=\"width:45%\"></div><div class=\"param__knob\" style=\"left:45%\"></div></div></div>\n          <div class=\"param\" data-name=\"density\"><div class=\"param__row\"><span class=\"param__label\">Density</span><span class=\"param__val\">40%</span></div><div class=\"param__track\"><div class=\"param__fill\" style=\"width:40%\"></div><div class=\"param__knob\" style=\"left:40%\"></div></div></div>\n          <div class=\"param\" data-name=\"bright\"><div class=\"param__row\"><span class=\"param__label\">Brightness</span><span class=\"param__val\">50%</span></div><div class=\"param__track\"><div class=\"param__fill\" style=\"width:50%\"></div><div class=\"param__knob\" style=\"left:50%\"></div></div></div>\n          <div class=\"param\" data-name=\"pitch\"><div class=\"param__row\"><span class=\"param__label\">Pitch</span><span class=\"param__val\">0 st</span></div><div class=\"param__track\"><div class=\"param__fill\" style=\"width:50%\"></div><div class=\"param__knob\" style=\"left:50%\"></div></div></div>\n          <div class=\"param__status\" id=\"s3-status\"><span class=\"dot\"></span><span>Tightening space\u2026</span></div>\n        </div>\n      </div>\n      <!-- waveform panel -->\n      <div class=\"panel\" style=\"display:flex;flex-direction:column\">\n        <div class=\"tabs\">\n          <div class=\"tab active\">Waveform</div>\n          <div class=\"tab\">Spectrum</div>\n          <div class=\"tab\">Stereo</div>\n          <div class=\"right\"><span class=\"ico\"><svg viewBox=\"0 0 12 12\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\"><circle cx=\"6\" cy=\"6\" r=\"4\"></circle><line x1=\"6\" y1=\"2\" x2=\"6\" y2=\"6\"></line></svg></span>0:12.4</div>\n        </div>\n        <div class=\"bigwave\">\n          <div class=\"bigwave__before-after\"><span class=\"pill before\">Before</span><span class=\"pill after\">After</span></div>\n          <div class=\"bigwave__baseline\"></div>\n          <div class=\"bigwave__bars\" id=\"s3-bars\"></div>\n        </div>\n      </div>\n    </div>\n  </div>\n\n  <div class=\"slide__foot\">\n    <div class=\"left\"><strong>Ch. 03</strong> Timbre shaping</div>\n    <div>doseedo / Workbench</div>\n  </div>\n</section>",
  "<section class=\"slide\" data-screen-label=\"04 Score\">\n  <div class=\"slide__brand\">\n    <div class=\"mark\">D</div>\n    <strong>doseedo</strong>\n    <span class=\"sep\">/</span>\n    <span>Workbench \u00b7 Capabilities</span>\n  </div>\n  <div class=\"slide__marker\"><span class=\"dot\"></span>04 \u00b7 05</div>\n\n  <div class=\"slide__txt\">\n    <div class=\"slide__eyebrow\"><span class=\"bar\"></span>Score to picture</div>\n    <h1 class=\"slide__h\">Personalized, <em>adaptive</em> music.</h1>\n    <p class=\"slide__copy\">Adapt existing music to picture, or generate a fresh score tailored to your visuals. Track-level keyframing lets you polish scene by scene with precision.</p>\n    <dl class=\"slide__meta\">\n      <div><dt>Input</dt><dd>1080p \u00b7 24fps</dd></div>\n      <div><dt>Scenes</dt><dd>04 detected</dd></div>\n      <div><dt>Score</dt><dd><span class=\"accent\">4 stems</span></dd></div>\n      <div><dt>Sync</dt><dd>Frame-accurate</dd></div>\n    </dl>\n  </div>\n\n  <div class=\"slide__stage\" style=\"gap:14px\">\n    <div class=\"stage-label\"><span class=\"ix\">FIG 04</span> Scene analysis + Score <span class=\"sp\"></span><span>Reel 001</span></div>\n\n    <div class=\"scene-card\">\n      <div class=\"frame\">\n        <div class=\"tc\">TC 00:00:14:08</div>\n        <div class=\"rec\"><span class=\"d\"></span>REEL-001</div>\n        <!-- abstract \"scene\" composition -->\n        <svg width=\"100%\" height=\"100%\" viewBox=\"0 0 800 260\" preserveAspectRatio=\"xMidYMid slice\" style=\"position:absolute;inset:0\">\n          <defs>\n            <linearGradient id=\"sky\" x1=\"0\" x2=\"0\" y1=\"0\" y2=\"1\">\n              <stop offset=\"0\" stop-color=\"#c9b79a\" stop-opacity=\"0.35\"></stop>\n              <stop offset=\"1\" stop-color=\"#3a2f26\" stop-opacity=\"0\"></stop>\n            </linearGradient>\n          </defs>\n          <rect width=\"800\" height=\"140\" fill=\"url(#sky)\"></rect>\n          <polygon points=\"0,200 140,120 260,180 420,90 560,170 700,130 800,180 800,260 0,260\" fill=\"#1a2028\" opacity=\"0.85\"></polygon>\n          <polygon points=\"0,230 180,180 340,210 520,160 700,200 800,185 800,260 0,260\" fill=\"#0f1418\" opacity=\"0.9\"></polygon>\n          <circle cx=\"600\" cy=\"80\" r=\"22\" fill=\"#e8d8b8\" opacity=\"0.7\"></circle>\n        </svg>\n      </div>\n      <div class=\"scene-strip\">\n        <div class=\"scene-chip\" style=\"--c1:var(--trk-navy);--c2:var(--trk-teal)\">\n          <div class=\"scene-chip__thumb\"><svg width=\"18\" height=\"18\" viewBox=\"0 0 18 18\" fill=\"none\" stroke=\"#fff\" stroke-width=\"1.4\"><polygon points=\"6,4 14,9 6,14\" fill=\"#fff\"></polygon></svg></div>\n          <div class=\"scene-chip__meta\"><span>Sc 01</span><span class=\"bullet\"></span></div>\n          <div class=\"scene-chip__meta\" style=\"color:var(--wb-ink-mute)\"><span>0:00 \u2013 0:08</span></div>\n        </div>\n        <div class=\"scene-chip\" style=\"--c1:var(--trk-rust);--c2:var(--trk-ochre)\">\n          <div class=\"scene-chip__thumb\"><svg width=\"18\" height=\"18\" viewBox=\"0 0 18 18\" fill=\"none\" stroke=\"#fff\" stroke-width=\"1.4\"><polygon points=\"6,4 14,9 6,14\" fill=\"#fff\"></polygon></svg></div>\n          <div class=\"scene-chip__meta\"><span>Sc 02</span><span class=\"bullet\"></span></div>\n          <div class=\"scene-chip__meta\" style=\"color:var(--wb-ink-mute)\"><span>0:08 \u2013 0:16</span></div>\n        </div>\n        <div class=\"scene-chip\" style=\"--c1:var(--trk-moss);--c2:var(--trk-teal)\">\n          <div class=\"scene-chip__thumb\"><svg width=\"18\" height=\"18\" viewBox=\"0 0 18 18\" fill=\"none\" stroke=\"#fff\" stroke-width=\"1.4\"><polygon points=\"6,4 14,9 6,14\" fill=\"#fff\"></polygon></svg></div>\n          <div class=\"scene-chip__meta\"><span>Sc 03</span><span class=\"bullet\"></span></div>\n          <div class=\"scene-chip__meta\" style=\"color:var(--wb-ink-mute)\"><span>0:16 \u2013 0:22</span></div>\n        </div>\n        <div class=\"scene-chip\" style=\"--c1:var(--trk-plum);--c2:var(--trk-rust)\">\n          <div class=\"scene-chip__thumb\"><svg width=\"18\" height=\"18\" viewBox=\"0 0 18 18\" fill=\"none\" stroke=\"#fff\" stroke-width=\"1.4\"><polygon points=\"6,4 14,9 6,14\" fill=\"#fff\"></polygon></svg></div>\n          <div class=\"scene-chip__meta\"><span>Sc 04</span><span class=\"bullet\"></span></div>\n          <div class=\"scene-chip__meta\" style=\"color:var(--wb-ink-mute)\"><span>0:22 \u2013 0:30</span></div>\n        </div>\n      </div>\n    </div>\n\n    <div class=\"daw\" id=\"s4-daw\" style=\"--label-w:120px;--row-h:56px;flex:0 0 auto;height:310px\">\n      <div class=\"daw__bar\">\n        <div class=\"daw__btn on\"><svg width=\"10\" height=\"10\" viewBox=\"0 0 10 10\" fill=\"currentColor\"><polygon points=\"1,0 9,5 1,10\"></polygon></svg></div>\n        <div class=\"daw__lcd\">00:00:14:08</div>\n        <div class=\"daw__tag\">Reel 001</div>\n        <div class=\"daw__sep\"></div>\n        <div class=\"daw__tag\">Score v3</div>\n        <div class=\"right\"><div class=\"daw__tag\">Export stems</div></div>\n      </div>\n      <div class=\"daw__ruler\">\n        <div class=\"pad\">Score</div>\n        <div class=\"ticks\" id=\"s4-ticks\"></div>\n      </div>\n      <div class=\"daw__body\" id=\"s4-body\"></div>\n    </div>\n  </div>\n\n  <div class=\"slide__foot\">\n    <div class=\"left\"><strong>Ch. 04</strong> Score to picture</div>\n    <div>doseedo / Workbench</div>\n  </div>\n</section>",
  "<section class=\"slide\" data-screen-label=\"05 Editable\">\n  <div class=\"slide__brand\">\n    <div class=\"mark\">D</div>\n    <strong>doseedo</strong>\n    <span class=\"sep\">/</span>\n    <span>Workbench \u00b7 Capabilities</span>\n  </div>\n  <div class=\"slide__marker\"><span class=\"dot\"></span>05 \u00b7 05</div>\n\n  <div class=\"slide__txt\">\n    <div class=\"slide__eyebrow\"><span class=\"bar\"></span>Non-destructive editing</div>\n    <h1 class=\"slide__h\">Everything stays <em>editable.</em></h1>\n    <p class=\"slide__copy\">Nothing is ever frozen. Come back days, weeks or months later \u2014 reshape any part, re-render any stem, export fresh. The session is the source of truth.</p>\n    <dl class=\"slide__meta\">\n      <div><dt>State</dt><dd>Non-destructive</dd></div>\n      <div><dt>Versioning</dt><dd>Auto \u00b7 every save</dd></div>\n      <div><dt>Revisions</dt><dd><span class=\"accent\">Unlimited</span></dd></div>\n      <div><dt>Export</dt><dd>Stems \u00b7 Mix \u00b7 MIDI</dd></div>\n    </dl>\n  </div>\n\n  <div class=\"slide__stage\">\n    <div class=\"stage-label\"><span class=\"ix\">FIG 05</span> Re-shape any stem \u00b7 any time <span class=\"sp\"></span><span>Ch. Bass</span></div>\n    <div class=\"daw\" id=\"s5-daw\" style=\"--label-w:130px;--row-h:88px\">\n      <div class=\"daw__bar\">\n        <div class=\"daw__btn on\"><svg width=\"10\" height=\"10\" viewBox=\"0 0 10 10\" fill=\"currentColor\"><polygon points=\"1,0 9,5 1,10\"></polygon></svg></div>\n        <div class=\"daw__lcd\" id=\"s5-lcd\">00:00:05:12</div>\n        <div class=\"daw__tag\">120 BPM</div>\n        <div class=\"daw__tag\">Session \u00b7 Bass selected</div>\n        <div class=\"daw__sep\"></div>\n        <div class=\"right\"><div class=\"daw__tag\" style=\"color:var(--wb-accent-warm);border-color:var(--wb-accent-warm)\">Reshape</div><div class=\"daw__tag\">Export</div></div>\n      </div>\n      <div class=\"daw__ruler\">\n        <div class=\"pad\">Track</div>\n        <div class=\"ticks\" id=\"s5-ticks\"></div>\n      </div>\n      <div class=\"daw__body\" id=\"s5-body\"></div>\n    </div>\n  </div>\n\n  <div class=\"slide__foot\">\n    <div class=\"left\"><strong>Ch. 05</strong> Non-destructive</div>\n    <div>doseedo / Workbench</div>\n  </div>\n</section>"
];

const SLIDE_ADVANCE_MS = 9000;
const NATIVE_W = 1920;
const NATIVE_H = 1080;

// Ported verbatim from the bundled deck's inline <script>. Each slide IIFE
// sets up its own setTimeout/requestAnimationFrame cycle, so calling this
// once after the slide markup mounts is enough — animations loop forever.
function runSlideAnimations(root) {

/* ============================================================
   Shared utilities
   ============================================================ */
function seedRand(seed){
  let s = seed;
  return ()=>{ s = (s*16807) % 2147483647; return (s-1)/2147483646; };
}
function makeAmps(count, seed, envFn){
  const r = seedRand(seed); const out=[];
  for(let i=0;i<count;i++){
    const pos=i/count;
    const env = envFn ? envFn(pos) : (0.3 + 0.7*Math.sin(pos*Math.PI));
    const burst = r() > 0.88 ? 1.3 : 1;
    out.push(Math.min(1, r()*env*burst));
  }
  // smooth
  const sm=[];const R=2;
  for(let i=0;i<out.length;i++){
    let s=0,c=0;
    for(let j=Math.max(0,i-R);j<=Math.min(out.length-1,i+R);j++){s+=out[j];c++;}
    sm.push(s/c);
  }
  return sm;
}
function buildTicks(el, count, labels){
  for(let i=0;i<=count;i++){
    const t=document.createElement('div');
    t.className='t'+(i%5===0?' major':'');
    t.style.left=(i/count*100)+'%';
    el.appendChild(t);
    if(i%5===0 && i<count){
      const lbl=document.createElement('div');
      lbl.className='lbl';
      lbl.style.left=(i/count*100)+'%';
      lbl.textContent = labels ? labels(i) : i;
      el.appendChild(lbl);
    }
  }
}
function buildWaveBars(container, amps, baselinePct){
  container.innerHTML='';
  amps.forEach((a,idx)=>{
    const bar=document.createElement('i');
    bar.style.height=(Math.max(0.05,a)*100)+'%';
    bar.style.transitionDelay=(idx*4)+'ms';
    container.appendChild(bar);
  });
}
function buildClipBars(amps){
  const wrap=document.createElement('div');
  wrap.className='clip__wave';
  amps.forEach(a=>{
    const bar=document.createElement('i');
    bar.style.height=(Math.max(0.1,a)*100)+'%';
    wrap.appendChild(bar);
  });
  return wrap;
}
function mkClip(opts){
  const {left=0, width=100, trk='var(--wb-ink)', label, amps=makeAmps(60,1), dim=false, noise=false, noLabel=false} = opts;
  const c=document.createElement('div');
  c.className='clip'+(dim?' clip--dim':'')+(noise?' clip--noise':'');
  c.style.setProperty('--trk',trk);
  c.style.left=left+'%'; c.style.width=width+'%';
  const fill=document.createElement('div'); fill.className='clip__fill'; c.appendChild(fill);
  if(!noLabel && label){
    const tag=document.createElement('div'); tag.className='clip__tag'; tag.textContent=label; c.appendChild(tag);
  }
  c.appendChild(buildClipBars(amps));
  return c;
}
function mkRow(body, i, opts){
  const row=document.createElement('div');
  row.className='daw__row';
  row.style.top=(i*opts.rowH)+'px';
  row.style.setProperty('--row-h', opts.rowH+'px');
  row.style.setProperty('--trk', opts.trk);
  const lbl=document.createElement('div'); lbl.className='label';
  lbl.innerHTML = `<div class="dot"></div><div class="name">${opts.name}</div>`;
  row.appendChild(lbl);
  const lane=document.createElement('div'); lane.className='lane'; lane.dataset.lane='1';
  row.appendChild(lane);
  body.appendChild(row);
  return {row, lane};
}

/* ============================================================
   SLIDE 1 — Stem separation
   ============================================================ */
(function(){
  const ticks=root.querySelector('#' + 's1-ticks');
  buildTicks(ticks, 30, i=>i+'s');
  const body=root.querySelector('#' + 's1-body');
  const rowH=92;
  const stems=[
    {name:'Vocals',  trk:'var(--trk-rust)',  seed:501, env:p=>0.2+0.8*Math.pow(Math.sin(p*Math.PI),1.2)},
    {name:'Celeste', trk:'var(--trk-navy)',  seed:502, env:p=>0.1+0.9*Math.pow(Math.sin(p*Math.PI*3+0.5),2)*0.7},
    {name:'Strings', trk:'var(--trk-moss)',  seed:503, env:p=>0.15+0.85*Math.pow(p,0.6)*Math.sin(p*Math.PI*0.9)},
    {name:'Winds',   trk:'var(--trk-ochre)', seed:504, env:p=>0.1+0.9*(0.5+0.5*Math.cos(p*Math.PI*2-Math.PI))*0.6},
  ];
  const rows = stems.map((s,i)=>mkRow(body,i,{rowH,trk:s.trk,name:s.name}));
  // mix overlay covers all 4 rows
  const mixWrap=document.createElement('div');
  mixWrap.className='s1-fullmix';
  mixWrap.id='s1-mix';
  mixWrap.style.left=(130+8)+'px';
  mixWrap.style.right='8px';
  mixWrap.style.top='8px';
  mixWrap.style.bottom='8px';
  body.appendChild(mixWrap);
  const mixAmps=makeAmps(100, 777);
  const mixClip = mkClip({left:0,width:100,trk:'var(--wb-ink)',label:'Full Mix · demo.wav',amps:mixAmps,noLabel:false});
  mixClip.style.top='0'; mixClip.style.bottom='0'; mixClip.style.position='absolute';
  mixClip.style.left='0'; mixClip.style.width='100%';
  mixWrap.appendChild(mixClip);

  // placeholders for stem clips — built when separated
  const stemAmps = stems.map(s=>makeAmps(100, s.seed, s.env));
  function placeStemClips(){
    rows.forEach((r,i)=>{
      r.lane.innerHTML='';
      const clip = mkClip({left:0,width:100,trk:`var(--trk-${['rust','navy','moss','ochre'][i]})`,label:stems[i].name,amps:stemAmps[i]});
      r.lane.appendChild(clip);
    });
  }
  function clearStemClips(){ rows.forEach(r=>r.lane.innerHTML=''); }
  function placeNoiseClips(){
    rows.forEach((r,i)=>{
      r.lane.innerHTML='';
      const clip = mkClip({left:0,width:100,trk:`var(--trk-${['rust','navy','moss','ochre'][i]})`,label:stems[i].name,amps:makeAmps(60, Date.now()+i),noise:true});
      r.lane.appendChild(clip);
    });
  }
  clearStemClips();

  const badge=root.querySelector('#' + 's1-badge');
  const badgeText=root.querySelector('#' + 's1-badge-text');

  function cycle(){
    // phase 1: full mix only
    mixWrap.style.display='block';
    clearStemClips();
    badge.style.display='flex';
    badgeText.textContent='Analyzing stems…';
    setTimeout(()=>{
      badgeText.textContent='Separating…';
    }, 1600);
    setTimeout(()=>{
      // phase 2: mix fades, noise clips in rows
      mixWrap.style.display='none';
      placeNoiseClips();
    }, 2800);
    setTimeout(()=>{
      // phase 3: clips resolve
      badge.style.display='none';
      placeStemClips();
    }, 4200);
    setTimeout(cycle, 6500);
  }
  cycle();
})();

/* ============================================================
   SLIDE 2 — Track-aware generation
   ============================================================ */
(function(){
  const ticks=root.querySelector('#' + 's2-ticks');
  buildTicks(ticks, 30, i=>String(i+1).padStart(2,'0'));
  const body=root.querySelector('#' + 's2-body');
  const rowH=74;
  const tracks=[
    {name:'Piano',   trk:'var(--trk-teal)',  baseSeed:701, genSeed:801},
    {name:'Bass',    trk:'var(--trk-rust)',  baseSeed:702, genSeed:802},
    {name:'Drums',   trk:'var(--trk-ochre)', baseSeed:703, genSeed:803},
    {name:'Trumpet', trk:'var(--trk-moss)',  baseSeed:704, genSeed:804},
  ];
  const SEL_S=0.32, SEL_E=0.72;
  const rows=tracks.map((t,i)=>mkRow(body,i,{rowH,trk:t.trk,name:t.name}));

  const selbox=document.createElement('div');
  selbox.className='selbox';
  selbox.style.left=(SEL_S*100)+'%';
  selbox.style.width=((SEL_E-SEL_S)*100)+'%';
  selbox.style.display='none';
  selbox.innerHTML=`<div class="selbox__tag">Jazz · Bars 09–20</div>`;
  body.appendChild(selbox);
  // positioning context: selbox is in body, which has lane offset. Adjust left.
  selbox.style.left=`calc(130px + 8px + ${SEL_S*100}% * ${(body.clientWidth-138)/body.clientWidth || 1})`;
  // Simpler: use a wrapper inside lane area
  selbox.remove();
  const selWrap=document.createElement('div');
  selWrap.style.position='absolute';
  selWrap.style.left='130px';
  selWrap.style.right='0';
  selWrap.style.top='0';
  selWrap.style.bottom='0';
  selWrap.style.pointerEvents='none';
  body.appendChild(selWrap);
  const sel=document.createElement('div');
  sel.className='selbox';
  sel.style.left=(SEL_S*100)+'%';
  sel.style.width=((SEL_E-SEL_S)*100)+'%';
  sel.innerHTML=`<div class="selbox__tag">Jazz · Bars 09–20</div>`;
  selWrap.appendChild(sel);

  const playhead=document.createElement('div');
  playhead.className='playhead';
  selWrap.appendChild(playhead);

  function buildBaseClips(){
    rows.forEach((r,i)=>{
      r.lane.innerHTML='';
      const amps = makeAmps(140, tracks[i].baseSeed);
      const clip = mkClip({left:0,width:100,trk:`var(--trk-${['teal','rust','ochre','moss'][i]})`,label:tracks[i].name,amps:amps});
      r.lane.appendChild(clip);
    });
  }
  function buildSplitClips(opts){
    const {gen=false, resolved=[]} = opts;
    rows.forEach((r,i)=>{
      r.lane.innerHTML='';
      const trk = `var(--trk-${['teal','rust','ochre','moss'][i]})`;
      // before
      const beforeAmps = makeAmps(Math.floor(140*SEL_S), tracks[i].baseSeed);
      const before = mkClip({left:0, width:SEL_S*100, trk, label:tracks[i].name, amps:beforeAmps, dim:true});
      r.lane.appendChild(before);
      // middle (noise or gen)
      if(gen && !resolved.includes(i)){
        const noiseClip = mkClip({left:SEL_S*100, width:(SEL_E-SEL_S)*100, trk, label:'Gen', amps:makeAmps(60, Date.now()+i), noise:true, noLabel:true});
        r.lane.appendChild(noiseClip);
      } else if(resolved.includes(i)){
        const genAmps = makeAmps(Math.floor(140*(SEL_E-SEL_S)), tracks[i].genSeed);
        const genClip = mkClip({left:SEL_S*100, width:(SEL_E-SEL_S)*100, trk, label:'Jazz', amps:genAmps});
        r.lane.appendChild(genClip);
      }
      // after
      const afterAmps = makeAmps(Math.floor(140*(1-SEL_E)), tracks[i].baseSeed+500);
      const after = mkClip({left:SEL_E*100, width:(1-SEL_E)*100, trk, label:tracks[i].name, amps:afterAmps, dim:true, noLabel:true});
      r.lane.appendChild(after);
    });
  }

  const badge=root.querySelector('#' + 's2-badge');
  const lcd=root.querySelector('#' + 's2-lcd');

  let phStart=Date.now();
  function animPlay(){
    const t = ((Date.now()-phStart)/1000) % 30;
    const pct = (t/30)*100;
    playhead.style.left=pct+'%';
    const s = Math.floor(t);
    lcd.textContent = `00:00:${String(Math.floor(t/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
    requestAnimationFrame(animPlay);
  }
  animPlay();

  function cycle(){
    sel.style.display='none';
    badge.style.display='none';
    buildBaseClips();
    setTimeout(()=>{ sel.style.display='block'; }, 800);
    setTimeout(()=>{ buildSplitClips({gen:true,resolved:[]}); badge.style.display='flex'; }, 2200);
    const reveals=[0,1,2,3];
    reveals.forEach((i,idx)=>{
      setTimeout(()=>{ buildSplitClips({gen:true, resolved:reveals.slice(0,idx+1)}); }, 3400+idx*420);
    });
    setTimeout(()=>{ badge.style.display='none'; }, 3400+reveals.length*420);
    setTimeout(cycle, 8500);
  }
  cycle();
})();

/* ============================================================
   SLIDE 3 — Timbre shaping
   ============================================================ */
(function(){
  const bars=root.querySelector('#' + 's3-bars');
  const COUNT=72;
  const BASE = makeAmps(COUNT, 777);
  // phase targets (multipliers per bar index)
  const phases=[
    {label:'Tightening space…',    name:'space',   amps: makeAmps(COUNT, 877, p=>0.4+0.5*Math.sin(p*Math.PI))},
    {label:'Boosting clarity…',    name:'clarity', amps: makeAmps(COUNT, 977, p=>0.45+0.55*Math.pow(Math.sin(p*Math.PI),0.7))},
    {label:'Brightening tone…',    name:'bright',  amps: makeAmps(COUNT, 1077,p=>0.2+0.8*Math.pow(p,0.4))},
  ];
  const status=root.querySelector('#' + 's3-status').querySelector('span:last-child');

  // render initial bars (doubled-up: before ghost + after, using CSS we'll show after only and use filter)
  function render(amps){
    bars.innerHTML='';
    amps.forEach(a=>{
      const bar=document.createElement('i');
      bar.style.height=(Math.max(0.06,a)*100)+'%';
      bars.appendChild(bar);
    });
  }
  render(BASE);

  const paramSets={
    space:  {space:12, clarity:45, density:40, bright:50, pitch:0},
    clarity:{space:12, clarity:92, density:40, bright:50, pitch:0},
    bright: {space:12, clarity:92, density:40, bright:88, pitch:0},
  };
  function applyParams(set, activeKey){
    root.querySelectorAll('#s3-params .param').forEach(p=>{
      const n = p.dataset.name;
      p.classList.toggle('active', n===activeKey);
      const v = set[n];
      const isPitch = n==='pitch';
      const pct = isPitch ? 50+(v/12)*50 : v;
      p.querySelector('.param__fill').style.width=pct+'%';
      p.querySelector('.param__knob').style.left=pct+'%';
      const vv = p.querySelector('.param__val');
      vv.textContent = isPitch ? (v>=0?`+${v} st`:`${v} st`) : `${v}%`;
    });
  }

  let phase=0;
  function step(){
    const p = phases[phase];
    status.textContent = p.label;
    render(p.amps);
    applyParams(paramSets[p.name], p.name);
    phase=(phase+1)%phases.length;
    setTimeout(step, 2500);
  }
  applyParams(paramSets.space, null);
  setTimeout(step, 400);
})();

/* ============================================================
   SLIDE 4 — Score
   ============================================================ */
(function(){
  const ticks=root.querySelector('#' + 's4-ticks');
  buildTicks(ticks, 30, i=>i+'s');
  const body=root.querySelector('#' + 's4-body');
  const rowH=56;
  const tracks=[
    {name:'Strings', trk:'var(--trk-moss)',  seed:510},
    {name:'Piano',   trk:'var(--trk-navy)',  seed:620},
    {name:'Drums',   trk:'var(--trk-rust)',  seed:730},
    {name:'Bass',    trk:'var(--trk-ochre)', seed:840},
  ];
  const rows=tracks.map((t,i)=>mkRow(body,i,{rowH,trk:t.trk,name:t.name}));

  // layered clips — different starts to look like a score
  const spans=[
    [0, 100],
    [5, 80],
    [12, 88],
    [18, 70]
  ];
  rows.forEach((r,i)=>{
    const trkShort=['moss','navy','rust','ochre'][i];
    const [sL,sW]=spans[i];
    const amps=makeAmps(Math.max(30, Math.floor(sW*1.4)), tracks[i].seed);
    const clip=mkClip({left:sL, width:sW, trk:`var(--trk-${trkShort})`, label:tracks[i].name, amps});
    r.lane.appendChild(clip);
  });
  // playhead
  const playWrap=document.createElement('div');
  playWrap.style.cssText='position:absolute;left:120px;right:0;top:0;bottom:0;pointer-events:none';
  body.appendChild(playWrap);
  const ph=document.createElement('div');
  ph.className='playhead';
  ph.style.left='42%';
  playWrap.appendChild(ph);
})();

/* ============================================================
   SLIDE 5 — Editable
   ============================================================ */
(function(){
  const ticks=root.querySelector('#' + 's5-ticks');
  buildTicks(ticks, 30, i=>i+'s');
  const body=root.querySelector('#' + 's5-body');
  const rowH=88;
  const tracks=[
    {name:'Vocals', trk:'var(--trk-rust)',  seed:111},
    {name:'Drums',  trk:'var(--trk-ochre)', seed:222},
    {name:'Bass',   trk:'var(--trk-navy)',  seed:333, target:true},
    {name:'Synth',  trk:'var(--trk-plum)',  seed:444},
  ];
  const rows=tracks.map((t,i)=>mkRow(body,i,{rowH,trk:t.trk,name:t.name}));

  function fill(seedOverride){
    rows.forEach((r,i)=>{
      r.lane.innerHTML='';
      const trkShort=['rust','ochre','navy','plum'][i];
      const s = tracks[i].target && seedOverride!=null ? seedOverride : tracks[i].seed;
      const amps=makeAmps(140, s);
      const clip=mkClip({left:0,width:100, trk:`var(--trk-${trkShort})`, label:tracks[i].name, amps});
      if(tracks[i].target) clip.dataset.target='1';
      r.lane.appendChild(clip);
    });
  }
  function noiseTarget(){
    rows.forEach((r,i)=>{
      if(!tracks[i].target) return;
      r.lane.innerHTML='';
      const trkShort='navy';
      const clip = mkClip({left:0,width:100,trk:`var(--trk-${trkShort})`,label:tracks[i].name,amps:makeAmps(60, Date.now()),noise:true});
      r.lane.appendChild(clip);
    });
  }
  fill();

  // highlight target row
  const targetRow = rows[2];
  function highlight(on){
    targetRow.row.style.background = on ? 'rgba(29,76,122,0.06)' : '';
    targetRow.row.style.boxShadow = on ? 'inset 3px 0 0 var(--wb-accent)' : '';
  }

  // playhead
  const playWrap=document.createElement('div');
  playWrap.style.cssText='position:absolute;left:130px;right:0;top:0;bottom:0;pointer-events:none;z-index:30';
  body.appendChild(playWrap);
  const ph=document.createElement('div');
  ph.className='playhead';
  playWrap.appendChild(ph);
  const lcd=root.querySelector('#' + 's5-lcd');
  const phStart=Date.now();
  (function animPlay(){
    const t = ((Date.now()-phStart)/1000) % 30;
    ph.style.left=(t/30*100)+'%';
    lcd.textContent = `00:00:${String(Math.floor(t/60)).padStart(2,'0')}:${String(Math.floor(t)%60).padStart(2,'0')}`;
    requestAnimationFrame(animPlay);
  })();

  // reshape badge
  const badge=document.createElement('div');
  badge.className='status-badge';
  badge.style.display='none';
  badge.innerHTML=`<div class="dot"></div><span>Reshaping Bass…</span>`;
  body.parentElement.appendChild(badge);

  function cycle(){
    highlight(false);
    fill();
    setTimeout(()=>highlight(true), 1500);
    setTimeout(()=>{ noiseTarget(); badge.style.display='flex'; }, 3000);
    setTimeout(()=>{ fill(888); badge.style.display='none'; }, 4500);
    setTimeout(()=>{ highlight(false); }, 6500);
    setTimeout(cycle, 8500);
  }
  cycle();
})();

}

const WorkbenchSlides = () => {
  const hostRef = useRef(null);
  const stageRef = useRef(null);
  const didRunAnim = useRef(false);
  const [active, setActive] = useState(0);
  const [scale, setScale] = useState(0.4);

  // Run animation JS once, after all five slide strings are in the DOM.
  // Each slide's anim code finds its own elements by ID (s1-body, s2-ticks,
  // ...), so running them all at once is fine — they don't interfere.
  useEffect(() => {
    if (didRunAnim.current) return;
    const host = hostRef.current;
    if (!host) return;
    try {
      runSlideAnimations(host);
      didRunAnim.current = true;
    } catch (e) {
      console.warn('[WorkbenchSlides] animation init failed', e);
    }
  }, []);

  // Fit-to-container: compute a transform scale so the 1920×1080 slide
  // fills the available width. Re-run on resize via ResizeObserver.
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const compute = () => {
      const { clientWidth: w, clientHeight: h } = el.parentElement || el;
      if (!w || !h) return;
      setScale(Math.min(w / NATIVE_W, h / NATIVE_H));
    };
    compute();
    const ro = new ResizeObserver(compute);
    if (el.parentElement) ro.observe(el.parentElement);
    window.addEventListener('resize', compute);
    return () => { ro.disconnect(); window.removeEventListener('resize', compute); };
  }, []);

  // Auto-advance.
  useEffect(() => {
    const t = setTimeout(() => setActive((i) => (i + 1) % SLIDES_HTML.length), SLIDE_ADVANCE_MS);
    return () => clearTimeout(t);
  }, [active]);

  const goTo = useCallback((i) => setActive(i), []);

  return (
    <div className="wbDeck" ref={hostRef}>
      <div className="wbDeckViewport">
        <div
          className="wbDeckStage"
          ref={stageRef}
          style={{ transform: `translate(-50%, -50%) scale(${scale})` }}
        >
          {SLIDES_HTML.map((html, i) => (
            <div
              key={i}
              className="wbDeckSlideWrap"
              style={{ display: i === active ? 'block' : 'none' }}
              dangerouslySetInnerHTML={{ __html: html }}
            />
          ))}
        </div>
      </div>

      <div className="wbDeckDots">
        {SLIDES_HTML.map((_, i) => (
          <button
            key={i}
            type="button"
            className={`wbDeckDot ${i === active ? 'wbDeckDotActive' : ''}`}
            onClick={() => goTo(i)}
            aria-label={`Go to slide ${i + 1}`}
          />
        ))}
      </div>
    </div>
  );
};

export default WorkbenchSlides;
