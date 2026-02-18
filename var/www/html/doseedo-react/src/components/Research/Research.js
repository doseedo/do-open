import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import styles from './Research.module.css';

const papers = [
  {
    id: '2026-1',
    title: 'Trumpet Mute',
    date: 'January 2026',
    description: 'Exploring acoustic modeling of trumpet mute techniques using neural synthesis',
    icon: 'fa-trumpet',
    color: 'rgba(255, 183, 77, 0.2)',
  },
  {
    id: '2025-1',
    title: 'Midi Stem Generator',
    date: 'December 2025',
    description: 'A generative approach to producing individual instrument stems from MIDI input',
    icon: 'fa-wave-square',
    color: 'rgba(102, 126, 234, 0.2)',
  },
];

/**
 * Unwrap a Google Docs redirect URL, or return the URL as-is.
 */
function unwrapGoogleUrl(href) {
  const m = href.match(/google\.com\/url\?q=([^&]+)/);
  return m ? decodeURIComponent(m[1]) : href;
}

/**
 * Replace links to media files with inline players/embeds.
 * Supports: audio (.wav/.mp3/.ogg/.flac), MIDI (.mid/.midi), images (.png/.jpg/.gif/.webp)
 * For audio files under /media/demotests/, auto-inserts the matching mel spectrogram.
 * Handles both direct URLs and Google Docs redirect-wrapped URLs.
 */
function replaceMediaLinks(html) {
  return html.replace(
    /<a[^>]*href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/gi,
    (match, href, text) => {
      const url = unwrapGoogleUrl(href);
      const cleanText = text.replace(/<[^>]*>/g, '').trim();

      // ── Audio files ──
      if (/\/media\/demotests\/[^/]+\.(wav|mp3|ogg|flac)$/i.test(url)) {
        const filename = decodeURIComponent(url.split('/').pop().replace(/\.\w+$/, ''));
        const label = cleanText || filename;
        const spectrogramUrl = url.replace(/\/([^/]+)\.\w+$/, '/spectrograms/$1.png');
        return `<div class="research-media-block">
          <div class="research-media-label">${label}</div>
          <img class="research-spectrogram" src="${spectrogramUrl}" alt="Mel spectrogram: ${label}" loading="lazy" />
          <audio controls preload="metadata" src="${url}"></audio>
        </div>`;
      }

      // ── MIDI files ──
      if (/\/media\/demotests\/[^/]+\.(mid|midi)$/i.test(url)) {
        const filename = decodeURIComponent(url.split('/').pop().replace(/\.\w+$/, ''));
        const label = cleanText || filename;
        return `<div class="research-media-block research-midi-block" data-midi-src="${url}">
          <div class="research-midi-header">
            <div class="research-media-label"><span class="research-midi-icon">♪</span> ${label}</div>
            <a class="research-midi-download" href="${url}" download>Download MIDI</a>
          </div>
          <canvas class="research-midi-canvas"></canvas>
        </div>`;
      }

      // ── Image files ──
      if (/\/media\/demotests\/[^/]+\.(png|jpg|jpeg|gif|webp|svg)$/i.test(url)) {
        const label = cleanText || '';
        return `<div class="research-media-block">
          ${label ? `<div class="research-media-label">${label}</div>` : ''}
          <img class="research-media-image" src="${url}" alt="${label}" loading="lazy" />
        </div>`;
      }

      return match;
    }
  );
}

/**
 * Remap Google Doc colors for dark theme while preserving all other styling.
 * Only touches color values - fonts, sizes, weights, spacing stay intact.
 */
function remapForDarkTheme(css) {
  return css
    .replace(/color:#000000/g, 'color:#e0e0e0')
    .replace(/color:#434343/g, 'color:#b0b0b0')
    .replace(/color:#666666/g, 'color:#999999')
    .replace(/background-color:#ffffff/g, 'background-color:transparent')
    .replace(/border-(?:top|bottom|left|right)-color:#bfbfbf/g, (m) =>
      m.replace('#bfbfbf', 'rgba(255,255,255,0.15)')
    );
}

// ── Minimal MIDI parser (Standard MIDI File format) ──

function parseMidi(buf) {
  const view = new DataView(buf);
  let pos = 0;
  const read = (n) => { const v = buf.slice(pos, pos + n); pos += n; return v; };
  const u16 = () => { const v = view.getUint16(pos); pos += 2; return v; };
  const u32 = () => { const v = view.getUint32(pos); pos += 4; return v; };
  const varLen = () => {
    let val = 0;
    for (let i = 0; i < 4; i++) {
      const b = view.getUint8(pos++);
      val = (val << 7) | (b & 0x7f);
      if (!(b & 0x80)) break;
    }
    return val;
  };

  // MThd header
  read(4); // 'MThd'
  u32();   // header length
  const format = u16();
  const ntracks = u16();
  const tpb = u16();

  const notes = [];
  for (let t = 0; t < ntracks; t++) {
    read(4); // 'MTrk'
    const trkLen = u32();
    const trkEnd = pos + trkLen;
    let tick = 0;
    let runStatus = 0;
    const active = {}; // pitch -> {tick, vel, ch}

    while (pos < trkEnd) {
      tick += varLen();
      let status = view.getUint8(pos);
      if (status < 0x80) {
        status = runStatus; // running status
      } else {
        pos++;
        runStatus = status;
      }
      const type = status & 0xf0;
      const ch = status & 0x0f;

      if (type === 0x90) { // Note On
        const pitch = view.getUint8(pos++);
        const vel = view.getUint8(pos++);
        if (vel > 0) {
          active[pitch + ':' + ch] = { tick, vel, ch };
        } else { // vel 0 = note off
          const key = pitch + ':' + ch;
          if (active[key]) {
            notes.push({ pitch, start: active[key].tick, duration: tick - active[key].tick, velocity: active[key].vel, channel: ch, track: t });
            delete active[key];
          }
        }
      } else if (type === 0x80) { // Note Off
        const pitch = view.getUint8(pos++);
        pos++; // vel
        const key = pitch + ':' + ch;
        if (active[key]) {
          notes.push({ pitch, start: active[key].tick, duration: tick - active[key].tick, velocity: active[key].vel, channel: ch, track: t });
          delete active[key];
        }
      } else if (type === 0xc0 || type === 0xd0) {
        pos++; // 1 data byte
      } else if (type === 0xff) { // Meta event
        const metaType = view.getUint8(pos++);
        const len = varLen();
        pos += len;
      } else if (type === 0xf0 || type === 0xf7) { // SysEx
        const len = varLen();
        pos += len;
      } else {
        pos += 2; // default 2 data bytes
      }
    }
    pos = trkEnd;
  }
  return { notes, tpb };
}

const MIDI_TRACK_COLORS = [
  'rgba(88,166,255,0.85)', 'rgba(240,136,62,0.85)', 'rgba(63,185,80,0.85)',
  'rgba(210,168,255,0.85)', 'rgba(247,120,186,0.85)', 'rgba(165,214,255,0.85)',
];
const NOTE_NAMES_DISP = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const BLACK_KEYS_SET = new Set([1,3,6,8,10]);

function drawMidiPianoRoll(canvas, notes, tpb) {
  if (!notes.length) return;
  const dpr = window.devicePixelRatio || 1;
  // Try offsetWidth first (respects CSS width:100%), then walk up parents
  let W = canvas.offsetWidth;
  if (W < 50) {
    let parent = canvas.parentElement;
    while (parent && W < 50) {
      W = parent.offsetWidth || parent.clientWidth;
      parent = parent.parentElement;
    }
  }
  if (W < 50) W = 700; // final fallback
  W = Math.max(200, W - 36); // account for padding, floor at 200
  const H = 180;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  let minP = 127, maxP = 0, maxTick = 0;
  for (const n of notes) {
    minP = Math.min(minP, n.pitch);
    maxP = Math.max(maxP, n.pitch);
    maxTick = Math.max(maxTick, n.start + n.duration);
  }
  minP = Math.max(0, minP - 2);
  maxP = Math.min(127, maxP + 2);
  if (maxP - minP < 8) { minP = Math.max(0, minP - 4); maxP = minP + 8; }

  const keyW = 32;
  const pitchRange = maxP - minP + 1;
  const noteH = H / pitchRange;
  const ppt = (W - keyW) / maxTick;

  // Background
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, W, H);

  // Pitch rows
  for (let p = minP; p <= maxP; p++) {
    const y = H - (p - minP + 1) * noteH;
    ctx.fillStyle = BLACK_KEYS_SET.has(p % 12) ? '#0a0e14' : '#111820';
    ctx.fillRect(keyW, y, W - keyW, noteH);
    ctx.strokeStyle = '#1c2128';
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(keyW, y + noteH); ctx.lineTo(W, y + noteH); ctx.stroke();
  }

  // Beat lines
  for (let tick = 0; tick <= maxTick; tick += tpb) {
    const x = keyW + tick * ppt;
    const isMeasure = tick % (tpb * 4) === 0;
    ctx.strokeStyle = isMeasure ? '#30363d' : '#1c2128';
    ctx.lineWidth = isMeasure ? 1 : 0.5;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    if (isMeasure) {
      ctx.fillStyle = '#484f58';
      ctx.font = '8px monospace';
      ctx.fillText('' + Math.round(tick / tpb), x + 2, 9);
    }
  }

  // Piano keys
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, keyW, H);
  for (let p = minP; p <= maxP; p++) {
    const y = H - (p - minP + 1) * noteH;
    if (BLACK_KEYS_SET.has(p % 12)) {
      ctx.fillStyle = '#0a0e14';
      ctx.fillRect(0, y, keyW - 1, noteH);
    }
    if (p % 12 === 0) {
      ctx.fillStyle = '#484f58';
      ctx.font = (noteH > 8 ? '8' : '7') + 'px monospace';
      ctx.fillText('C' + (Math.floor(p / 12) - 1), 2, y + noteH - 1);
    }
  }
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(keyW, 0); ctx.lineTo(keyW, H); ctx.stroke();

  // Notes
  for (const n of notes) {
    const x = keyW + n.start * ppt;
    const y = H - (n.pitch - minP + 1) * noteH;
    const w = Math.max(2, n.duration * ppt - 0.5);
    const h = noteH - (noteH > 4 ? 1 : 0);
    const ci = (n.channel || 0) + (n.track || 0);
    ctx.fillStyle = MIDI_TRACK_COLORS[ci % MIDI_TRACK_COLORS.length];
    ctx.globalAlpha = 0.4 + 0.6 * (n.velocity / 127);
    ctx.fillRect(x, y, w, h);
    // top highlight
    if (noteH > 3) {
      ctx.fillStyle = '#ffffff';
      ctx.globalAlpha = 0.15;
      ctx.fillRect(x, y, w, 1);
    }
    ctx.globalAlpha = 1;
  }
}

/**
 * Research Component
 * Shows list of research papers and individual paper views
 * Paper content is fetched dynamically from Google Docs
 */
const Research = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [docStyle, setDocStyle] = useState('');
  const [docBody, setDocBody] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const contentRef = useRef(null);

  // Determine which paper to show (if any) from the URL
  const pathParts = location.pathname.split('/').filter(Boolean);
  const paperId = pathParts.length > 1 ? pathParts.slice(1).join('/') : null;

  useEffect(() => {
    if (!paperId) {
      setDocStyle('');
      setDocBody(null);
      setError(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    fetch(`/_gdoc/${paperId}`)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to load paper (${res.status})`);
        return res.text();
      })
      .then(html => {
        if (cancelled) return;

        // Extract the <style> block (contains font/layout class definitions)
        const styleMatch = html.match(/<style[^>]*>([\s\S]*?)<\/style>/i);
        const rawCss = styleMatch ? styleMatch[1] : '';

        // Extract <body> content
        const bodyMatch = html.match(/<body[^>]*>([\s\S]*)<\/body>/i);
        const rawBody = bodyMatch ? bodyMatch[1] : html;

        // Remap colors for dark theme, then replace audio links with players
        setDocStyle(remapForDarkTheme(rawCss));
        setDocBody(replaceMediaLinks(remapForDarkTheme(rawBody)));
        setLoading(false);
      })
      .catch(err => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => { cancelled = true; };
  }, [paperId]);

  // After docBody renders, find MIDI placeholders and draw piano rolls
  useEffect(() => {
    if (!docBody || !contentRef.current) return;
    // Small delay to ensure the DOM has fully laid out and elements have dimensions
    const timer = setTimeout(() => {
      const blocks = contentRef.current.querySelectorAll('[data-midi-src]');
      blocks.forEach(async (block) => {
        const src = block.getAttribute('data-midi-src');
        const canvas = block.querySelector('canvas.research-midi-canvas');
        if (!src || !canvas || canvas.dataset.rendered) return;
        canvas.dataset.rendered = '1';
        try {
          const resp = await fetch(src);
          if (!resp.ok) {
            console.error('MIDI fetch failed:', src, resp.status);
            return;
          }
          const buf = await resp.arrayBuffer();
          const { notes, tpb } = parseMidi(buf);
          if (notes.length > 0) {
            requestAnimationFrame(() => {
              drawMidiPianoRoll(canvas, notes, tpb);
            });
          }
        } catch (e) {
          console.error('MIDI render error:', e);
        }
      });
    }, 100);
    return () => clearTimeout(timer);
  }, [docBody]);

  const handlePaperClick = (id) => {
    navigate(`/research/${id}`);
  };

  const handleBackToList = () => {
    navigate('/research');
  };

  // Individual paper view
  if (paperId) {
    return (
      <div className={styles.research}>
        <button className={styles.backBtn} onClick={handleBackToList}>
          <i className="fa-solid fa-arrow-left"></i>
          <span>Back to Research</span>
        </button>

        {loading && (
          <div className={styles.loadingState}>
            <i className="fa-solid fa-spinner fa-spin"></i>
            <span>Loading paper...</span>
          </div>
        )}

        {error && (
          <div className={styles.errorState}>
            <i className="fa-solid fa-exclamation-triangle"></i>
            <span>{error}</span>
          </div>
        )}

        {docBody && (
          <article className={styles.paper}>
            {/* Google Doc's own style block with dark-theme color remapping */}
            <style>{docStyle}</style>
            {/* Minimal overrides for layout integration */}
            <style>{`
              .doc-content {
                background: transparent !important;
                padding: 0 !important;
                max-width: 800px !important;
              }
              .doc-content a {
                color: rgba(186, 156, 255, 0.9) !important;
              }
              .doc-content img {
                max-width: 100% !important;
              }
              .research-media-block {
                margin: 20px 0;
                padding: 14px 18px;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 10px;
              }
              .research-media-label {
                font-size: 13px;
                font-weight: 500;
                color: rgba(186, 156, 255, 0.9);
                margin-bottom: 10px;
              }
              .research-spectrogram {
                width: 100%;
                border-radius: 6px;
                margin-bottom: 10px;
                display: block;
              }
              .research-media-block audio {
                width: 100%;
                height: 36px;
                border-radius: 6px;
              }
              .research-media-image {
                width: 100%;
                border-radius: 6px;
                display: block;
              }
              .research-midi-block {
                display: flex;
                flex-direction: column;
                gap: 0;
              }
              .research-midi-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                margin-bottom: 10px;
              }
              .research-midi-header .research-media-label {
                margin-bottom: 0;
              }
              .research-midi-icon {
                display: inline-block;
                margin-right: 4px;
                font-size: 16px;
              }
              .research-midi-canvas {
                width: 100%;
                border-radius: 6px;
                display: block;
                background: #0d1117;
              }
              .research-midi-download {
                display: inline-block;
                padding: 6px 14px;
                background: rgba(186, 156, 255, 0.12);
                border: 1px solid rgba(186, 156, 255, 0.25);
                border-radius: 6px;
                color: rgba(186, 156, 255, 0.9) !important;
                font-size: 12px;
                font-weight: 500;
                text-decoration: none !important;
                white-space: nowrap;
                transition: background 0.15s;
              }
              .research-midi-download:hover {
                background: rgba(186, 156, 255, 0.2);
              }
            `}</style>
            <div
              ref={contentRef}
              className={styles.gdocContent}
              dangerouslySetInnerHTML={{ __html: docBody }}
            />
          </article>
        )}
      </div>
    );
  }

  // Research list view
  return (
    <div className={styles.research}>
      <div className={styles.header}>
        <h1 className={styles.title}>Research</h1>
      </div>

      <div className={styles.sessionsContainer}>
        <div className={styles.sessionsList}>
          {/* Table Header */}
          <div className={styles.sessionsHeader}>
            <div className={styles.headerNumber}>#</div>
            <div className={styles.headerTitle}>Title</div>
            <div className={styles.headerDate}>Date</div>
            <div className={styles.headerStatus}>Status</div>
          </div>

          {/* Papers */}
          {papers.map((paper, index) => (
            <div
              key={paper.id}
              className={styles.sessionRow}
              onClick={() => handlePaperClick(paper.id)}
            >
              <div className={styles.sessionNumber}>{index + 1}</div>
              <div className={styles.sessionTitle}>
                <div className={styles.paperIcon} style={{ background: paper.color }}>
                  <i className={`fa-solid ${paper.icon}`}></i>
                </div>
                <div className={styles.paperInfo}>
                  <div className={styles.sessionName}>{paper.title}</div>
                  <div className={styles.paperDesc}>{paper.description}</div>
                </div>
              </div>
              <div className={styles.sessionDate}>{paper.date}</div>
              <div className={styles.sessionStatus}>
                <span className={styles.statusBadge}>Published</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Research;
