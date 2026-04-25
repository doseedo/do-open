/**
 * renderDSL(dsl) → React element. Pure function: same DSL → same tree.
 *
 * - Injects palette as CSS custom properties on a root container.
 * - Wires Google Fonts from dsl.type.
 * - Walks dsl.rows and delegates to row renderers.
 */

import React, { useEffect } from 'react';
import { renderRow } from './rows';
import { validatePluginDSL } from './pluginDSL';

// Google Font loader — idempotent, attaches a <link> per unique family
const loadedFonts = new Set();
function loadGoogleFont(family) {
  if (!family || loadedFonts.has(family)) return;
  loadedFonts.add(family);
  if (typeof document === 'undefined') return;
  const href = `https://fonts.googleapis.com/css2?family=${encodeURIComponent(family).replace(/%20/g, '+')}:wght@400;500;600;700&display=swap`;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = href;
  link.dataset.rrFont = family;
  document.head.appendChild(link);
}

function buildPaletteVars(palette, type) {
  const p = palette || {};
  const t = type || {};
  const vars = {};

  (p.bg || []).forEach((c, i) => { vars[`--bg-${i}`] = c; });
  (p.ink || []).forEach((c, i) => {
    const name = ['--ink', '--ink-dim', '--ink-faint', '--ink-ghost'][i] || `--ink-${i}`;
    vars[name] = c;
  });
  if (p.line)      vars['--line'] = p.line;
  if (p.lineSoft)  vars['--line-soft'] = p.lineSoft;
  if (p.accent)    vars['--accent'] = p.accent;
  if (p.accent2)   vars['--accent2'] = p.accent2;
  if (p.led)       { vars['--led'] = p.led; vars['--led-glow'] = p.led; }
  if (p.ok)        vars['--ok'] = p.ok;
  if (p.warn)      vars['--warn'] = p.warn;

  if (t.ui)      vars['--font-ui']    = `'${t.ui}', system-ui, sans-serif`;
  if (t.mono)    vars['--font-mono']  = `'${t.mono}', ui-monospace, monospace`;
  if (t.brand)   vars['--font-brand'] = `'${t.brand}', serif`;
  if (t.display) vars['--font-display'] = `'${t.display}', sans-serif`;

  return vars;
}

function Header({ header, meta }) {
  if (!header) return null;
  return (
    <div className="rr-header">
      <div className="rr-brand">
        <div className="rr-brand__logo">{(meta?.name || 'X')[0]}</div>
        <div>
          <div className="rr-brand__name">{meta?.name}</div>
          <div className="rr-brand__sub">
            {(meta?.productType || '').toUpperCase()}{meta?.version ? ' · ' + meta.version : ''}
          </div>
        </div>
      </div>
      {header.tabs && (
        <div className="rr-tabs">
          {header.tabs.map((t, i) => (
            <div key={t} className={`rr-tab ${i === 0 ? 'is-active' : ''}`}>{t}</div>
          ))}
        </div>
      )}
      {header.preset && (
        <div className="rr-patchbar">
          <span className="rr-patchbar__name">
            PRESET — {header.preset.name}
          </span>
          <span className="rr-patchbar__meta">
            <span className="rr-k">AUTHOR</span>
            <span className="rr-v">{header.preset.author || '—'}</span>
          </span>
        </div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────

export function RenderDSL({ dsl, fit = true }) {
  useEffect(() => {
    const t = dsl?.type;
    if (t) {
      loadGoogleFont(t.brand);
      loadGoogleFont(t.ui);
      loadGoogleFont(t.mono);
      if (t.display) loadGoogleFont(t.display);
    }
  }, [dsl?.type?.brand, dsl?.type?.ui, dsl?.type?.mono, dsl?.type?.display]);

  if (!dsl) return null;

  const [canvasW, canvasH] = dsl.meta?.canvas || [1280, 800];
  const vars = buildPaletteVars(dsl.palette, dsl.type);

  return (
    <div
      className={`rr-shell rr-chassis--${dsl.meta?.chassis || 'plugin-window'}`}
      style={{
        ...vars,
        width: canvasW,
        height: canvasH,
        fontFamily: vars['--font-ui'] || 'system-ui, sans-serif',
      }}
    >
      <Header header={dsl.header} meta={dsl.meta} />
      <div className="rr-body">
        {(dsl.rows || []).map((row, i) => renderRow(row, i))}
      </div>
    </div>
  );
}

// Convenience: validate and either render or show errors.
export function RenderDSLSafe({ dsl }) {
  const { ok, errors } = validatePluginDSL(dsl);
  if (!ok) {
    return (
      <pre className="rr-error">
        DSL invalid:{'\n'}
        {errors.map((e) => `  - ${e}`).join('\n')}
      </pre>
    );
  }
  return <RenderDSL dsl={dsl} />;
}

export default RenderDSL;
