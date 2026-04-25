/**
 * Module-kind renderers. Each takes a ModuleSpec from the DSL and
 * composes primitives + displays at the reference proportions. These
 * are what give the rendered plugin its "hand-crafted" quality — the
 * DSL picks the kind, the renderer owns the layout math.
 *
 * Reference: /Users/hydroadmin/Downloads/plugin editor/helix/oscillators.jsx
 * and helix/modmatrix.jsx.
 */

import React from 'react';
import { Knob, VSlider, LED, Chip, Select, Icon, MorphBar, FluxLane } from '../primitives';
import {
  WavetableStacked,
  WavetableStepped,
  GranularSample,
  NoiseWave,
  FilterCurve,
  EnvCurve,
  LFOCurve,
  VelocityCurve,
  XYPadConstellation,
  XYPadGrid,
  ModLane,
} from '../displays';

// ────────────────────────────────────────────────────────────────
// Shared module header

function ModuleHead({ label, on = true, active = false, dest, mono = false }) {
  return (
    <div className={`rr-mod__head ${active ? 'is-active' : ''}`}>
      <LED on={on} color={active ? 'amber' : 'ok'} />
      <span className="rr-mod__label">{label}</span>
      <span className="rr-spacer" />
      {mono && <span className="rr-mod__mini-label">M</span>}
      {dest && <Chip label={dest} on />}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// SubOsc — narrow column, waveform selector, PAN knob + LEVEL slider

export function SubOscModule({ module: m }) {
  const wave = m.waveform || 'sine';
  return (
    <div className="rr-mod" style={{ width: 72, flex: '0 0 72px' }}>
      <ModuleHead label={m.label || 'SUB'} mono on />
      <div className="rr-mod__body">
        <div className="rr-mod__meta-col">
          <div className="rr-mod__metarow"><span className="rr-label">OCT</span><span className="rr-readout">{m.oct ?? '-2'}</span></div>
          <div className="rr-mod__metarow"><span className="rr-label">CRS</span><span className="rr-readout">{m.crs ?? '—'}</span></div>
        </div>
        <div className="rr-mod__wave-select">
          <span className={`rr-wsi ${wave === 'sine' ? 'is-on' : ''}`}>{Icon.sine()}</span>
          <span className={`rr-wsi ${wave === 'tri'  ? 'is-on' : ''}`}>{Icon.tri()}</span>
          <span className={`rr-wsi ${wave === 'sq'   ? 'is-on' : ''}`}>{Icon.sq()}</span>
          <span className={`rr-wsi ${wave === 'saw'  ? 'is-on' : ''}`}>{Icon.saw()}</span>
        </div>
        <div className="rr-mod__spacer" />
      </div>
      <div className="rr-mod__knob-row rr-mod__knob-row--tall">
        <Knob label="PAN"   value={(m.knobs?.[0]?.value) ?? 0.5} size={34} bipolar format="pan" primary />
        <div style={{ height: 6 }} />
        <VSlider label="LEVEL" value={(m.knobs?.[1]?.value) ?? 0.65} height={38} />
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// WavetableOsc — the core Helix module. Takes variant for the display.

export function WavetableOscModule({ module: m }) {
  const variant = m.display?.variant || 'wavetable-stacked';
  const Display =
    variant === 'wavetable-stepped' ? WavetableStepped :
    variant === 'granular-sample'   ? GranularSample :
                                      WavetableStacked;
  const isGranular = variant === 'granular-sample';
  const knobs = m.knobs || [];
  const flex = m.flex ?? 1.3;

  return (
    <div className="rr-mod" style={{ flex }}>
      <ModuleHead
        label={m.label || m.id ? `OSC ${m.id || m.label}` : 'OSC'}
        dest={m.header?.dest}
        active={!!m.active}
        on
      />
      <div className="rr-mod__body">
        <div className="rr-mod__sub-head">
          {Icon.chev('d')}
          <span className="rr-mod__table-name">{m.header?.tableName || 'Table'}</span>
          <Select value={m.header?.engine || 'WAVETABLE'} />
        </div>
        {m.meta && m.meta.length > 0 && (
          <div className="rr-mod__meta-row">
            {m.meta.map((mt, i) => (
              <div key={i} className="rr-mod__meta">
                <span className="rr-label">{mt.k}</span>
                <span className="rr-value">{mt.v}</span>
              </div>
            ))}
          </div>
        )}
        <div className="rr-mod__display">
          <Display />
        </div>
        <div className="rr-mod__display-footer">
          <span className="rr-tiny-num">{m.id || '1'}</span>
          <span className="rr-readout--dim">φ</span>
          <span className="rr-readout">180°</span>
          <span className="rr-spacer" />
          {isGranular ? (
            <>
              <Select value="ONE-SHOT" />
              <span className="rr-label">UNISON</span>
              <span className="rr-readout--amber">1</span>
            </>
          ) : (
            <>
              <span className="rr-label">RAND</span>
              <span className="rr-readout">100</span>
            </>
          )}
        </div>
      </div>
      <div className="rr-mod__knob-row">
        {knobs.slice(0, 6).map((k, i) => (
          <Knob
            key={i}
            label={k.label}
            value={k.value}
            bipolar={!!k.bipolar}
            format={k.format || 'raw'}
            size={30}
            accent={k.accent || (isGranular ? 'ok' : 'primary')}
            primary={!!k.primary}
          />
        ))}
      </div>
      {knobs.length > 6 && (
        <div className="rr-mod__knob-row rr-mod__knob-row--secondary">
          {knobs.slice(6, 12).map((k, i) => (
            <Knob
              key={i}
              label={k.label}
              value={k.value}
              bipolar={!!k.bipolar}
              format={k.format || 'raw'}
              size={26}
              accent={k.accent || 'ok'}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// Noise

export function NoiseModule({ module: m }) {
  return (
    <div className="rr-mod" style={{ width: 132, flex: '0 0 132px' }}>
      <ModuleHead label={m.label || 'NOISE'} on />
      <div className="rr-mod__body">
        <div className="rr-mod__sub-head rr-center">
          <span className="rr-mod__table-name">{m.header?.tableName || 'Cream'}</span>
        </div>
        <div className="rr-mod__display">
          <NoiseWave />
        </div>
        <div className="rr-mod__display-footer">
          <span className="rr-label">STEREO</span>
          <span className="rr-readout--amber">0</span>
        </div>
      </div>
      <div className="rr-mod__knob-row rr-mod__knob-row--col">
        {(m.knobs || [{ label: 'PAN', value: 0.5 }, { label: 'LEVEL', value: 0.6 }]).slice(0, 2).map((k, i) => (
          <Knob
            key={i}
            label={k.label}
            value={k.value}
            bipolar={!!k.bipolar}
            format={k.format || 'raw'}
            size={i === 0 ? 36 : 32}
            primary={i === 0}
          />
        ))}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// Filter

export function FilterModule({ module: m }) {
  const modes = m.modes || ['PRE', 'A', 'B', 'C', 'POST', 'VIZ'];
  const active = m.activeMode ?? 1;
  const cutoffKnob = (m.knobs || []).find((k) => k.label.toLowerCase().includes('cut'));
  const resKnob    = (m.knobs || []).find((k) => k.label.toLowerCase().startsWith('res'));
  return (
    <div className="rr-mod" style={{ width: 186, flex: '0 0 186px' }}>
      <ModuleHead label={m.label || 'FILTER'} active={!!m.active} on />
      <div className="rr-mod__body">
        <div className="rr-mod__sub-head rr-center">
          {Icon.chev('l')}
          <span className="rr-mod__table-name">{m.header?.routing || 'Low-Pass 24'}</span>
          {Icon.chev('r')}
        </div>
        <div className="rr-mod__display">
          <FilterCurve cutoff={cutoffKnob?.value ?? 0.4} res={resKnob?.value ?? 0.4} hue={m.hue || 'amber'} />
        </div>
        <div className="rr-mod__filter-modes">
          {modes.map((mode, i) => (
            <span key={mode} className={`rr-fm ${i === active ? 'is-on' : ''}`}>{mode}</span>
          ))}
        </div>
      </div>
      <div className="rr-mod__knob-row">
        {(m.knobs || []).slice(0, 3).map((k, i) => (
          <Knob key={i} label={k.label} value={k.value} bipolar={!!k.bipolar} format={k.format || 'raw'} size={30} />
        ))}
      </div>
      {(m.knobs || []).length > 3 && (
        <div className="rr-mod__knob-row rr-mod__knob-row--secondary">
          {m.knobs.slice(3, 7).map((k, i) => (
            <Knob key={i} label={k.label} value={k.value} bipolar={!!k.bipolar} format={k.format || 'raw'} size={26} />
          ))}
        </div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// Envelope (ADSR)

export function EnvelopeModule({ module: m }) {
  const byLabel = (prefix) => (m.knobs || []).find((k) => k.label.toLowerCase().startsWith(prefix));
  const atk = byLabel('a')?.value ?? 0.15;
  const dec = byLabel('d')?.value ?? 0.25;
  const sus = byLabel('s')?.value ?? 0.6;
  const rel = byLabel('r')?.value ?? 0.35;

  return (
    <div className="rr-mod rr-mod--panel">
      <div className="rr-mod__panel-head">
        <span>{m.label || 'ENVELOPE'}</span>
        <Chip label={m.dest || 'VOL'} on />
      </div>
      <div className="rr-mod__panel-body">
        <div className="rr-mod__display rr-mod__display--tall">
          <EnvCurve atk={atk} dec={dec} sus={sus} rel={rel} />
        </div>
        <div className="rr-mod__knob-row rr-mod__knob-row--panel">
          {(m.knobs || []).map((k, i) => (
            <Knob key={i} label={k.label} value={k.value} format={k.format || 'ms'} size={28} />
          ))}
        </div>
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// LFO

export function LFOModule({ module: m }) {
  return (
    <div className="rr-mod rr-mod--panel">
      <div className="rr-mod__panel-head">
        <span>{m.label || 'LFO'}</span>
        <Chip label={m.dest || 'PITCH'} on />
      </div>
      <div className="rr-mod__panel-body">
        <div className="rr-mod__display rr-mod__display--tall">
          <LFOCurve points={m.display?.points} />
        </div>
        <div className="rr-mod__knob-row rr-mod__knob-row--panel">
          {(m.knobs || []).map((k, i) => (
            <Knob key={i} label={k.label} value={k.value} format={k.format || 'raw'} size={28} />
          ))}
        </div>
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// Macros — 4 or 8 generic assignable macros

export function MacrosModule({ module: m }) {
  return (
    <div className="rr-mod rr-mod--panel">
      <div className="rr-mod__panel-head">
        <span>{m.label || 'MACROS'}</span>
      </div>
      <div className="rr-mod__panel-body">
        <div className="rr-macros">
          {(m.knobs || []).slice(0, 8).map((k, i) => (
            <Knob key={i} label={k.label} value={k.value} size={32} primary={!!k.primary} />
          ))}
        </div>
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// Velocity

export function VelocityModule({ module: m }) {
  const curve = m.display?.curve ?? 1;
  return (
    <div className="rr-mod rr-mod--panel">
      <div className="rr-mod__panel-head">
        <span>{m.label || 'VELOCITY'}</span>
      </div>
      <div className="rr-mod__panel-body">
        <div className="rr-mod__display rr-mod__display--tall">
          <VelocityCurve curve={curve} />
        </div>
        <div className="rr-mod__knob-row rr-mod__knob-row--panel">
          {(m.knobs || [{ label: 'CURVE', value: 0.5 }, { label: 'DEPTH', value: 0.7 }]).map((k, i) => (
            <Knob key={i} label={k.label} value={k.value} size={28} />
          ))}
        </div>
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// CharacterModule — RC-20 / VHS-88-style tinted column. Optional
// type-select header (VINYL / TAPE), optional morph bar (WOW↔FLUTTER),
// 2-4 small knobs in a grid, optional flux-lane strip at bottom.

export function CharacterModule({ module: m }) {
  const tint = m.tint || 'var(--accent)';
  const style = {
    // Cascade tint into local CSS vars so the children pick it up.
    '--accent': tint,
    '--accent-soft': `color-mix(in oklab, ${tint} 60%, transparent)`,
    flex: 1,
  };
  return (
    <div className="rr-mod rr-mod--character" style={style}>
      <div className="rr-mod__accent-top" />
      <div className="rr-mod__head rr-mod__head--character">{m.label}</div>
      <div className="rr-mod__body">
        {m.typeSelect && (
          <div className="rr-mod__sub-head rr-center">
            <span className="rr-mod__type">◀ {m.typeSelect} ▶</span>
          </div>
        )}
        {m.morph && (
          <MorphBar a={m.morph.a} b={m.morph.b} position={m.morph.position ?? 0.5} />
        )}
        <div className="rr-mod__knob-grid-2x2">
          {(m.knobs || []).slice(0, 4).map((k, i) => (
            <div key={i} className="rr-mod__knob-cell">
              <Knob label={k.label} value={k.value} size={30} format={k.format || 'raw'} />
            </div>
          ))}
        </div>
        {m.fluxLane !== false && <FluxLane />}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// XYPadModule — 2D control pad for granular / performance UIs.

export function XYPadModule({ module: m }) {
  const Bg = m.background === 'constellation'
    ? XYPadConstellation
    : m.background === 'grid'
      ? XYPadGrid
      : XYPadGrid;
  return (
    <div className="rr-mod rr-mod--xy">
      <div className="rr-mod__head">{m.label || 'XY'}</div>
      <div className="rr-mod__xy-body">
        <Bg cursor={m.cursor || [0.55, 0.45]} />
        {m.axisLabels && (
          <div className="rr-mod__xy-axis">
            <span className="rr-mod__xy-x">{m.axisLabels[0]}</span>
            <span className="rr-mod__xy-y">{m.axisLabels[1]}</span>
          </div>
        )}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// ModLaneModule — editable step/curve mod lane.

export function ModLaneModule({ module: m }) {
  const color = {
    accent: 'var(--accent)',
    accent2: 'var(--accent2, var(--accent))',
    ok: 'var(--ok)',
    warn: 'var(--warn)',
  }[m.color || 'accent'];
  return (
    <div className="rr-mod rr-mod--mod-lane">
      <div className="rr-mod__head">
        <span style={{ color }}>●</span>
        <span className="rr-mod__label">{m.label || 'MOD'}</span>
        <span className="rr-spacer" />
        <Chip label="SYNC" />
      </div>
      <div className="rr-mod__modlane-body">
        {(m.knobs || []).length > 0 && (
          <div className="rr-mod__modlane-knobs">
            {m.knobs.map((k, i) => (
              <Knob key={i} label={k.label} value={k.value} size={26} />
            ))}
          </div>
        )}
        <div className="rr-mod__display rr-mod__display--tall">
          <ModLane points={m.points} mode={m.mode || 'curve'} color={color} />
        </div>
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// KnobBank — generic fallback for EQ / compressor / saturator / anything
// without a dedicated module-kind. Grid of labeled knobs.

export function KnobBankModule({ module: m }) {
  const cols = Math.max(1, Math.min(6, m.columns || 4));
  const knobs = m.knobs || [];
  return (
    <div className="rr-mod rr-mod--knob-bank" style={{ flex: 1 }}>
      <div className="rr-mod__head">{m.label || 'CONTROLS'}</div>
      <div
        className="rr-mod__knob-grid"
        style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
      >
        {knobs.map((k, i) => (
          <Knob
            key={i}
            label={k.label}
            value={k.value}
            bipolar={!!k.bipolar}
            format={k.format || 'raw'}
            size={k.size || 36}
            primary={!!k.primary}
            accent={k.accent || 'primary'}
          />
        ))}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// Module dispatcher

export function renderModule(module, idx) {
  switch (module.kind) {
    case 'sub':              return <SubOscModule        key={idx} module={module} />;
    case 'wavetable-osc':    return <WavetableOscModule  key={idx} module={module} />;
    case 'noise':            return <NoiseModule         key={idx} module={module} />;
    case 'filter':           return <FilterModule        key={idx} module={module} />;
    case 'envelope':         return <EnvelopeModule      key={idx} module={module} />;
    case 'lfo':              return <LFOModule           key={idx} module={module} />;
    case 'macros':           return <MacrosModule        key={idx} module={module} />;
    case 'velocity':         return <VelocityModule      key={idx} module={module} />;
    case 'character-module': return <CharacterModule     key={idx} module={module} />;
    case 'xy-pad':           return <XYPadModule         key={idx} module={module} />;
    case 'mod-lane':         return <ModLaneModule       key={idx} module={module} />;
    case 'knob-bank':        return <KnobBankModule      key={idx} module={module} />;
    default:
      // Graceful unknown fallback — renders as a generic knob-bank so
      // Qwen can propose novel combinations without breaking visually.
      if (module.knobs) return <KnobBankModule key={idx} module={{ ...module, label: module.label || module.kind }} />;
      return (
        <div key={idx} className="rr-mod rr-mod--unknown">
          <div className="rr-mod__head">unknown: {module.kind}</div>
        </div>
      );
  }
}
