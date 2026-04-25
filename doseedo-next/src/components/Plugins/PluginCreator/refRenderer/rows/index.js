/**
 * Row renderers — module-strip, mod-matrix, keyboard-strip, button-row,
 * slider-bank, led-display. Each consumes a RowSpec and emits a
 * horizontally-laid-out band in the plugin canvas.
 */

import React from 'react';
import { renderModule } from '../modules';
import {
  VSlider,
  ProgramButton,
  ParamButton,
  SegDisplay,
  StereoMeter,
  CharacterKnob,
  Knob,
} from '../primitives';
import { Keyboard } from '../displays';

// ────────────────────────────────────────────────────────────────

export function ModuleStripRow({ row }) {
  return (
    <div className="rr-row rr-row--modules" style={{ height: row.height || 340 }}>
      {(row.modules || []).map((mod, i) => renderModule(mod, i))}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────

export function ModMatrixRow({ row }) {
  // mod-matrix rows are panels side-by-side, each rendered via the module
  // dispatcher using the panel's `kind` as the module kind.
  return (
    <div className="rr-row rr-row--modmatrix" style={{ height: row.height || 220 }}>
      {(row.panels || []).map((panel, i) =>
        renderModule(
          { ...panel, kind: panel.kind },
          i,
        ),
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────

export function KeyboardStripRow({ row }) {
  const { spec = {} } = row;
  return (
    <div className="rr-row rr-row--keyboard" style={{ height: row.height || 70 }}>
      <Keyboard octaves={spec.octaves || 4} pressed={spec.pressed || []} />
    </div>
  );
}

// ────────────────────────────────────────────────────────────────

export function ButtonRow({ row }) {
  const buttons = row.buttons || [];
  const isProgram = row.style === 'program';
  return (
    <div className={`rr-row rr-row--buttons ${isProgram ? 'is-program' : 'is-param'}`} style={{ height: row.height || 82 }}>
      {buttons.map((b, i) =>
        isProgram ? (
          <ProgramButton
            key={i}
            n={b.n ?? i + 1}
            name={b.name || b.label}
            active={!!b.active}
          />
        ) : (
          <ParamButton
            key={i}
            label={b.label || ''}
            ledOn={!!b.ledOn}
            active={!!b.active}
          />
        ),
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────

export function SliderBankRow({ row }) {
  const sliders = row.sliders || [];
  const brackets = row.brackets || [];
  return (
    <div className="rr-row rr-row--sliders" style={{ height: row.height || 180 }}>
      <div className="rr-slider-bank">
        {sliders.map((s, i) => (
          <VSlider
            key={i}
            label={s.label}
            value={s.value ?? 0.5}
            height={s.height || 140}
            accent={s.accent || 'primary'}
          />
        ))}
      </div>
      {brackets.length > 0 && (
        <div
          className="rr-bracket-grid"
          style={{ gridTemplateColumns: `repeat(${sliders.length}, 1fr)` }}
        >
          {brackets.map((br, i) => {
            const [from, to] = br.span || [0, sliders.length - 1];
            return (
              <div
                key={i}
                className="rr-bracket"
                style={{ gridColumn: `${from + 1} / span ${to - from + 1}` }}
              >
                <div className="rr-bracket__line" />
                <div className="rr-bracket__label">{br.label}</div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────

export function LedDisplayRow({ row }) {
  const spec = row.spec || {};
  const units = spec.units || ['sec', 'ms', 'Hz', 'kHz'];
  const activeUnit = spec.activeUnit || 0;
  return (
    <div className="rr-row rr-row--led" style={{ height: row.height || 140 }}>
      <div className="rr-display">
        {spec.meter && (
          <StereoMeter
            lChannel={spec.meter.lChannel ?? 5}
            rChannel={spec.meter.rChannel ?? 4}
          />
        )}
        <div className="rr-seg-host">
          <SegDisplay value={spec.value || '2.4'} size={spec.digitSize || 50} />
        </div>
        <div className="rr-units">
          {units.map((u, i) => (
            <div key={u} className={`rr-unit ${i === activeUnit ? 'is-on' : ''}`}>{u}</div>
          ))}
        </div>
        {spec.indicator && (
          <div className="rr-open-indicator">
            <div className="rr-open-indicator__dot" />
            <div className="rr-open-indicator__lbl">{spec.indicator}</div>
          </div>
        )}
      </div>
      {spec.captions && (
        <div className="rr-display-captions">
          {spec.captions.map((c, i) => <span key={i}>{c}</span>)}
        </div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// CharacterRow — hero "amount" knobs for RC-20 / VHS-88-family processors.

export function CharacterRow({ row }) {
  const knobs = row.knobs || [];
  return (
    <div className="rr-row rr-row--character" style={{ height: row.height || 140 }}>
      {knobs.map((k, i) => (
        <CharacterKnob
          key={i}
          label={k.label}
          value={k.value ?? 0.5}
          active={k.active !== false}
          tint={k.tint || 'var(--accent)'}
          size={k.size || 72}
        />
      ))}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// EQ-strip — compact footer row: IN GAIN · EQ CUT band · OUT WIDTH/GAIN.

export function EQStripRow({ row }) {
  const io = row.io || {};
  const eq = row.eq || {};
  return (
    <div className="rr-row rr-row--eq-strip" style={{ height: row.height || 80 }}>
      {io.in && (
        <div className="rr-eq-block">
          <span className="rr-eq-tag">IN</span>
          <Knob label={io.in.label || 'GAIN'} value={io.in.value ?? 0.5} size={32} format="db" />
        </div>
      )}
      {eq.cutLeft != null && (
        <div className="rr-eq-strip__band">
          <span className="rr-eq-tag">EQ</span>
          <div className="rr-eq-strip__cut">
            <div className="rr-eq-strip__cap" style={{ left: `${eq.cutLeft * 100}%` }} />
            <div className="rr-eq-strip__cap" style={{ left: `${(eq.cutRight ?? 0.9) * 100}%` }} />
          </div>
          {eq.tone != null && <Knob label="TONE" value={eq.tone} size={30} />}
        </div>
      )}
      {io.out && (
        <div className="rr-eq-block">
          <span className="rr-eq-tag">OUT</span>
          {(io.out.labels || ['WIDTH', 'GAIN']).map((lbl, i) => (
            <Knob
              key={lbl}
              label={lbl}
              value={(io.out.values || [0.5, 0.5])[i] ?? 0.5}
              size={32}
              format={lbl === 'GAIN' ? 'db' : 'raw'}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// Row dispatcher

export function renderRow(row, idx) {
  switch (row.kind) {
    case 'module-strip':   return <ModuleStripRow   key={idx} row={row} />;
    case 'mod-matrix':     return <ModMatrixRow     key={idx} row={row} />;
    case 'keyboard-strip': return <KeyboardStripRow key={idx} row={row} />;
    case 'button-row':     return <ButtonRow        key={idx} row={row} />;
    case 'slider-bank':    return <SliderBankRow    key={idx} row={row} />;
    case 'led-display':    return <LedDisplayRow    key={idx} row={row} />;
    case 'character-row':  return <CharacterRow     key={idx} row={row} />;
    case 'eq-strip':       return <EQStripRow       key={idx} row={row} />;
    default:
      return (
        <div key={idx} className="rr-row rr-row--unknown" style={{ height: row.height || 100 }}>
          unknown row: {row.kind}
        </div>
      );
  }
}
