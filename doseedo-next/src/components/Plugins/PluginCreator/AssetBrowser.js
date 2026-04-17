import React, { useState, useCallback, useMemo } from 'react';
import { SVGKnobRenderer, SVGSliderRenderer, SVGButtonRenderer } from './SVGRenderer';
import { SpriteKnobRenderer, SpriteSliderRenderer, SpriteButtonRenderer } from './SpriteRenderer';
import { generateKnobSVG, KNOB_STYLES, MOOG_KNOB_IMAGES, generateSliderSVG, SLIDER_STYLES, generateButtonSVG, BUTTON_STYLES } from './svgComponentLibrary';
import { generateFluxComponentImage } from './fluxComponentCache';
import styles from './PluginCreator.module.css';

const ASSET_TYPES = [
  { id: 'knobs', label: 'Knobs', icon: 'fa-circle-dot' },
  { id: 'sliders', label: 'Sliders', icon: 'fa-sliders' },
  { id: 'meters', label: 'Meters', icon: 'fa-gauge-high' },
  { id: 'buttons', label: 'Buttons', icon: 'fa-square' },
];

const makeStyleList = (stylesObj) => Object.entries(stylesObj)
  .filter(([id]) => id !== 'flux')
  .map(([id, desc]) => ({ id, label: id.split('-').map(w => w[0].toUpperCase() + w.slice(1)).join(' '), desc }));

const SVG_KNOB_STYLES = makeStyleList(KNOB_STYLES);
const SVG_SLIDER_STYLES = makeStyleList(SLIDER_STYLES);
const SVG_BUTTON_STYLES = makeStyleList(BUTTON_STYLES);

const AssetBrowser = ({ onAddToCanvas }) => {
  const [view, setView] = useState('grid');

  // Shared browse state
  const [browseSelected, setBrowseSelected] = useState(null);
  const [bodyColor, setBodyColor] = useState('#333333');
  const [indicatorColor, setIndicatorColor] = useState('#667eea');
  const [accentColor, setAccentColor] = useState('#888888');
  const [browsePreviewValue, setBrowsePreviewValue] = useState(0.65);
  const [buttonPressed, setButtonPressed] = useState(false);

  // Shared generate state
  const [styleDescription, setStyleDescription] = useState('');
  const [generatedAsset, setGeneratedAsset] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [previewValue, setPreviewValue] = useState(0.65);
  const [genIndicatorColor, setGenIndicatorColor] = useState('#667eea');
  const [genButtonPressed, setGenButtonPressed] = useState(false);

  // Knob-specific rotation controls
  const [innerRotates, setInnerRotates] = useState(false);
  const [outerRotates, setOuterRotates] = useState(true);
  const [innerRotateSpeed, setInnerRotateSpeed] = useState(1);
  const [outerRotateSpeed, setOuterRotateSpeed] = useState(1);
  const [showInnerSettings, setShowInnerSettings] = useState(false);
  const [showOuterSettings, setShowOuterSettings] = useState(false);

  // Meter state
  const [meterColor, setMeterColor] = useState('#4caf50');

  // -- Helpers --
  const resetGenState = useCallback(() => {
    setGeneratedAsset(null);
    setError(null);
    setStyleDescription('');
    setPreviewValue(0.65);
    setGenButtonPressed(false);
  }, []);

  const resetBrowseState = useCallback(() => {
    setBrowseSelected(null);
    setBrowsePreviewValue(0.65);
    setButtonPressed(false);
  }, []);

  const handleBrowseSelect = useCallback((styleId) => {
    setBrowseSelected(styleId);
    setBrowsePreviewValue(0.65);
    setButtonPressed(false);
  }, []);

  // -- Mini SVG preview generators (memoized) --
  const knobMiniPreviews = useMemo(() => {
    const p = {};
    SVG_KNOB_STYLES.forEach(s => {
      if (!MOOG_KNOB_IMAGES[s.id]) {
        p[s.id] = generateKnobSVG(s.id, { width: 40, height: 40, bodyColor: '#444', indicatorColor: '#667eea', accentColor: '#888', uid: `mini-k-${s.id}` });
      }
    });
    return p;
  }, []);

  const sliderMiniPreviews = useMemo(() => {
    const p = {};
    SVG_SLIDER_STYLES.forEach(s => {
      p[s.id] = generateSliderSVG(s.id, { width: 20, height: 50, bodyColor: '#444', indicatorColor: '#667eea', accentColor: '#888', uid: `mini-s-${s.id}` });
    });
    return p;
  }, []);

  const buttonMiniPreviews = useMemo(() => {
    const p = {};
    SVG_BUTTON_STYLES.forEach(s => {
      p[s.id] = generateButtonSVG(s.id, { width: 50, height: 22, bodyColor: '#444', indicatorColor: '#667eea', accentColor: '#888', uid: `mini-b-${s.id}`, label: 'On' });
    });
    return p;
  }, []);

  // ═══════════════════════════════════════════════════════════
  // BROWSE: Generate preview SVG for selected style
  // ═══════════════════════════════════════════════════════════

  const browsePreviewSvg = useMemo(() => {
    if (!browseSelected) return null;
    const opts = { bodyColor, indicatorColor, accentColor, uid: `browse-preview-${browseSelected}` };

    if (view === 'browse-knobs') {
      if (MOOG_KNOB_IMAGES[browseSelected]) return null;
      return generateKnobSVG(browseSelected, { ...opts, width: 80, height: 80 });
    }
    if (view === 'browse-sliders') {
      return generateSliderSVG(browseSelected, { ...opts, width: 30, height: 120 });
    }
    if (view === 'browse-buttons') {
      return generateButtonSVG(browseSelected, { ...opts, width: 70, height: 28, label: 'Button' });
    }
    return null;
  }, [browseSelected, bodyColor, indicatorColor, accentColor, view]);

  // ═══════════════════════════════════════════════════════════
  // BROWSE: Add selected item to canvas
  // ═══════════════════════════════════════════════════════════

  const handleBrowseAdd = useCallback(() => {
    if (!browseSelected && view !== 'browse-meters') return;

    if (view === 'browse-knobs') {
      const data = { type: 'knob', width: 60, height: 70, color: indicatorColor, indicatorColor, svgStyle: browseSelected, bodyColor, accentColor };
      if (MOOG_KNOB_IMAGES[browseSelected]) {
        data.sprite = MOOG_KNOB_IMAGES[browseSelected].full;
        data.flatRing = MOOG_KNOB_IMAGES[browseSelected].flat;
      } else {
        data.svg = generateKnobSVG(browseSelected, { width: 60, height: 70, bodyColor, indicatorColor, accentColor, uid: `knob-${Date.now()}` });
      }
      onAddToCanvas(data);
    } else if (view === 'browse-sliders') {
      onAddToCanvas({
        type: 'slider', width: 30, height: 120, color: indicatorColor, indicatorColor, svgStyle: browseSelected, bodyColor, accentColor,
        svg: generateSliderSVG(browseSelected, { width: 30, height: 120, bodyColor, indicatorColor, accentColor, uid: `slider-${Date.now()}` }),
      });
    } else if (view === 'browse-buttons') {
      onAddToCanvas({
        type: 'button', width: 70, height: 28, color: indicatorColor, indicatorColor, svgStyle: browseSelected, bodyColor, accentColor,
        svg: generateButtonSVG(browseSelected, { width: 70, height: 28, bodyColor, indicatorColor, accentColor, uid: `btn-${Date.now()}`, label: 'Button' }),
      });
    } else if (view === 'browse-meters') {
      onAddToCanvas({ type: 'meter', width: 24, height: 100, color: meterColor });
    }
  }, [browseSelected, view, bodyColor, indicatorColor, accentColor, meterColor, onAddToCanvas]);

  // ═══════════════════════════════════════════════════════════
  // GENERATE: Create AI asset
  // ═══════════════════════════════════════════════════════════

  const handleGenerate = useCallback(async () => {
    setIsGenerating(true);
    setError(null);
    const currentView = view;

    try {
      if (currentView === 'generate-knobs') {
        const prompt = styleDescription.trim() || 'professional audio knob with metallic finish and ridged grip';
        const result = await generateFluxComponentImage('knob', prompt, { size: 128 });
        if (result?.dataUrl) {
          setGeneratedAsset({ type: 'knob', spriteUrl: result.dataUrl, flatRingUrl: result.flatRingUrl || null });
        } else setError('Generation failed - no image returned');

      } else if (currentView === 'generate-sliders') {
        const prompt = styleDescription.trim() || 'professional audio fader with brushed aluminum cap';
        const [trackResult, thumbResult] = await Promise.all([
          generateFluxComponentImage('slider-track', prompt, { size: 128 }),
          generateFluxComponentImage('slider-thumb', prompt, { size: 128 }),
        ]);
        if (thumbResult?.dataUrl) {
          setGeneratedAsset({ type: 'slider', thumbUrl: thumbResult.dataUrl, trackUrl: trackResult?.dataUrl || null });
        } else setError('Generation failed - no image returned');

      } else if (currentView === 'generate-buttons') {
        const prompt = styleDescription.trim() || 'professional audio toggle button switch';
        const result = await generateFluxComponentImage('button', prompt, { size: 128 });
        if (result?.dataUrl) {
          setGeneratedAsset({ type: 'button', spriteUrl: result.dataUrl, pressedSpriteUrl: result.pressedDataUrl || result.dataUrl });
        } else setError('Generation failed - no image returned');

      } else if (currentView === 'generate-meters') {
        const prompt = styleDescription.trim() || 'vertical LED level meter with green yellow red segments';
        const result = await generateFluxComponentImage('meter', prompt, { size: 128 });
        if (result?.dataUrl) {
          setGeneratedAsset({ type: 'meter', spriteUrl: result.dataUrl });
        } else setError('Generation failed - no image returned');
      }
    } catch (err) {
      setError(err.message || 'Failed to generate');
    } finally {
      setIsGenerating(false);
    }
  }, [styleDescription, view]);

  // ═══════════════════════════════════════════════════════════
  // GENERATE: Add generated asset to canvas
  // ═══════════════════════════════════════════════════════════

  const handleGenAdd = useCallback(() => {
    if (!generatedAsset) return;
    const { type } = generatedAsset;

    if (type === 'knob') {
      onAddToCanvas({
        type: 'knob', width: 60, height: 70,
        color: genIndicatorColor, indicatorColor: genIndicatorColor,
        sprite: generatedAsset.spriteUrl, flatRing: generatedAsset.flatRingUrl,
        innerRotates, outerRotates, innerRotateSpeed, outerRotateSpeed,
      });
    } else if (type === 'slider') {
      onAddToCanvas({
        type: 'slider', width: 30, height: 120,
        color: genIndicatorColor, indicatorColor: genIndicatorColor,
        sprite: generatedAsset.thumbUrl,
      });
    } else if (type === 'button') {
      onAddToCanvas({
        type: 'button', width: 70, height: 28,
        color: genIndicatorColor, indicatorColor: genIndicatorColor,
        sprite: generatedAsset.spriteUrl,
        pressedSprite: generatedAsset.pressedSpriteUrl,
      });
    } else if (type === 'meter') {
      onAddToCanvas({
        type: 'meter', width: 24, height: 100,
        color: genIndicatorColor, indicatorColor: genIndicatorColor,
        sprite: generatedAsset.spriteUrl,
      });
    }
  }, [generatedAsset, genIndicatorColor, innerRotates, outerRotates, innerRotateSpeed, outerRotateSpeed, onAddToCanvas]);

  // ═══════════════════════════════════════════════════════════
  // GRID VIEW
  // ═══════════════════════════════════════════════════════════

  if (view === 'grid') {
    return (
      <div className={styles.assetBrowseGrid}>
        {ASSET_TYPES.map(a => (
          <div key={a.id} className={styles.assetRow}>
            <div className={styles.assetRowHeader}>
              <i className={`fa-solid ${a.icon}`} />
              <span>{a.label}</span>
            </div>
            <div className={styles.assetRowBtns}>
              <button className={styles.assetSquareBtn} onClick={() => { setView(`generate-${a.id}`); resetGenState(); }}>
                <i className="fa-solid fa-wand-magic-sparkles" />
                <span>Generate</span>
              </button>
              <button className={styles.assetSquareBtn} onClick={() => { setView(`browse-${a.id}`); resetBrowseState(); }}>
                <i className="fa-solid fa-folder-open" />
                <span>Browse</span>
              </button>
            </div>
          </div>
        ))}
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════
  // BROWSE KNOBS
  // ═══════════════════════════════════════════════════════════

  if (view === 'browse-knobs') {
    return (
      <div className={styles.knobGenerator}>
        <button className={styles.generatorBackBtn} onClick={() => setView('grid')}>
          <i className="fa-solid fa-arrow-left" /> Back
        </button>
        <h3 className={styles.generatorTitle}>Knob Library</h3>

        <div className={styles.styleGrid}>
          {SVG_KNOB_STYLES.map(style => (
            <button key={style.id} className={browseSelected === style.id ? styles.styleCardActive : styles.styleCard}
              onClick={() => handleBrowseSelect(style.id)} title={style.desc}>
              {MOOG_KNOB_IMAGES[style.id] ? (
                <img src={MOOG_KNOB_IMAGES[style.id].full} alt="" style={{ width: 40, height: 40, borderRadius: '50%' }} />
              ) : knobMiniPreviews[style.id] ? (
                <div dangerouslySetInnerHTML={{ __html: knobMiniPreviews[style.id] }} />
              ) : null}
              <span>{style.label}</span>
            </button>
          ))}
        </div>

        <div className={styles.colorRow}>
          <label>Body <input type="color" value={bodyColor} onChange={e => setBodyColor(e.target.value)} /></label>
          <label>Indicator <input type="color" value={indicatorColor} onChange={e => setIndicatorColor(e.target.value)} /></label>
          <label>Accent <input type="color" value={accentColor} onChange={e => setAccentColor(e.target.value)} /></label>
        </div>

        {browseSelected && (
          <div className={styles.knobPreview}>
            <div className={styles.knobPreviewArea}>
              {MOOG_KNOB_IMAGES[browseSelected] ? (
                <SpriteKnobRenderer spriteUrl={MOOG_KNOB_IMAGES[browseSelected].full} flatRingUrl={MOOG_KNOB_IMAGES[browseSelected].flat}
                  size={80} value={browsePreviewValue} isTestMode onChange={setBrowsePreviewValue} indicatorColor={indicatorColor} />
              ) : browsePreviewSvg ? (
                <SVGKnobRenderer svgString={browsePreviewSvg} size={80} value={browsePreviewValue} color={indicatorColor} isTestMode onChange={setBrowsePreviewValue} />
              ) : null}
              <div className={styles.previewValueLabel}>{Math.round(browsePreviewValue * 100)}%</div>
            </div>
            <p className={styles.previewHint}>Drag up/down to turn</p>
            <button className={styles.addToCanvasBtn} onClick={handleBrowseAdd}>
              <i className="fa-solid fa-plus" /> Add to Canvas
            </button>
          </div>
        )}
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════
  // BROWSE SLIDERS
  // ═══════════════════════════════════════════════════════════

  if (view === 'browse-sliders') {
    return (
      <div className={styles.knobGenerator}>
        <button className={styles.generatorBackBtn} onClick={() => setView('grid')}>
          <i className="fa-solid fa-arrow-left" /> Back
        </button>
        <h3 className={styles.generatorTitle}>Slider Library</h3>

        <div className={styles.styleGrid}>
          {SVG_SLIDER_STYLES.map(style => (
            <button key={style.id} className={browseSelected === style.id ? styles.styleCardActive : styles.styleCard}
              onClick={() => handleBrowseSelect(style.id)} title={style.desc}>
              {sliderMiniPreviews[style.id] ? (
                <div dangerouslySetInnerHTML={{ __html: sliderMiniPreviews[style.id] }} style={{ display: 'flex', justifyContent: 'center' }} />
              ) : null}
              <span>{style.label}</span>
            </button>
          ))}
        </div>

        <div className={styles.colorRow}>
          <label>Body <input type="color" value={bodyColor} onChange={e => setBodyColor(e.target.value)} /></label>
          <label>Indicator <input type="color" value={indicatorColor} onChange={e => setIndicatorColor(e.target.value)} /></label>
          <label>Accent <input type="color" value={accentColor} onChange={e => setAccentColor(e.target.value)} /></label>
        </div>

        {browseSelected && browsePreviewSvg && (
          <div className={styles.knobPreview}>
            <div className={styles.knobPreviewArea}>
              <SVGSliderRenderer svgString={browsePreviewSvg} width={30} height={120}
                value={browsePreviewValue} color={indicatorColor} isTestMode onChange={setBrowsePreviewValue} />
              <div className={styles.previewValueLabel}>{Math.round(browsePreviewValue * 100)}%</div>
            </div>
            <p className={styles.previewHint}>Drag up/down to slide</p>
            <button className={styles.addToCanvasBtn} onClick={handleBrowseAdd}>
              <i className="fa-solid fa-plus" /> Add to Canvas
            </button>
          </div>
        )}
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════
  // BROWSE BUTTONS
  // ═══════════════════════════════════════════════════════════

  if (view === 'browse-buttons') {
    return (
      <div className={styles.knobGenerator}>
        <button className={styles.generatorBackBtn} onClick={() => setView('grid')}>
          <i className="fa-solid fa-arrow-left" /> Back
        </button>
        <h3 className={styles.generatorTitle}>Button Library</h3>

        <div className={styles.styleGrid}>
          {SVG_BUTTON_STYLES.map(style => (
            <button key={style.id} className={browseSelected === style.id ? styles.styleCardActive : styles.styleCard}
              onClick={() => handleBrowseSelect(style.id)} title={style.desc}>
              {buttonMiniPreviews[style.id] ? (
                <div dangerouslySetInnerHTML={{ __html: buttonMiniPreviews[style.id] }} style={{ display: 'flex', justifyContent: 'center' }} />
              ) : null}
              <span>{style.label}</span>
            </button>
          ))}
        </div>

        <div className={styles.colorRow}>
          <label>Body <input type="color" value={bodyColor} onChange={e => setBodyColor(e.target.value)} /></label>
          <label>Indicator <input type="color" value={indicatorColor} onChange={e => setIndicatorColor(e.target.value)} /></label>
          <label>Accent <input type="color" value={accentColor} onChange={e => setAccentColor(e.target.value)} /></label>
        </div>

        {browseSelected && browsePreviewSvg && (
          <div className={styles.knobPreview}>
            <div className={styles.knobPreviewArea}>
              <div onClick={() => setButtonPressed(p => !p)} style={{ cursor: 'pointer' }}>
                <SVGButtonRenderer svgString={browsePreviewSvg} width={70} height={28} pressed={buttonPressed} color={indicatorColor} />
              </div>
              <div className={styles.previewValueLabel}>{buttonPressed ? 'ON' : 'OFF'}</div>
            </div>
            <p className={styles.previewHint}>Click to toggle</p>
            <button className={styles.addToCanvasBtn} onClick={handleBrowseAdd}>
              <i className="fa-solid fa-plus" /> Add to Canvas
            </button>
          </div>
        )}
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════
  // BROWSE METERS
  // ═══════════════════════════════════════════════════════════

  if (view === 'browse-meters') {
    return (
      <div className={styles.knobGenerator}>
        <button className={styles.generatorBackBtn} onClick={() => setView('grid')}>
          <i className="fa-solid fa-arrow-left" /> Back
        </button>
        <h3 className={styles.generatorTitle}>Meter Library</h3>

        <div className={styles.colorRow}>
          <label>Color <input type="color" value={meterColor} onChange={e => setMeterColor(e.target.value)} /></label>
        </div>

        <div className={styles.knobPreview}>
          <div className={styles.knobPreviewArea}>
            <MeterPreview color={meterColor} width={24} height={100} />
          </div>
          <p className={styles.previewHint}>LED segment meter</p>
          <button className={styles.addToCanvasBtn} onClick={handleBrowseAdd}>
            <i className="fa-solid fa-plus" /> Add to Canvas
          </button>
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════
  // GENERATE VIEW (shared for all types)
  // ═══════════════════════════════════════════════════════════

  const isKnobGen = view === 'generate-knobs';
  const isSliderGen = view === 'generate-sliders';
  const isButtonGen = view === 'generate-buttons';
  const isMeterGen = view === 'generate-meters';

  const titles = { 'generate-knobs': 'AI Knob Generator', 'generate-sliders': 'AI Slider Generator', 'generate-buttons': 'AI Button Generator', 'generate-meters': 'AI Meter Generator' };
  const placeholders = {
    'generate-knobs': 'e.g., tire knob, crystal knob, wooden knob...',
    'generate-sliders': 'e.g., brushed metal fader, glass slider...',
    'generate-buttons': 'e.g., vintage toggle switch, arcade button...',
    'generate-meters': 'e.g., VU meter, LED bar graph, analog needle...',
  };

  return (
    <div className={styles.knobGenerator}>
      <button className={styles.generatorBackBtn} onClick={() => setView('grid')}>
        <i className="fa-solid fa-arrow-left" /> Back
      </button>
      <h3 className={styles.generatorTitle}>{titles[view]}</h3>

      <input
        className={styles.propInput}
        value={styleDescription}
        onChange={e => setStyleDescription(e.target.value)}
        placeholder={placeholders[view]}
        onKeyDown={e => e.key === 'Enter' && !isGenerating && handleGenerate()}
        style={{ width: '100%', boxSizing: 'border-box' }}
      />
      <button className={styles.generateBtn} onClick={handleGenerate} disabled={isGenerating}>
        {isGenerating ? (
          <><i className="fa-solid fa-spinner fa-spin" /> Generating...</>
        ) : (
          <><i className="fa-solid fa-bolt" /> Generate</>
        )}
      </button>

      {/* Knob-specific: rotation layer controls */}
      {isKnobGen && (
        <div className={styles.layerControlStack}>
          <div className={styles.layerControl}>
            <div className={styles.layerControlHeader}>
              <span className={styles.layerControlLabel}>Outer ring</span>
              <button className={showOuterSettings ? styles.layerSettingsOpen : styles.layerSettingsBtn}
                onClick={() => setShowOuterSettings(s => !s)}>
                <i className="fa-solid fa-gear" />
              </button>
              <div className={outerRotates ? styles.toggleSwitchOn : styles.toggleSwitch}
                onClick={() => setOuterRotates(r => !r)}>
                <div className={styles.toggleSwitchThumb} />
              </div>
            </div>
            {showOuterSettings && (
              <div className={styles.layerSettingsPanel}>
                <div className={styles.speedRow}>
                  <span className={styles.speedLabel}>Speed</span>
                  <input type="range" min="0.1" max="3" step="0.1" value={outerRotateSpeed}
                    onChange={e => setOuterRotateSpeed(parseFloat(e.target.value))} className={styles.speedSlider} />
                  <span className={styles.speedValue}>{outerRotateSpeed.toFixed(1)}x</span>
                </div>
              </div>
            )}
          </div>
          <div className={styles.layerControl}>
            <div className={styles.layerControlHeader}>
              <span className={styles.layerControlLabel}>Inner cap</span>
              <button className={showInnerSettings ? styles.layerSettingsOpen : styles.layerSettingsBtn}
                onClick={() => setShowInnerSettings(s => !s)}>
                <i className="fa-solid fa-gear" />
              </button>
              <div className={innerRotates ? styles.toggleSwitchOn : styles.toggleSwitch}
                onClick={() => setInnerRotates(r => !r)}>
                <div className={styles.toggleSwitchThumb} />
              </div>
            </div>
            {showInnerSettings && (
              <div className={styles.layerSettingsPanel}>
                <div className={styles.speedRow}>
                  <span className={styles.speedLabel}>Speed</span>
                  <input type="range" min="0.1" max="3" step="0.1" value={innerRotateSpeed}
                    onChange={e => setInnerRotateSpeed(parseFloat(e.target.value))} className={styles.speedSlider} />
                  <span className={styles.speedValue}>{innerRotateSpeed.toFixed(1)}x</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Indicator color */}
      <div className={styles.colorRow}>
        <label>Indicator <input type="color" value={genIndicatorColor} onChange={e => setGenIndicatorColor(e.target.value)} /></label>
      </div>

      {error && (
        <div style={{ padding: '8px 12px', background: 'rgba(255,100,100,0.1)', border: '1px solid rgba(255,100,100,0.2)', borderRadius: 8, fontSize: 12, color: '#ff8888' }}>
          <i className="fa-solid fa-triangle-exclamation" /> {error}
        </div>
      )}

      {/* Preview */}
      {generatedAsset && (
        <div className={styles.knobPreview}>
          <div className={styles.knobPreviewArea}>
            {generatedAsset.type === 'knob' && (
              <>
                <SpriteKnobRenderer
                  spriteUrl={generatedAsset.spriteUrl} flatRingUrl={generatedAsset.flatRingUrl}
                  size={80} value={previewValue} isTestMode onChange={setPreviewValue}
                  indicatorColor={genIndicatorColor} innerRotates={innerRotates} outerRotates={outerRotates}
                  innerRotateSpeed={innerRotateSpeed} outerRotateSpeed={outerRotateSpeed} />
                <div className={styles.previewValueLabel}>{Math.round(previewValue * 100)}%</div>
              </>
            )}
            {generatedAsset.type === 'slider' && (
              <>
                <SpriteSliderRenderer
                  spriteUrl={generatedAsset.thumbUrl} width={30} height={120}
                  value={previewValue} isTestMode onChange={setPreviewValue} />
                <div className={styles.previewValueLabel}>{Math.round(previewValue * 100)}%</div>
              </>
            )}
            {generatedAsset.type === 'button' && (
              <>
                <div onClick={() => setGenButtonPressed(p => !p)} style={{ cursor: 'pointer' }}>
                  <SpriteButtonRenderer
                    spriteUrl={generatedAsset.spriteUrl}
                    pressedSpriteUrl={generatedAsset.pressedSpriteUrl}
                    width={70} height={28}
                    pressed={genButtonPressed} />
                </div>
                <div className={styles.previewValueLabel}>{genButtonPressed ? 'ON' : 'OFF'}</div>
              </>
            )}
            {generatedAsset.type === 'meter' && (
              <>
                <img src={generatedAsset.spriteUrl} alt="" style={{ width: 24, height: 100, objectFit: 'contain' }} />
              </>
            )}
          </div>
          <p className={styles.previewHint}>
            {isKnobGen ? 'Drag up/down to turn' : isSliderGen ? 'Drag up/down to slide' : isButtonGen ? 'Click to toggle' : 'Meter preview'}
          </p>
          <button className={styles.addToCanvasBtn} onClick={handleGenAdd}>
            <i className="fa-solid fa-plus" /> Add to Canvas
          </button>
        </div>
      )}
    </div>
  );
};

// Simple meter preview component for browse view
const MeterPreview = React.memo(({ color, width, height }) => {
  const segments = 12;
  const segH = (height - 6) / segments;
  const litCount = Math.round(0.65 * segments);
  return (
    <div style={{
      width, height, background: '#111', borderRadius: 3,
      padding: 3, display: 'flex', flexDirection: 'column-reverse', gap: 1,
      border: '1px solid rgba(255,255,255,0.1)',
    }}>
      {[...Array(segments)].map((_, i) => {
        const lit = i < litCount;
        const ratio = i / segments;
        let segColor = color;
        if (ratio > 0.85) segColor = '#ff4444';
        else if (ratio > 0.7) segColor = '#ffaa00';
        return (
          <div key={i} style={{
            flex: 1, minHeight: segH - 1, borderRadius: 1,
            background: lit ? segColor : 'rgba(255,255,255,0.06)',
            opacity: lit ? 0.9 : 0.3,
          }} />
        );
      })}
    </div>
  );
});

export default AssetBrowser;
