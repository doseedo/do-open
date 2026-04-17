import React from 'react';
import { generateKnobSVG, generateSliderSVG, generateButtonSVG, KNOB_STYLES, SLIDER_STYLES, BUTTON_STYLES, MOOG_KNOB_IMAGES } from './svgComponentLibrary';
import styles from './PluginCreator.module.css';

const PropertyPanel = ({ component, onUpdate, onDelete, onDuplicate, onOpenImageBrowser, paramMapping, dspConfig, onRemapParam }) => {
  if (!component) return null;

  const typeName = component.type.charAt(0).toUpperCase() + component.type.slice(1);
  const isTextType = ['label', 'button', 'dropdown'].includes(component.type);
  const isPanel = component.type === 'panel';
  const isImage = component.type === 'image';
  const isInteractive = ['knob', 'slider', 'button'].includes(component.type);
  const isBindable = ['knob', 'slider', 'xy-pad', 'click-knob'].includes(component.type);

  // Find which param this component is bound to
  const boundParamId = paramMapping ? Object.entries(paramMapping).find(([, cid]) => cid === component.id)?.[0] || '' : '';

  // Get the appropriate style list for this component type
  const getStyleOptions = () => {
    if (component.type === 'knob') return KNOB_STYLES;
    if (component.type === 'slider') return SLIDER_STYLES;
    if (component.type === 'button') return BUTTON_STYLES;
    return {};
  };

  // Handle style change — regenerate SVG from library
  const handleStyleChange = (style) => {
    // Moog styles with pre-baked images: use sprite path for arc-indicator rendering
    if (component.type === 'knob' && MOOG_KNOB_IMAGES[style]) {
      onUpdate({ svgStyle: style, svg: '', sprite: MOOG_KNOB_IMAGES[style].full, flatRing: MOOG_KNOB_IMAGES[style].flat });
      return;
    }
    const uid = (component.id || 'c').replace(/[^a-zA-Z0-9]/g, '').slice(0, 12);
    const params = {
      width: component.width || 60, height: component.height || 60,
      bodyColor: component.bodyColor || component.color || '#333',
      indicatorColor: component.indicatorColor || component.color || '#fff',
      accentColor: component.accentColor || component.color || '#888',
      uid, label: component.label || '',
    };
    let svg = '';
    if (component.type === 'knob') svg = generateKnobSVG(style, params);
    else if (component.type === 'slider') svg = generateSliderSVG(style, params);
    else if (component.type === 'button') svg = generateButtonSVG(style, params);
    onUpdate({ svgStyle: style, svg, sprite: '' });
  };

  return (
    <div className={styles.propertyPanel}>
      <div className={styles.propHeader}>
        <span className={styles.propTitle}>{typeName}</span>
        <div style={{ display: 'flex', gap: 4 }}>
          {onDuplicate && (
            <button className={styles.propDuplicateBtn} onClick={onDuplicate} title="Duplicate (Ctrl+D)">
              <i className="fa-solid fa-copy" />
            </button>
          )}
          <button className={styles.propDeleteBtn} onClick={onDelete} title="Delete">
            <i className="fa-solid fa-trash" />
          </button>
        </div>
      </div>

      {/* Label */}
      <div className={styles.propRow}>
        <label>Label</label>
        <input
          type="text"
          value={component.label}
          onChange={(e) => onUpdate({ label: e.target.value })}
          className={styles.propInput}
        />
      </div>

      {/* Color */}
      <div className={styles.propRow}>
        <label>Color</label>
        <div className={styles.colorControl}>
          <input
            type="color"
            value={component.color}
            onChange={(e) => onUpdate({ color: e.target.value })}
            className={styles.colorPicker}
          />
          <span className={styles.colorHex}>{component.color}</span>
        </div>
      </div>

      {/* Size */}
      <div className={styles.propRow}>
        <label>Size</label>
        <div className={styles.sizeInputs}>
          <input
            type="number"
            value={component.width}
            onChange={(e) => onUpdate({ width: Math.max(10, parseInt(e.target.value) || 10) })}
            className={styles.propInputSmall}
            min={10} max={800}
          />
          <span>&times;</span>
          <input
            type="number"
            value={component.height}
            onChange={(e) => onUpdate({ height: Math.max(10, parseInt(e.target.value) || 10) })}
            className={styles.propInputSmall}
            min={10} max={800}
          />
        </div>
      </div>

      {/* Position */}
      <div className={styles.propRow}>
        <label>Pos</label>
        <div className={styles.sizeInputs}>
          <input
            type="number"
            value={Math.round(component.x)}
            onChange={(e) => onUpdate({ x: parseInt(e.target.value) || 0 })}
            className={styles.propInputSmall}
          />
          <span>,</span>
          <input
            type="number"
            value={Math.round(component.y)}
            onChange={(e) => onUpdate({ y: parseInt(e.target.value) || 0 })}
            className={styles.propInputSmall}
          />
        </div>
      </div>

      <div className={styles.propDivider} />

      {/* Opacity */}
      <div className={styles.propRow}>
        <label>Opacity</label>
        <input
          type="range"
          min={0} max={1} step={0.05}
          value={component.opacity ?? 1}
          onChange={(e) => onUpdate({ opacity: parseFloat(e.target.value) })}
          className={styles.propRange}
        />
        <span className={styles.propRangeVal}>{Math.round((component.opacity ?? 1) * 100)}%</span>
      </div>

      {/* Rotation */}
      <div className={styles.propRow}>
        <label>Rotate</label>
        <input
          type="number"
          value={component.rotation ?? 0}
          onChange={(e) => onUpdate({ rotation: parseInt(e.target.value) || 0 })}
          className={styles.propInputSmall}
          min={-360} max={360}
        />
        <span className={styles.propUnit}>deg</span>
      </div>

      {/* Border Radius */}
      <div className={styles.propRow}>
        <label>Radius</label>
        <input
          type="number"
          value={component.borderRadius ?? 0}
          onChange={(e) => onUpdate({ borderRadius: Math.max(0, parseInt(e.target.value) || 0) })}
          className={styles.propInputSmall}
          min={0} max={200}
        />
        <span className={styles.propUnit}>px</span>
      </div>

      {/* Z-Index */}
      <div className={styles.propRow}>
        <label>Layer</label>
        <input
          type="number"
          value={component.zIndex ?? 1}
          onChange={(e) => onUpdate({ zIndex: parseInt(e.target.value) || 0 })}
          className={styles.propInputSmall}
          min={0} max={100}
        />
      </div>

      {/* Sprite indicator */}
      {isInteractive && component.sprite && (
        <div className={styles.propRow}>
          <label>Visual</label>
          <span style={{ fontSize: 10, padding: '2px 8px', borderRadius: 10, background: 'rgba(255,152,0,0.2)', color: '#ffb74d' }}>
            Sprite
          </span>
          <button
            onClick={() => onUpdate({ sprite: '', svg: '', svgStyle: '' })}
            style={{ marginLeft: 'auto', background: 'none', border: 'none', color: 'rgba(255,255,255,0.3)', fontSize: 11, cursor: 'pointer', padding: '2px 6px' }}
            title="Clear sprite"
          >
            <i className="fa-solid fa-xmark" />
          </button>
        </div>
      )}

      {/* SVG Style dropdown — for knobs, sliders, buttons */}
      {isInteractive && (
        <div className={styles.propRow}>
          <label>Style</label>
          <select
            value={component.svgStyle || component.knobStyle || 'default'}
            onChange={(e) => handleStyleChange(e.target.value)}
            className={styles.propInput}
            style={{ cursor: 'pointer' }}
          >
            {!component.svgStyle && !Object.keys(getStyleOptions()).includes(component.knobStyle) && (
              <option value={component.knobStyle || 'default'}>
                {(component.knobStyle || 'default').charAt(0).toUpperCase() + (component.knobStyle || 'default').slice(1)}
              </option>
            )}
            {Object.entries(getStyleOptions()).map(([key, desc]) => (
              <option key={key} value={key}>{key}</option>
            ))}
          </select>
        </div>
      )}

      {/* DSP Param Binding */}
      {isBindable && dspConfig?.parameters?.length > 0 && onRemapParam && (
        <div className={styles.propRow}>
          <label>Param</label>
          <select
            value={boundParamId}
            onChange={(e) => onRemapParam(component.id, e.target.value || null)}
            className={styles.propInput}
            style={{ cursor: 'pointer', fontSize: 11 }}
          >
            <option value="">— Unbound —</option>
            {dspConfig.parameters.map(p => (
              <option key={p.id} value={p.id}>
                {p.name || p.id}
                {paramMapping[p.id] && paramMapping[p.id] !== component.id ? ' (used)' : ''}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Font size - only for text types */}
      {isTextType && (
        <div className={styles.propRow}>
          <label>Font</label>
          <input
            type="number"
            value={component.fontSize ?? 13}
            onChange={(e) => onUpdate({ fontSize: Math.max(8, parseInt(e.target.value) || 13) })}
            className={styles.propInputSmall}
            min={8} max={72}
          />
          <span className={styles.propUnit}>px</span>
        </div>
      )}

      {/* Font family - only for text types */}
      {isTextType && (
        <div className={styles.propRow}>
          <label>Family</label>
          <select
            value={component.fontFamily || 'Inter'}
            onChange={(e) => onUpdate({ fontFamily: e.target.value })}
            className={styles.propInput}
            style={{ cursor: 'pointer' }}
          >
            {['Inter', 'Roboto', 'Helvetica', 'Arial', 'Georgia', 'monospace'].map(f => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        </div>
      )}

      {/* Font weight - only for text types */}
      {isTextType && (
        <div className={styles.propRow}>
          <label>Weight</label>
          <select
            value={component.fontWeight ?? 400}
            onChange={(e) => onUpdate({ fontWeight: parseInt(e.target.value) })}
            className={styles.propInput}
            style={{ cursor: 'pointer' }}
          >
            <option value={400}>Normal</option>
            <option value={500}>Medium</option>
            <option value={700}>Bold</option>
          </select>
        </div>
      )}

      {/* Text alignment - only for text types */}
      {isTextType && (
        <div className={styles.propRow}>
          <label>Align</label>
          <div style={{ display: 'inline-flex', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 4, overflow: 'hidden' }}>
            {[
              { value: 'left', icon: 'fa-align-left' },
              { value: 'center', icon: 'fa-align-center' },
              { value: 'right', icon: 'fa-align-right' },
            ].map(({ value, icon }) => (
              <button
                key={value}
                onClick={() => onUpdate({ textAlign: value })}
                title={value.charAt(0).toUpperCase() + value.slice(1)}
                style={{
                  background: (component.textAlign || 'left') === value ? 'rgba(255,255,255,0.15)' : 'transparent',
                  border: 'none',
                  color: (component.textAlign || 'left') === value ? '#fff' : 'rgba(255,255,255,0.5)',
                  padding: '4px 8px',
                  cursor: 'pointer',
                  fontSize: 11,
                  borderRight: value !== 'right' ? '1px solid rgba(255,255,255,0.1)' : 'none',
                }}
              >
                <i className={`fa-solid ${icon}`} />
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Panel-specific */}
      {isPanel && (
        <>
          <div className={styles.propDivider} />
          <div className={styles.propRow}>
            <label>Border</label>
            <div className={styles.colorControl}>
              <input
                type="color"
                value={(() => {
                  const v = component.borderColor || 'rgba(255,255,255,0.1)';
                  return v.startsWith('#') && (v.length === 7 || v.length === 4) ? v : '#ffffff';
                })()}
                onChange={(e) => onUpdate({ borderColor: e.target.value })}
                className={styles.colorPicker}
              />
              <input
                type="text"
                value={component.borderColor || 'rgba(255,255,255,0.1)'}
                onChange={(e) => onUpdate({ borderColor: e.target.value })}
                className={styles.colorHex}
                style={{ cursor: 'text', flex: 1 }}
                placeholder="rgba or hex"
              />
            </div>
          </div>
          <div className={styles.propRow}>
            <label>B.Width</label>
            <input
              type="number"
              value={component.borderWidth ?? 1}
              onChange={(e) => onUpdate({ borderWidth: Math.max(0, Math.min(10, parseInt(e.target.value) || 0)) })}
              className={styles.propInputSmall}
              min={0} max={10}
            />
            <span className={styles.propUnit}>px</span>
          </div>
          <div className={styles.propRow}>
            <label>Fill</label>
            <div className={styles.colorControl}>
              <input
                type="color"
                value={(() => {
                  const v = component.bgColor || 'rgba(255,255,255,0.03)';
                  return v.startsWith('#') && (v.length === 7 || v.length === 4) ? v : '#1a1a1a';
                })()}
                onChange={(e) => onUpdate({ bgColor: e.target.value })}
                className={styles.colorPicker}
              />
              <input
                type="text"
                value={component.bgColor || 'rgba(255,255,255,0.03)'}
                onChange={(e) => onUpdate({ bgColor: e.target.value })}
                className={styles.colorHex}
                style={{ cursor: 'text', flex: 1 }}
                placeholder="rgba or hex"
              />
            </div>
          </div>
        </>
      )}

      {/* Image-specific */}
      {isImage && (
        <>
          <div className={styles.propDivider} />
          <div className={styles.propRow}>
            <label>Image</label>
            <input
              type="text"
              value={component.image || ''}
              onChange={(e) => onUpdate({ image: e.target.value })}
              className={styles.propInput}
              placeholder="Image URL"
            />
          </div>
          {onOpenImageBrowser && (
            <button className={styles.browseImageBtn} onClick={onOpenImageBrowser}>
              <i className="fa-solid fa-image" /> Browse Images
            </button>
          )}
        </>
      )}
    </div>
  );
};

export default PropertyPanel;
