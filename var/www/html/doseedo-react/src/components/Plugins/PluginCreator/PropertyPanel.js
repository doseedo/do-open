import React from 'react';
import styles from './PluginCreator.module.css';

const PropertyPanel = ({ component, onUpdate, onDelete, onDuplicate, onOpenImageBrowser }) => {
  if (!component) return null;

  const typeName = component.type.charAt(0).toUpperCase() + component.type.slice(1);
  const isTextType = ['label', 'button', 'dropdown'].includes(component.type);
  const isPanel = component.type === 'panel';
  const isImage = component.type === 'image';

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

      {/* Panel-specific */}
      {isPanel && (
        <>
          <div className={styles.propDivider} />
          <div className={styles.propRow}>
            <label>Border</label>
            <input
              type="text"
              value={component.borderColor || 'rgba(255,255,255,0.1)'}
              onChange={(e) => onUpdate({ borderColor: e.target.value })}
              className={styles.propInput}
              placeholder="rgba or hex"
            />
          </div>
          <div className={styles.propRow}>
            <label>Fill</label>
            <input
              type="text"
              value={component.bgColor || 'rgba(255,255,255,0.03)'}
              onChange={(e) => onUpdate({ bgColor: e.target.value })}
              className={styles.propInput}
              placeholder="rgba or hex"
            />
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
