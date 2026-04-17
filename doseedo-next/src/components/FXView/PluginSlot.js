import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import pluginFX from '../../services/pluginFX';
import styles from './PluginSlot.module.css';

// Icons for different plugin categories
const CATEGORY_ICONS = {
  'Spatial': 'fa-water',
  'Time-Based': 'fa-clock',
  'Modulation': 'fa-wave-square',
  'Dynamics': 'fa-compress',
  'EQ': 'fa-sliders-h',
  'Filter': 'fa-filter',
  'Distortion': 'fa-bolt',
  'Spectral': 'fa-chart-bar',
  'Creative': 'fa-wand-magic-sparkles',
  'Utility': 'fa-wrench',
  'Analysis': 'fa-chart-line',
  'Vintage': 'fa-record-vinyl',
  'default': 'fa-plug'
};

/**
 * PluginSlot - A single FX slot in the chain
 * Displays plugin controls and allows swapping plugins
 */
const PluginSlot = ({ slotId, compact = false }) => {
  const { state, dispatch } = useApp();
  const [isExpanded, setIsExpanded] = useState(false);
  const [showPluginSelector, setShowPluginSelector] = useState(false);
  const [availablePlugins, setAvailablePlugins] = useState({});
  const [pluginParams, setPluginParams] = useState([]);

  // Get slot data from state
  const slotData = state.fxSlots?.[slotId] || {
    pluginName: null,
    enabled: false,
    params: {}
  };

  const { pluginName, enabled, params } = slotData;

  // Get available plugins on mount
  useEffect(() => {
    if (pluginFX.initialized) {
      setAvailablePlugins(pluginFX.getAvailablePlugins());
    }
  }, [pluginFX.initialized]);

  // Get plugin parameters when plugin changes
  useEffect(() => {
    if (pluginFX.initialized && pluginName) {
      const slotInfo = pluginFX.getSlotInfo(slotId);
      if (slotInfo && slotInfo.parameterConfigs) {
        setPluginParams(slotInfo.parameterConfigs);
      }
    } else {
      setPluginParams([]);
    }
  }, [pluginName, slotId]);

  // Get plugin info
  const pluginInfo = useMemo(() => {
    if (!pluginName) return null;
    return pluginFX.getPluginInfo(pluginName);
  }, [pluginName]);

  // Get icon for plugin category
  const getIcon = useCallback(() => {
    if (!pluginInfo) return CATEGORY_ICONS.default;
    return CATEGORY_ICONS[pluginInfo.category] || CATEGORY_ICONS.default;
  }, [pluginInfo]);

  // Handle plugin change
  const handlePluginChange = async (newPluginName) => {
    if (newPluginName === pluginName) {
      setShowPluginSelector(false);
      return;
    }

    // Update state
    dispatch({
      type: 'SET_FX_SLOT',
      payload: {
        slotId,
        pluginName: newPluginName,
        enabled: true,
        params: {}
      }
    });

    // Update audio service
    if (pluginFX.initialized) {
      await pluginFX.setSlot(slotId, newPluginName, true);
    }

    setShowPluginSelector(false);
  };

  // Handle clear slot
  const handleClearSlot = async () => {
    dispatch({
      type: 'CLEAR_FX_SLOT',
      payload: { slotId }
    });

    if (pluginFX.initialized) {
      pluginFX.clearSlot(slotId);
    }

    setShowPluginSelector(false);
  };

  // Handle bypass toggle
  const handleBypassToggle = () => {
    dispatch({
      type: 'TOGGLE_FX_SLOT',
      payload: { slotId }
    });

    if (pluginFX.initialized) {
      pluginFX.setSlotEnabled(slotId, !enabled);
    }
  };

  // Handle parameter change
  const handleParamChange = (paramName, value) => {
    // Update state
    dispatch({
      type: 'SET_FX_SLOT_PARAM',
      payload: {
        slotId,
        paramName,
        value
      }
    });

    // Update audio service
    if (pluginFX.initialized) {
      pluginFX.setParameter(slotId, paramName, value);
    }
  };

  // Format parameter value for display
  const formatParamValue = (param, value) => {
    if (param.unit === 'Hz') {
      if (value >= 1000) {
        return `${(value / 1000).toFixed(1)}kHz`;
      }
      return `${Math.round(value)}Hz`;
    }
    if (param.unit === 'dB') {
      return `${value.toFixed(1)}dB`;
    }
    if (param.unit === 's' || param.unit === 'seconds') {
      if (value < 1) {
        return `${Math.round(value * 1000)}ms`;
      }
      return `${value.toFixed(2)}s`;
    }
    if (param.unit === 'ms') {
      return `${Math.round(value)}ms`;
    }
    if (param.unit === '%' || param.max === 1) {
      return `${Math.round(value * 100)}%`;
    }
    if (param.type === 'discrete') {
      return value.toString();
    }
    return value.toFixed(2);
  };

  // Render empty slot
  if (!pluginName) {
    return (
      <div className={`${styles.slot} ${styles.empty}`}>
        <div className={styles.slotHeader}>
          <span className={styles.slotNumber}>{slotId + 1}</span>
          <span className={styles.emptyLabel}>Empty Slot</span>
        </div>
        <button
          className={styles.addButton}
          onClick={() => setShowPluginSelector(true)}
        >
          <i className="fa-solid fa-plus"></i>
          <span>Add Effect</span>
        </button>

        {showPluginSelector && (
          <PluginSelector
            availablePlugins={availablePlugins}
            onSelect={handlePluginChange}
            onClose={() => setShowPluginSelector(false)}
          />
        )}
      </div>
    );
  }

  // Render plugin slot
  return (
    <div className={`${styles.slot} ${!enabled ? styles.bypassed : ''} ${compact ? styles.compact : ''}`}>
      <div className={styles.slotHeader}>
        <span className={styles.slotNumber}>{slotId + 1}</span>
        <i className={`fa-solid ${getIcon()}`}></i>
        <span className={styles.pluginName}>{pluginName}</span>

        <div className={styles.slotControls}>
          <button
            className={`${styles.bypassButton} ${!enabled ? styles.active : ''}`}
            onClick={handleBypassToggle}
            title={enabled ? 'Bypass' : 'Enable'}
          >
            <i className={`fa-solid ${enabled ? 'fa-power-off' : 'fa-ban'}`}></i>
          </button>
          <button
            className={styles.expandButton}
            onClick={() => setIsExpanded(!isExpanded)}
            title={isExpanded ? 'Collapse' : 'Expand'}
          >
            <i className={`fa-solid ${isExpanded ? 'fa-chevron-up' : 'fa-chevron-down'}`}></i>
          </button>
          <button
            className={styles.swapButton}
            onClick={() => setShowPluginSelector(true)}
            title="Change Plugin"
          >
            <i className="fa-solid fa-exchange-alt"></i>
          </button>
        </div>
      </div>

      {/* Quick Parameters (always visible) */}
      <div className={styles.quickParams}>
        {pluginParams.slice(0, 3).map(param => (
          <div key={param.name} className={styles.quickParam}>
            <label>{param.label || param.name}</label>
            <input
              type="range"
              min={param.min}
              max={param.max}
              step={(param.max - param.min) / 100}
              value={params[param.name] ?? param.default ?? param.value}
              onChange={(e) => handleParamChange(param.name, parseFloat(e.target.value))}
              disabled={!enabled}
            />
            <span className={styles.paramValue}>
              {formatParamValue(param, params[param.name] ?? param.default ?? param.value)}
            </span>
          </div>
        ))}
      </div>

      {/* Expanded Parameters */}
      {isExpanded && pluginParams.length > 3 && (
        <div className={styles.expandedParams}>
          {pluginParams.slice(3).map(param => (
            <div key={param.name} className={styles.param}>
              <label>{param.label || param.name}</label>
              <input
                type="range"
                min={param.min}
                max={param.max}
                step={(param.max - param.min) / 100}
                value={params[param.name] ?? param.default ?? param.value}
                onChange={(e) => handleParamChange(param.name, parseFloat(e.target.value))}
                disabled={!enabled}
              />
              <span className={styles.paramValue}>
                {formatParamValue(param, params[param.name] ?? param.default ?? param.value)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Plugin Selector Modal */}
      {showPluginSelector && (
        <PluginSelector
          availablePlugins={availablePlugins}
          currentPlugin={pluginName}
          onSelect={handlePluginChange}
          onClear={handleClearSlot}
          onClose={() => setShowPluginSelector(false)}
        />
      )}
    </div>
  );
};

/**
 * PluginSelector - Dropdown to select a plugin
 */
const PluginSelector = ({ availablePlugins, currentPlugin, onSelect, onClear, onClose }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState(null);

  // Get categories
  const categories = Object.keys(availablePlugins).sort();

  // Filter plugins by search and category
  const filteredPlugins = useMemo(() => {
    const result = {};

    for (const [category, plugins] of Object.entries(availablePlugins)) {
      if (selectedCategory && category !== selectedCategory) continue;

      const filtered = plugins.filter(plugin =>
        plugin.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        plugin.description?.toLowerCase().includes(searchTerm.toLowerCase())
      );

      if (filtered.length > 0) {
        result[category] = filtered;
      }
    }

    return result;
  }, [availablePlugins, searchTerm, selectedCategory]);

  return (
    <div className={styles.selectorOverlay} onClick={onClose}>
      <div className={styles.selectorModal} onClick={e => e.stopPropagation()}>
        <div className={styles.selectorHeader}>
          <h4>Select Effect</h4>
          <button className={styles.closeButton} onClick={onClose}>
            <i className="fa-solid fa-times"></i>
          </button>
        </div>

        <div className={styles.selectorSearch}>
          <i className="fa-solid fa-search"></i>
          <input
            type="text"
            placeholder="Search effects..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            autoFocus
          />
        </div>

        <div className={styles.categoryTabs}>
          <button
            className={`${styles.categoryTab} ${!selectedCategory ? styles.active : ''}`}
            onClick={() => setSelectedCategory(null)}
          >
            All
          </button>
          {categories.map(category => (
            <button
              key={category}
              className={`${styles.categoryTab} ${selectedCategory === category ? styles.active : ''}`}
              onClick={() => setSelectedCategory(category)}
            >
              {category}
            </button>
          ))}
        </div>

        <div className={styles.pluginList}>
          {Object.entries(filteredPlugins).map(([category, plugins]) => (
            <div key={category} className={styles.categoryGroup}>
              <h5>{category}</h5>
              {plugins.map(plugin => (
                <button
                  key={plugin.name}
                  className={`${styles.pluginOption} ${plugin.name === currentPlugin ? styles.current : ''}`}
                  onClick={() => onSelect(plugin.name)}
                >
                  <i className={`fa-solid ${CATEGORY_ICONS[category] || CATEGORY_ICONS.default}`}></i>
                  <div className={styles.pluginInfo}>
                    <span className={styles.pluginOptionName}>{plugin.name}</span>
                    {plugin.description && (
                      <span className={styles.pluginDescription}>{plugin.description}</span>
                    )}
                  </div>
                  {plugin.name === currentPlugin && (
                    <i className="fa-solid fa-check"></i>
                  )}
                </button>
              ))}
            </div>
          ))}
        </div>

        {onClear && (
          <div className={styles.selectorFooter}>
            <button className={styles.clearButton} onClick={onClear}>
              <i className="fa-solid fa-trash"></i>
              Remove Effect
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default PluginSlot;
