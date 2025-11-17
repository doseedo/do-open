import React, { useCallback, useMemo, useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { useGeneration } from '../../hooks/useGeneration';
import { useAudioRecorder } from '../../hooks/useAudioRecorder';
import { isMidiFile, isAudioFile } from '../../utils/audioUtils';
import { sceneToDurations } from '../../services/videoAPI';
import * as generationAPI from '../../services/generationAPI';
import DrumSampler from '../DrumSampler/DrumSampler';
import GlassButtonWrapper from '../GlassButton/GlassButtonWrapper';
import styles from './GenerationPanel.module.css';

/**
 * GenerationPanelOptimized - Optimized generation panel using CSS Grid and CSS Modules
 *
 * Key Optimizations:
 * 1. CSS Grid for layout (no absolute positioning)
 * 2. CSS Modules for scoped styles
 * 3. Compound component pattern for organization
 * 4. Memoization for performance
 * 5. GPU-accelerated transforms
 */

// ========== Custom Dropdown Component ==========
const CustomDropdown = React.memo(({ value, onChange, options }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = React.useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen]);

  const selectedOption = options.find(opt => opt.value === value);

  return (
    <div className={styles.customDropdown} ref={dropdownRef}>
      <div
        className={styles.dropdownTrigger}
        onClick={() => setIsOpen(!isOpen)}
      >
        <span>{selectedOption?.label || 'Select...'}</span>
        <i className={`fa-solid fa-chevron-down ${styles.dropdownIcon} ${isOpen ? styles.open : ''}`}></i>
      </div>
      {isOpen && (
        <div className={styles.dropdownMenu}>
          {options.map((option) => (
            <div
              key={option.value}
              className={`${styles.dropdownItem} ${option.value === value ? styles.selected : ''}`}
              onClick={() => {
                onChange(option.value);
                setIsOpen(false);
              }}
            >
              {option.label}
              {option.value === value && <i className="fa-solid fa-check"></i>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

CustomDropdown.displayName = 'CustomDropdown';

// ========== Collapsible Section Component ==========
const CollapsibleSection = React.memo(({ title, isCollapsed, onToggle, children, number }) => {
  return (
    <div className={styles.section}>
      <div className={styles.sectionHeader} onClick={onToggle}>
        <div className={styles.sectionTitle}>
          {number && <span>{number}.</span>}
          <span>{title}</span>
        </div>
        <i className={`fa-solid fa-caret-down ${styles.expandIcon} ${isCollapsed ? styles.collapsed : ''}`}></i>
      </div>
      <div className={`${styles.sectionContent} ${isCollapsed ? styles.collapsed : ''}`}>
        {!isCollapsed && children}
      </div>
    </div>
  );
});

CollapsibleSection.displayName = 'CollapsibleSection';

// ========== Instrument Selection Component ==========
const InstrumentSelection = React.memo(({ params, updateParam }) => {
  // Instrument group to subgroup mapping - MUST match backend APPROVED_SUBGROUPS
  const instrumentSubgroups = useMemo(() => ({
    piano: ['acoustic_piano', 'keys'],
    guitar: ['acoustic_guitar', 'electric_guitar'],
    bass: ['electric_bass', 'upright_bass'],
    strings: ['ensemble_strings', 'violin', 'cello'],
    brass: ['ensemble_brass', 'trumpet', 'trombone'],
    winds: ['ensemble_winds', 'flute', 'sax']
  }), []);

  // Instrument groups with custom icon images
  const instrumentGroups = useMemo(() => [
    { id: 'piano', label: 'Piano', iconImg: '/assets/icons/piano.png' },
    { id: 'guitar', label: 'Guitar', iconImg: '/assets/icons/acguitar.png' },
    { id: 'bass', label: 'Bass', iconImg: '/assets/icons/elecbass.png' },
    { id: 'strings', label: 'Strings', iconImg: '/assets/icons/violin.png' },
    { id: 'brass', label: 'Brass', iconImg: '/assets/icons/tpt.png' },
    { id: 'winds', label: 'Winds', iconImg: '/assets/icons/sax.png' }
  ], []);

  // Subgroup icon mapping
  const subgroupIcons = useMemo(() => ({
    // Piano subgroups
    'acoustic_piano': '/assets/icons/piano.png',
    'keys': '/assets/icons/keyboard.png',
    // Guitar subgroups
    'acoustic_guitar': '/assets/icons/acguitar.png',
    'electric_guitar': '/assets/icons/elecgtr.png',
    // Bass subgroups
    'electric_bass': '/assets/icons/elecbass.png',
    'upright_bass': '/assets/icons/elecbass.png', // Use same icon for now
    // Strings subgroups
    'violin': '/assets/icons/violin.png',
    'viola': '/assets/icons/violin.png', // Use violin icon
    'cello': '/assets/icons/cello.png',
    'ensemble_strings': '/assets/icons/violin.png', // Use violin icon as fallback
    // Brass subgroups
    'trumpet': '/assets/icons/tpt.png',
    'trombone': '/assets/icons/tbn.png',
    'french_horn': '/assets/icons/brassens.png',
    'tuba': '/assets/icons/tuba.png',
    'ensemble_brass': '/assets/icons/brassens.png',
    // Winds subgroups
    'bassoon': '/assets/icons/clarinet.png', // Use clarinet as fallback
    'clarinet': '/assets/icons/clarinet.png',
    'flute': '/assets/icons/flute.png',
    'oboe': '/assets/icons/flute.png', // Use flute as fallback
    'sax': '/assets/icons/sax.png',
    'ensemble_winds': '/assets/icons/windens.png'
  }), []);

  // Preload all instrument icons on mount
  useEffect(() => {
    const allIcons = new Set([
      ...instrumentGroups.map(g => g.iconImg),
      ...Object.values(subgroupIcons)
    ]);

    allIcons.forEach(iconPath => {
      if (iconPath) {
        const img = new Image();
        img.src = iconPath;
      }
    });
  }, [instrumentGroups, subgroupIcons]);

  // Get available subgroups for current instrument group
  const availableSubgroups = useMemo(() => {
    return instrumentSubgroups[params.instrumentGroup || 'strings'] || instrumentSubgroups.strings;
  }, [params.instrumentGroup, instrumentSubgroups]);

  // Handle instrument group change - update subgroup to first valid option
  const handleGroupChange = useCallback((newGroup) => {
    updateParam('instrumentGroup', newGroup);
    // Set subgroup to first option of new group
    const newSubgroups = instrumentSubgroups[newGroup] || instrumentSubgroups.strings;
    updateParam('instrumentSubgroup', newSubgroups[0]);
  }, [updateParam, instrumentSubgroups]);

  // Handle subgroup change - auto-enable/disable monophonic and arrange mode
  const handleSubgroupChange = useCallback((newSubgroup) => {
    updateParam('instrumentSubgroup', newSubgroup);

    // If ensemble is selected, enable monophonic and arrange mode
    if (newSubgroup === 'ensemble_strings' || newSubgroup === 'ensemble_brass' || newSubgroup === 'ensemble_winds') {
      updateParam('monophonicMode', true);
      updateParam('arrangeMode', true);
    } else {
      // If switching to solo instrument, disable monophonic and arrange mode
      updateParam('monophonicMode', false);
      updateParam('arrangeMode', false);
    }
  }, [updateParam]);

  // Auto-enable monophonic and arrange mode if ensemble is selected on mount/load
  React.useEffect(() => {
    const subgroup = params.instrumentSubgroup;
    if (subgroup === 'ensemble_strings' || subgroup === 'ensemble_brass' || subgroup === 'ensemble_winds') {
      if (!params.monophonicMode || !params.arrangeMode) {
        updateParam('monophonicMode', true);
        updateParam('arrangeMode', true);
      }
    }
  }, [params.instrumentSubgroup, params.monophonicMode, params.arrangeMode, updateParam]);

  return (
    <>
      {/* Instrument Group Grid */}
      <div className={styles.instrumentGroupGrid}>
        {instrumentGroups.map(group => (
          <button
            key={group.id}
            type="button"
            className={`${styles.instrumentGroupBtn} ${params.instrumentGroup === group.id ? styles.active : ''}`}
            onClick={() => handleGroupChange(group.id)}
          >
            <img
              src={group.iconImg}
              alt={group.label}
              style={{
                width: '40px',
                height: '40px',
                objectFit: 'contain',
                filter: 'invert(1)'
              }}
              aria-hidden="true"
            />
            <span>{group.label}</span>
          </button>
        ))}
      </div>

      {/* Instrument Subgroup Grid - Only show when a group is selected */}
      {params.instrumentGroup && (
        <>
          <div className={styles.instrumentGroupLabel} style={{ marginTop: '15px' }}>Instrument Subgroup:</div>
          <div className={styles.instrumentSubgroupGrid}>
            {availableSubgroups.map(subgroup => {
              // Format label - handle ensemble specially
              let displayLabel = subgroup.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
              if (subgroup.startsWith('ensemble_')) {
                displayLabel = 'Ensemble';
              }

              return (
                <button
                  key={subgroup}
                  type="button"
                  className={`${styles.instrumentSubgroupBtn} ${params.instrumentSubgroup === subgroup ? styles.active : ''}`}
                  onClick={() => handleSubgroupChange(subgroup)}
                >
                  {subgroupIcons[subgroup] && (
                    <img
                      src={subgroupIcons[subgroup]}
                      alt={subgroup}
                      style={{
                        width: '40px',
                        height: '40px',
                        objectFit: 'contain',
                        filter: 'invert(1)'
                      }}
                      aria-hidden="true"
                    />
                  )}
                  <span>{displayLabel}</span>
                </button>
              );
            })}
          </div>
        </>
      )}

      <div className={styles.paramRow}>
        <label>
          <div className={styles.paramLabel}>
            <span>Key (for no-input generation):</span>
          </div>
          <select
            className={styles.controlSelect}
            value={params.generationKey || 'C'}
            onChange={(e) => updateParam('generationKey', e.target.value)}
          >
            {['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'].map(key => (
              <option key={key} value={key}>{key}</option>
            ))}
          </select>
        </label>
      </div>
    </>
  );
});

InstrumentSelection.displayName = 'InstrumentSelection';

// ========== MIDI Target Selection Component ==========
const MIDITargetSelection = React.memo(({ params, updateParam }) => {
  const targets = useMemo(() => [
    { id: 'drums', label: 'Drums', icon: 'fa-drum' },
    { id: 'chords', label: 'Chords', icon: 'fa-music' },
    { id: 'melody', label: 'Melody', icon: 'fa-microphone' },
    { id: 'bass', label: 'Bass', icon: 'fa-guitar' }
  ], []);

  return (
    <>
      <div className={styles.midiTargetGrid}>
        {targets.map(target => (
          <button
            key={target.id}
            type="button"
            className={`${styles.midiTargetBtn} ${params.midiTarget === target.id ? styles.active : ''}`}
            onClick={() => updateParam('midiTarget', target.id)}
          >
            <i className={`fa-solid ${target.icon}`}></i>
            <span>{target.label}</span>
          </button>
        ))}
      </div>

      {/* Drum Subgroup Selection - only show when drums is selected */}
      {params.midiTarget === 'drums' && (
        <>
          <div style={{ marginTop: '15px' }}>
            <label htmlFor="drum-subgroup" style={{ color: 'white', display: 'block', marginBottom: '5px', fontSize: '12px' }}>
              Drum Type:
            </label>
            <select
              id="drum-subgroup"
              value={params.drumSubgroup || 'orchestral'}
              onChange={(e) => updateParam('drumSubgroup', e.target.value)}
              style={{
                width: '100%',
                padding: '8px',
                borderRadius: '5px',
                background: '#1a1a2e',
                color: 'white',
                border: '1px solid #444',
                cursor: 'pointer'
              }}
            >
              <option value="orchestral">Orchestral</option>
              <option value="riser">Riser</option>
              <option value="ai_sampler">🤖 AI Drum Sampler</option>
            </select>
          </div>

          {/* Drum Timing Mode - only show for orchestral */}
          {params.drumSubgroup === 'orchestral' && (
            <>
              {/* Active Samples Selection */}
              <div style={{ marginTop: '15px' }}>
                <label style={{ color: 'white', display: 'block', marginBottom: '8px', fontSize: '12px' }}>
                  Active Samples:
                </label>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '8px',
                  padding: '10px',
                  background: '#1a1a2e',
                  borderRadius: '5px',
                  border: '1px solid #444'
                }}>
                  {['bass_drum', 'timpani', 'cymbals', 'percussion'].map(sample => {
                    const activeSamples = params.activeSamples || ['bass_drum'];
                    const isActive = activeSamples.includes(sample);
                    const displayName = sample.split('_').map(word =>
                      word.charAt(0).toUpperCase() + word.slice(1)
                    ).join(' ');

                    return (
                      <label
                        key={sample}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                          color: 'white',
                          fontSize: '11px',
                          cursor: 'pointer',
                          userSelect: 'none'
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={isActive}
                          onChange={(e) => {
                            const newActiveSamples = e.target.checked
                              ? [...activeSamples, sample]
                              : activeSamples.filter(s => s !== sample);
                            // Ensure at least one sample is selected
                            if (newActiveSamples.length > 0) {
                              updateParam('activeSamples', newActiveSamples);
                            }
                          }}
                          style={{ cursor: 'pointer' }}
                        />
                        {displayName}
                      </label>
                    );
                  })}
                </div>
              </div>

              <div style={{ marginTop: '15px' }}>
                <label htmlFor="drum-timing" style={{ color: 'white', display: 'block', marginBottom: '5px', fontSize: '12px' }}>
                  Timing:
                </label>
                <select
                  id="drum-timing"
                  value={params.drumTiming || 'scene_changes'}
                  onChange={(e) => updateParam('drumTiming', e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px',
                    borderRadius: '5px',
                    background: '#1a1a2e',
                    color: 'white',
                    border: '1px solid #444',
                    cursor: 'pointer'
                  }}
                >
                  <option value="scene_changes">Scene Changes</option>
                  <option value="bar_pattern">Bar Pattern</option>
                </select>
              </div>

              {/* Show pattern selector only when bar_pattern is selected */}
              {params.drumTiming === 'bar_pattern' && (
                <div style={{ marginTop: '15px' }}>
                  <label htmlFor="drum-pattern" style={{ color: 'white', display: 'block', marginBottom: '5px', fontSize: '12px' }}>
                    Pattern (downbeat interval):
                  </label>
                  <select
                    id="drum-pattern"
                    value={params.drumPattern || 1}
                    onChange={(e) => updateParam('drumPattern', parseInt(e.target.value))}
                    style={{
                      width: '100%',
                      padding: '8px',
                      borderRadius: '5px',
                      background: '#1a1a2e',
                      color: 'white',
                      border: '1px solid #444',
                      cursor: 'pointer'
                    }}
                  >
                    <option value="1">1 Bar (every downbeat)</option>
                    <option value="2">2 Bar (every 2 bars)</option>
                    <option value="4">4 Bar (every 4 bars)</option>
                  </select>
                </div>
              )}
            </>
          )}

          {/* Riser Timing Mode - only show for riser */}
          {params.drumSubgroup === 'riser' && (
            <>
              <div style={{ marginTop: '15px' }}>
                <label htmlFor="riser-timing" style={{ color: 'white', display: 'block', marginBottom: '5px', fontSize: '12px' }}>
                  Timing:
                </label>
                <select
                  id="riser-timing"
                  value={params.riserTiming || 'scene_changes'}
                  onChange={(e) => updateParam('riserTiming', e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px',
                    borderRadius: '5px',
                    background: '#1a1a2e',
                    color: 'white',
                    border: '1px solid #444',
                    cursor: 'pointer'
                  }}
                >
                  <option value="scene_changes">Scene Changes (2s before)</option>
                  <option value="bar_pattern">Bar Pattern</option>
                </select>
              </div>

              {/* Show pattern selector only when bar_pattern is selected */}
              {params.riserTiming === 'bar_pattern' && (
                <div style={{ marginTop: '15px' }}>
                  <label htmlFor="riser-pattern" style={{ color: 'white', display: 'block', marginBottom: '5px', fontSize: '12px' }}>
                    Pattern (downbeat interval):
                  </label>
                  <select
                    id="riser-pattern"
                    value={params.riserPattern || 4}
                    onChange={(e) => updateParam('riserPattern', parseInt(e.target.value))}
                    style={{
                      width: '100%',
                      padding: '8px',
                      borderRadius: '5px',
                      background: '#1a1a2e',
                      color: 'white',
                      border: '1px solid #444',
                      cursor: 'pointer'
                    }}
                  >
                    <option value="1">1 Bar (every bar)</option>
                    <option value="2">2 Bar (every 2 bars)</option>
                    <option value="4">4 Bar (every 4 bars)</option>
                  </select>
                </div>
              )}
            </>
          )}

          {/* AI Drum Sampler - only show when ai_sampler is selected */}
          {params.drumSubgroup === 'ai_sampler' && (
            <DrumSampler />
          )}
        </>
      )}
    </>
  );
});

MIDITargetSelection.displayName = 'MIDITargetSelection';

// ========== Processing Mode Component ==========
const ProcessingMode = React.memo(({ params, updateParam, uploadedFile, timelineBpm }) => {
  const fileTypeInfo = useMemo(() => {
    if (!uploadedFile) return 'Upload a file to see processing mode';
    return uploadedFile.fileType === 'midi' ? 'MIDI File Detected' : 'Audio File Detected';
  }, [uploadedFile]);

  return (
    <>
      <div className={styles.infoDisplay}>
        {fileTypeInfo}
      </div>

      <div className={styles.checkboxRow}>
        <label>
          <input
            type="checkbox"
            checked={params.midiMode || false}
            onChange={(e) => updateParam('midiMode', e.target.checked)}
          />
          MIDI Mode (enable for MIDI files)
        </label>
      </div>

      <div className={styles.checkboxRow}>
        <label>
          <input
            type="checkbox"
            checked={params.renderAndExtract || false}
            onChange={(e) => updateParam('renderAndExtract', e.target.checked)}
          />
          Render & Extract (use FluidSynth for MIDI)
        </label>
      </div>

      <div className={styles.checkboxRow}>
        <label>
          <input
            type="checkbox"
            checked={params.renderExtractMono || false}
            onChange={(e) => updateParam('renderExtractMono', e.target.checked)}
            disabled={!params.renderAndExtract}
          />
          Mono Mode (extract single voice only)
        </label>
      </div>

      <div className={styles.checkboxRow}>
        <label>
          <input
            type="checkbox"
            checked={params.debugMode || false}
            onChange={(e) => updateParam('debugMode', e.target.checked)}
          />
          Debug Mode (enable trajectory logging)
        </label>
      </div>

      <div className={styles.paramRow}>
        <label>
          <div className={styles.paramLabel}>
            <span>Timeline BPM:</span>
            <span className={styles.paramValue}>{timelineBpm || 120.0} BPM</span>
          </div>
        </label>
      </div>

      <div className={styles.paramRow}>
        <label>
          <div className={styles.paramLabel}>
            <span>Tempo Override:</span>
          </div>
          <input
            type="number"
            className={styles.controlSelect}
            min="60"
            max="200"
            step="0.1"
            value={params.tempoOverride ?? timelineBpm ?? 120}
            onChange={(e) => updateParam('tempoOverride', parseFloat(e.target.value))}
          />
        </label>
      </div>
    </>
  );
});

ProcessingMode.displayName = 'ProcessingMode';

// ========== Generation Parameters Component ==========
const GenerationParameters = React.memo(({ params, updateParam }) => {
  const parameters = useMemo(() => [
    { id: 'seed', label: 'Seed', min: 0, max: 10000, step: 1 },
    { id: 'steps', label: 'Steps', min: 10, max: 100, step: 1 },
    { id: 'adapterScale', label: 'Adapter Scale', min: 0, max: 2, step: 0.1 },
    { id: 'cfgWeight', label: 'CFG Weight', min: 0, max: 10, step: 0.1 },
    { id: 't0', label: 'T0', min: 0, max: 1, step: 0.05 },
    { id: 'noiseLevel', label: 'Noise Level', min: 0, max: 1, step: 0.1 }
  ], []);

  return (
    <>
      {parameters.map(param => (
        <div key={param.id} className={styles.paramRow}>
          <label>
            <div className={styles.paramLabel}>
              <span>{param.label}:</span>
              <span className={styles.paramValue}>{params[param.id] ?? param.min}</span>
            </div>
            <input
              type="range"
              className={styles.controlRange}
              min={param.min}
              max={param.max}
              step={param.step}
              value={params[param.id] ?? param.min}
              onChange={(e) => updateParam(param.id, parseFloat(e.target.value))}
            />
          </label>
        </div>
      ))}

      {/* Temperature/Variance Scaling */}
      <div className={styles.paramRow} style={{ marginTop: '15px' }}>
        <label>
          <div className={styles.paramLabel}>
            <span>Temperature (Variance Scaling):</span>
            <span className={styles.paramValue}>{params.temperature ?? 1.0}</span>
          </div>
          <input
            type="range"
            className={styles.controlRange}
            min={0.1}
            max={2.0}
            step={0.1}
            value={params.temperature ?? 1.0}
            onChange={(e) => updateParam('temperature', parseFloat(e.target.value))}
          />
          <div style={{ fontSize: '11px', color: '#888', marginTop: '4px', lineHeight: '1.3' }}>
            Lower = less random/more focused, Higher = more diverse (default: 1.0)
          </div>
        </label>
      </div>

      {/* Stochastic Sampling (Eta) */}
      <div className={styles.paramRow}>
        <label>
          <div className={styles.paramLabel}>
            <span>Stochastic Sampling (Eta):</span>
            <span className={styles.paramValue}>{params.eta ?? 0.5}</span>
          </div>
          <input
            type="range"
            className={styles.controlRange}
            min={0.0}
            max={1.0}
            step={0.05}
            value={params.eta ?? 0.5}
            onChange={(e) => updateParam('eta', parseFloat(e.target.value))}
          />
          <div style={{ fontSize: '11px', color: '#888', marginTop: '4px', lineHeight: '1.3' }}>
            Controls randomness during sampling: 0 = deterministic, 1 = fully stochastic (default: 0.5)
          </div>
        </label>
      </div>
    </>
  );
});

GenerationParameters.displayName = 'GenerationParameters';

// ========== Conditioning Stream Gains Component ==========
const ConditioningStreamGains = React.memo(({ params, updateParam }) => {
  const parameters = useMemo(() => [
    { id: 'instrumentStrength', label: 'Instrument Strength', min: 0, max: 5, step: 0.1, default: 1 },
    { id: 'instBoost', label: 'Instrument Token Boost', min: 1, max: 5, step: 0.1, default: 2.5 },
    { id: 'pianoRollGain', label: 'Piano Roll Gain', min: 0, max: 4, step: 0.1, default: 1 },
    { id: 'ampGain', label: 'Amplitude Gain', min: 0, max: 2, step: 0.1, default: 1 },
    { id: 'rframeGain', label: 'RFrame Gain', min: 0, max: 2, step: 0.1, default: 1 },
    { id: 'rbendGain', label: 'RBend Gain', min: 0, max: 2, step: 0.1, default: 1 },
    { id: 'encodecGain', label: 'EnCodec Gain', min: 0, max: 2, step: 0.1, default: 1 }
  ], []);

  return (
    <>
      {parameters.map(param => (
        <div key={param.id} className={styles.paramRow}>
          <label>
            <div className={styles.paramLabel}>
              <span>{param.label}:</span>
              <span className={styles.paramValue}>{params[param.id] ?? param.default}</span>
            </div>
            <input
              type="range"
              className={styles.controlRange}
              min={param.min}
              max={param.max}
              step={param.step}
              value={params[param.id] ?? param.default}
              onChange={(e) => updateParam(param.id, parseFloat(e.target.value))}
            />
          </label>
        </div>
      ))}
    </>
  );
});

ConditioningStreamGains.displayName = 'ConditioningStreamGains';

// ========== Extraction Formats Component ==========
const ExtractionFormats = React.memo(({ params, updateParam }) => {
  // Available extraction formats (latent is excluded - determined by noise level)
  const formats = useMemo(() => [
    { id: 'midi', label: 'MIDI (BasicPitch)', description: 'Extract MIDI notes from audio' },
    { id: 'amp', label: 'Amplitude Envelope', description: 'Extract volume envelope' },
    { id: 'rframe', label: 'Voicing Frame', description: 'Extract voiced/unvoiced regions' },
    { id: 'rbend', label: 'Pitch Bend', description: 'Extract pitch variations' },
    { id: 'encodec', label: 'EnCodec Tokens', description: 'Extract audio codec tokens' }
  ], []);

  // Initialize extractFormats if not set (all enabled by default)
  const extractFormats = params.extractFormats || ['midi', 'amp', 'rframe', 'rbend', 'encodec'];

  const toggleFormat = useCallback((formatId) => {
    const currentFormats = params.extractFormats || ['midi', 'amp', 'rframe', 'rbend', 'encodec'];
    const newFormats = currentFormats.includes(formatId)
      ? currentFormats.filter(f => f !== formatId)
      : [...currentFormats, formatId];
    updateParam('extractFormats', newFormats);
  }, [params.extractFormats, updateParam]);

  const isChecked = useCallback((formatId) => {
    return extractFormats.includes(formatId);
  }, [extractFormats]);

  return (
    <div className={styles.extractionFormats}>
      <div className={styles.extractionInfo}>
        <i className="fa-solid fa-info-circle"></i>
        <span>Select which audio features to extract. Unselected formats will use synthetic/dummy streams for faster processing.</span>
      </div>
      {formats.map(format => (
        <div key={format.id} className={styles.checkboxRow}>
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={isChecked(format.id)}
              onChange={() => toggleFormat(format.id)}
              className={styles.checkbox}
            />
            <span className={styles.formatLabel}>{format.label}</span>
            <span className={styles.formatDescription}>{format.description}</span>
          </label>
        </div>
      ))}
      <div className={styles.extractionNote}>
        <i className="fa-solid fa-lightbulb"></i>
        <span>Note: Latent extraction is controlled by Noise Level (extracted when &lt; 1.0)</span>
      </div>
    </div>
  );
});

ExtractionFormats.displayName = 'ExtractionFormats';

// ========== MIDI Fidelity Enhancements Component ==========
const MidiFidelityEnhancements = React.memo(({ params, updateParam }) => {
  const parameters = useMemo(() => [
    { id: 'pitchFidelityBoost', label: 'Pitch Fidelity Boost', min: 0, max: 2, step: 0.1, default: 1 },
    { id: 'onsetGuidanceBoost', label: 'Onset Guidance Boost', min: 0, max: 5, step: 0.1, default: 0 },
    { id: 'pitchSnapStrength', label: 'Pitch Snap Strength', min: 0, max: 1, step: 0.05, default: 0.5 }
  ], []);

  return (
    <>
      {parameters.map(param => (
        <div key={param.id} className={styles.paramRow}>
          <label>
            <div className={styles.paramLabel}>
              <span>{param.label}:</span>
              <span className={styles.paramValue}>{params[param.id] ?? param.default}</span>
            </div>
            <input
              type="range"
              className={styles.controlRange}
              min={param.min}
              max={param.max}
              step={param.step}
              value={params[param.id] ?? param.default}
              onChange={(e) => updateParam(param.id, parseFloat(e.target.value))}
            />
          </label>
        </div>
      ))}
    </>
  );
});

MidiFidelityEnhancements.displayName = 'MidiFidelityEnhancements';

// ========== Sample Recreation Features Component ==========
const SampleRecreationFeatures = React.memo(({ params, updateParam }) => {
  return (
    <>
      <div className={styles.checkboxRow}>
        <label>
          <input
            type="checkbox"
            checked={params.useTimeVaryingNoise || false}
            onChange={(e) => updateParam('useTimeVaryingNoise', e.target.checked)}
          />
          Time-Varying Noise (Preserve Attacks)
        </label>
      </div>

      <div className={styles.paramRow}>
        <label>
          <div className={styles.paramLabel}>
            <span>Onset Preservation:</span>
            <span className={styles.paramValue}>{params.onsetPreservation ?? 0.7}</span>
          </div>
          <input
            type="range"
            className={styles.controlRange}
            min="0"
            max="1"
            step="0.05"
            value={params.onsetPreservation ?? 0.7}
            onChange={(e) => updateParam('onsetPreservation', parseFloat(e.target.value))}
          />
        </label>
      </div>

      <div className={styles.checkboxRow}>
        <label>
          <input
            type="checkbox"
            checked={params.useMultiresolutionMixing ?? true}
            onChange={(e) => updateParam('useMultiresolutionMixing', e.target.checked)}
          />
          Multi-Resolution Mixing (Preserve Low Freqs)
        </label>
      </div>

      <div className={styles.checkboxRow}>
        <label>
          <input
            type="checkbox"
            checked={params.useOnsetWeightedEncodec || false}
            onChange={(e) => updateParam('useOnsetWeightedEncodec', e.target.checked)}
          />
          Onset-Weighted Encodec (Preserve Attack Timbre)
        </label>
      </div>

      <div className={styles.paramRow}>
        <label>
          <div className={styles.paramLabel}>
            <span>Encodec Onset Boost:</span>
            <span className={styles.paramValue}>{params.encodecOnsetBoost ?? 2}</span>
          </div>
          <input
            type="range"
            className={styles.controlRange}
            min="1"
            max="5"
            step="0.1"
            value={params.encodecOnsetBoost ?? 2}
            onChange={(e) => updateParam('encodecOnsetBoost', parseFloat(e.target.value))}
          />
        </label>
      </div>
    </>
  );
});

SampleRecreationFeatures.displayName = 'SampleRecreationFeatures';

// ========== Audio Decoder Options Component ==========
const AudioDecoderOptions = React.memo(({ params, updateParam }) => {
  return (
    <>
      <div className={styles.checkboxRow}>
        <label>
          <input
            type="checkbox"
            checked={params.useOverlapDecoder ?? true}
            onChange={(e) => updateParam('useOverlapDecoder', e.target.checked)}
          />
          Use Overlap Decoder (Better Quality)
        </label>
      </div>

      <div className={styles.checkboxRow}>
        <label>
          <input
            type="checkbox"
            checked={params.monophonicMode ?? true}
            onChange={(e) => updateParam('monophonicMode', e.target.checked)}
          />
          🎵 Monophonic Mode (separate voices)
        </label>
      </div>

      <div className={styles.checkboxRow}>
        <label>
          <input
            type="checkbox"
            checked={params.arrangeMode || false}
            onChange={(e) => updateParam('arrangeMode', e.target.checked)}
          />
          🎼 Arrange Mode (auto-assign instruments by range)
        </label>
      </div>

      <div className={styles.checkboxRow}>
        <label>
          <input
            type="checkbox"
            checked={params.fattenMode ?? true}
            onChange={(e) => updateParam('fattenMode', e.target.checked)}
          />
          🎚️ Fatten Mode
        </label>
        <select
          className={styles.controlSelect}
          style={{ marginLeft: '10px', display: 'inline-block', width: 'auto' }}
          value={params.fattenType || 'real'}
          onChange={(e) => updateParam('fattenType', e.target.value)}
        >
          <option value="real">Real (generate octave-up)</option>
          <option value="fake">Fake (pitch shift output)</option>
        </select>
      </div>

      <div className={styles.checkboxRow}>
        <label>
          <input
            type="checkbox"
            checked={params.bestOfNMode || false}
            onChange={(e) => updateParam('bestOfNMode', e.target.checked)}
          />
          🎲 Best-of-N Sampling (requires audio input)
        </label>
        <input
          type="number"
          className={styles.controlSelect}
          style={{ marginLeft: '10px', width: '60px', display: 'inline-block' }}
          value={params.nCandidates || 12}
          min="3"
          max="24"
          step="1"
          onChange={(e) => updateParam('nCandidates', parseInt(e.target.value))}
          title="Number of candidates to generate and rank"
        />
      </div>

      <div className={styles.checkboxRow}>
        <label>
          <input
            type="checkbox"
            checked={params.selfConsistencyMode || false}
            onChange={(e) => updateParam('selfConsistencyMode', e.target.checked)}
          />
          🔄 Self-Consistency Ensembling
        </label>
        <input
          type="number"
          className={styles.controlSelect}
          style={{ marginLeft: '10px', width: '60px', display: 'inline-block' }}
          value={params.consistencySamples || 3}
          min="2"
          max="5"
          step="1"
          onChange={(e) => updateParam('consistencySamples', parseInt(e.target.value))}
          title="Number of ensemble samples"
        />
      </div>
    </>
  );
});

AudioDecoderOptions.displayName = 'AudioDecoderOptions';

// ========== Output Configuration Component ==========
const OutputConfiguration = React.memo(({ params, updateParam }) => {
  return (
    <>
      <div className={styles.infoDisplay}>
        Generated audio will appear below after processing
      </div>

      <div className={styles.checkboxRow}>
        <label>
          <input
            type="checkbox"
            checked={params.enableVoiceSeparation || false}
            onChange={(e) => updateParam('enableVoiceSeparation', e.target.checked)}
          />
          Enable Individual Voice Outputs
        </label>
      </div>

      <div className={styles.checkboxRow}>
        <label>
          <input
            type="checkbox"
            checked={params.enableMidiExport ?? true}
            onChange={(e) => updateParam('enableMidiExport', e.target.checked)}
          />
          Enable MIDI Export
        </label>
      </div>

      <div className={styles.checkboxRow}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span>Fast Mode:</span>
          <select
            className={styles.controlSelect}
            style={{ flex: 1 }}
            value={params.fastModeVariant || ''}
            onChange={(e) => updateParam('fastModeVariant', e.target.value)}
          >
            <option value="">🐌 Normal (FluidSynth)</option>
            <option value="zero">⚡ Fast Zero (Piano Roll Only)</option>
            <option value="encodec">🔊 Fast Encodec (Encodec Tokens Only)</option>
            <option value="synthetic">🚀 Fast Synthetic (Synthetic Signals)</option>
          </select>
        </label>
      </div>
    </>
  );
});

OutputConfiguration.displayName = 'OutputConfiguration';

// ========== Tape Speed Controls Component ==========
const TapeSpeedControls = React.memo(({ params, updateParam }) => {
  return (
    <>
      {/* Tape Speed Slider */}
      <div className={styles.paramGroup}>
        <label>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '13px', fontWeight: '500' }}>
              <i className="fa-solid fa-gauge"></i> Tape Speed
            </span>
            <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#667eea' }}>
              {(params.tapeSpeed ?? 1.0).toFixed(2)}x
            </span>
          </div>
          <input
            type="range"
            min="0.5"
            max="1.0"
            step="0.05"
            value={params.tapeSpeed ?? 1.0}
            onChange={(e) => updateParam('tapeSpeed', parseFloat(e.target.value))}
            className={styles.slider}
          />
          <div style={{ fontSize: '11px', color: '#888', marginTop: '6px', lineHeight: '1.4' }}>
            Slows down output audio (0.5x = half speed, 1.0x = normal speed)
          </div>
        </label>
      </div>

      {/* Slowdown Method Radio Buttons */}
      <div className={styles.paramGroup} style={{ marginTop: '15px' }}>
        <label style={{ fontSize: '13px', fontWeight: '500', display: 'block', marginBottom: '10px' }}>
          <i className="fa-solid fa-sliders"></i> Slowdown Method
        </label>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontSize: '12px' }}>
            <input
              type="radio"
              name="slowdown-method"
              value="tape"
              checked={params.slowdownMethod === 'tape'}
              onChange={(e) => updateParam('slowdownMethod', e.target.value)}
              style={{ marginRight: '8px', cursor: 'pointer' }}
            />
            <span>Tape - Changes pitch when slowed down</span>
          </label>

          <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontSize: '12px' }}>
            <input
              type="radio"
              name="slowdown-method"
              value="stretch"
              checked={params.slowdownMethod === 'stretch'}
              onChange={(e) => updateParam('slowdownMethod', e.target.value)}
              style={{ marginRight: '8px', cursor: 'pointer' }}
            />
            <span>Stretch - Maintains pitch when slowed down</span>
          </label>
        </div>

        <div style={{ fontSize: '11px', color: '#888', marginTop: '8px', lineHeight: '1.4' }}>
          Choose how audio is slowed: tape-style (lower pitch) or time-stretch (same pitch)
        </div>
      </div>

      {/* Upsample Mode Checkbox */}
      <div className={styles.paramGroup} style={{ marginTop: '15px' }}>
        <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontSize: '13px', fontWeight: '500' }}>
          <input
            type="checkbox"
            checked={params.upsampleMode ?? false}
            onChange={(e) => updateParam('upsampleMode', e.target.checked)}
            style={{ marginRight: '8px', cursor: 'pointer', width: '16px', height: '16px' }}
          />
          <i className="fa-solid fa-arrow-up-right-dots" style={{ marginRight: '6px' }}></i> Upsample Mode
        </label>
        <div style={{ fontSize: '11px', color: '#888', marginTop: '6px', lineHeight: '1.4' }}>
          After speed restoration, refine audio with additional diffusion steps for higher quality
        </div>
      </div>

      {/* Upsample Parameters (only shown when upsample mode is enabled) */}
      {params.upsampleMode && (
        <>
          {/* Upsample Noise Level Slider */}
          <div className={styles.paramGroup} style={{ marginTop: '15px', marginLeft: '24px' }}>
            <label>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ fontSize: '12px', fontWeight: '500', color: '#667eea' }}>
                  <i className="fa-solid fa-volume-low"></i> Upsample Noise Level
                </span>
                <span style={{ fontSize: '13px', fontWeight: 'bold', color: '#667eea' }}>
                  {(params.upsampleNoiseLevel ?? 0.3).toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={params.upsampleNoiseLevel ?? 0.3}
                onChange={(e) => updateParam('upsampleNoiseLevel', parseFloat(e.target.value))}
                className={styles.slider}
              />
              <div style={{ fontSize: '10px', color: '#888', marginTop: '4px', lineHeight: '1.3' }}>
                Amount of noise to add for refinement (0 = no change, 1 = full regeneration)
              </div>
            </label>
          </div>

          {/* Upsample Steps Slider */}
          <div className={styles.paramGroup} style={{ marginTop: '12px', marginLeft: '24px' }}>
            <label>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ fontSize: '12px', fontWeight: '500', color: '#667eea' }}>
                  <i className="fa-solid fa-stairs"></i> Upsample Steps
                </span>
                <span style={{ fontSize: '13px', fontWeight: 'bold', color: '#667eea' }}>
                  {params.upsampleSteps ?? 20}
                </span>
              </div>
              <input
                type="range"
                min="5"
                max="50"
                step="5"
                value={params.upsampleSteps ?? 20}
                onChange={(e) => updateParam('upsampleSteps', parseInt(e.target.value))}
                className={styles.slider}
              />
              <div style={{ fontSize: '10px', color: '#888', marginTop: '4px', lineHeight: '1.3' }}>
                Number of diffusion steps for upsampling (more steps = higher quality, slower)
              </div>
            </label>
          </div>
        </>
      )}
    </>
  );
});

TapeSpeedControls.displayName = 'TapeSpeedControls';

// ========== dø Vox Controls Component ==========
const DoVoxControls = React.memo(({ params, updateParam }) => {
  const { state } = useApp();
  const [advancedExpanded, setAdvancedExpanded] = React.useState(false);
  const [isTranslating, setIsTranslating] = React.useState(false);
  const [selectedLanguage, setSelectedLanguage] = React.useState('english');

  const languages = [
    { value: 'english', label: 'English' },
    { value: 'spanish', label: 'Spanish' },
    { value: 'french', label: 'French' },
    { value: 'german', label: 'German' },
    { value: 'mandarin', label: 'Mandarin' },
    { value: 'japanese', label: 'Japanese' }
  ];

  const buildDAWContext = () => {
    return {
      bpm: state.bpm || 120,
      key: state.generationParams?.key || 'C',
      isBPMMode: state.isBPMMode || false,
      totalDuration: state.totalDuration || 30,
      trackCount: state.buses?.reduce((total, bus) => total + bus.tracks.length, 0) || 0,
      buses: state.buses?.map(bus => ({
        id: bus.id,
        name: bus.name,
        type: bus.type,
        trackCount: bus.tracks.length
      })) || []
    };
  };

  // Helper function to count syllables in a word
  const countSyllables = (word) => {
    word = word.toLowerCase().trim();
    if (word.length === 0) return 0;
    if (word.length <= 3) return 1;

    // Remove non-alphabetic characters except hyphens and apostrophes
    word = word.replace(/[^a-záéíóúñü'-]/gi, '');

    // Count vowel groups (including Spanish vowels)
    word = word.replace(/(?:[^laeiouyáéíóúü]es|ed|[^laeiouyáéíóúü]e)$/i, ''); // remove silent e
    word = word.replace(/^y/i, ''); // y at start is consonant
    const syllables = word.match(/[aeiouyáéíóúü]{1,2}/gi);

    return syllables ? Math.max(syllables.length, 1) : 1;
  };

  // Count syllables in a full line of text
  const countLineSyllables = (line) => {
    const cleanLine = line.trim();
    if (cleanLine === '') return 0;

    // Remove punctuation and split into words
    const words = cleanLine.replace(/[^\w\s'-áéíóúñü]/gi, '').split(/\s+/);
    const totalSyllables = words.reduce((count, word) => {
      if (word.length === 0) return count;
      return count + countSyllables(word);
    }, 0);

    return totalSyllables;
  };

  const handleLanguageChange = async (languageValue) => {
    setSelectedLanguage(languageValue);

    // Only translate if there are lyrics
    if (!params.aceLyrics || params.aceLyrics.trim() === '') {
      return;
    }

    setIsTranslating(true);

    try {
      // Import chat API
      const { sendChatMessage } = await import('../../services/chatAPI');
      const { SYSTEM_PROMPT } = await import('../ChatWindow/systemPrompt');

      const language = languages.find(l => l.value === languageValue)?.label || languageValue;

      // Count syllables for each line
      const lines = params.aceLyrics.split('\n');
      const linesWithCounts = lines.map(line => {
        const cleanLine = line.trim();
        if (cleanLine === '') return '';

        // Count syllables in the line
        const words = cleanLine.replace(/[^\w\s'-]/g, '').split(/\s+/);
        const syllableCount = words.reduce((count, word) => {
          if (word.length === 0) return count;
          return count + countSyllables(word);
        }, 0);

        return `[${syllableCount} syllables] ${cleanLine}`;
      }).join('\n');

      // Prepare chat payload for lyric translation with explicit syllable counts
      const payload = {
        system_prompt: SYSTEM_PROMPT,
        daw_context: buildDAWContext(),
        message: `TASK: LYRIC CHANGE\nPAYLOAD:\nLanguage: ${language}\nOriginal Lyrics (with syllable counts per line):\n${linesWithCounts}`,
        conversation_history: []
      };

      console.log(`🌐 Translating lyrics to ${language}...`);
      console.log(`📝 Syllable-counted input:\n${linesWithCounts}`);
      const response = await sendChatMessage(payload);

      console.log('📦 Chat API response:', response);

      // Check different possible response structures
      let translatedLyrics = null;
      if (response?.response) {
        translatedLyrics = response.response;
      } else if (response?.message) {
        translatedLyrics = response.message;
      } else if (response?.content) {
        translatedLyrics = response.content;
      } else if (typeof response === 'string') {
        translatedLyrics = response;
      }

      if (translatedLyrics) {
        // Parse original and translated lines
        const originalLines = lines;
        const translatedLines = translatedLyrics.split('\n');

        // Strip out syllable count numbers (X) from the end of each line
        const cleanedLines = translatedLines.map(line => line.replace(/\s*\(\d+\)\s*$/, '').trim());

        // Verify syllable counts
        console.group('🔢 Syllable Count Analysis');
        let perfectMatches = 0;
        let closeMatches = 0;
        const verification = [];

        originalLines.forEach((origLine, index) => {
          const cleanOrigLine = origLine.trim();
          if (cleanOrigLine === '') {
            verification.push({ line: index + 1, status: 'empty' });
            return;
          }

          const translatedLine = cleanedLines[index] || '';

          // Get required syllable count from original
          const origWords = cleanOrigLine.replace(/[^\w\s'-áéíóúñü]/gi, '').split(/\s+/);
          const requiredSyllables = origWords.reduce((count, word) => {
            if (word.length === 0) return count;
            return count + countSyllables(word);
          }, 0);

          // Count actual syllables in translation
          const actualSyllables = countLineSyllables(translatedLine);

          const diff = actualSyllables - requiredSyllables;
          const matches = diff === 0;
          const closeEnough = Math.abs(diff) <= 1;

          if (matches) perfectMatches++;
          if (closeEnough) closeMatches++;

          let status = '✓';
          let suggestion = '';
          if (diff > 0) {
            status = '✗';
            suggestion = `(${diff} too many - try shorter words or remove filler)`;
          } else if (diff < 0) {
            status = '✗';
            suggestion = `(${Math.abs(diff)} too few - try longer words or add filler)`;
          }

          verification.push({
            line: index + 1,
            original: cleanOrigLine,
            translation: translatedLine,
            required: requiredSyllables,
            actual: actualSyllables,
            diff,
            matches,
            closeEnough,
            status,
            suggestion
          });

          console.log(
            `Line ${index + 1}: ${status} Need ${requiredSyllables}, Got ${actualSyllables} ${diff !== 0 ? suggestion : '✓ Perfect!'}\n` +
            `  "${cleanOrigLine}"\n` +
            `  "${translatedLine}"`
          );
        });

        console.log('\n📊 Summary:');
        console.log(`Perfect matches (exact syllable count): ${perfectMatches}/${verification.length}`);
        console.log(`Close matches (±1 syllable): ${closeMatches}/${verification.length}`);
        console.log(`\n💡 Tip: Syllable counting is approximate. Manual adjustment may be needed for singing.`);
        console.groupEnd();

        const cleanedLyrics = cleanedLines.join('\n');

        // Update lyrics with ONLY the translated text (no syllable counts)
        updateParam('aceLyrics', cleanedLyrics.trim());

        console.log(`✅ Lyrics translated to ${language} (${perfectMatches}/${verification.length} exact matches)`);
      } else {
        console.warn('⚠️ Could not find translated lyrics in response:', response);
        alert('Translation received but could not extract lyrics. Check console for details.');
      }
    } catch (error) {
      console.error('❌ Failed to translate lyrics:', error);
      alert(`Failed to translate lyrics: ${error.message}`);
    } finally {
      setIsTranslating(false);
    }
  };

  return (
    <>
      {/* Prompt Input */}
      <div className={styles.paramRow}>
        <label>
          <div className={styles.paramLabel}>
            <span>Prompt:</span>
          </div>
          <input
            type="text"
            className={styles.controlSelect}
            placeholder="Describe the music style..."
            value={params.acePrompt || ''}
            onChange={(e) => updateParam('acePrompt', e.target.value)}
            style={{ width: '100%' }}
          />
        </label>
      </div>

      {/* Lyrics Input */}
      <div className={styles.paramRow}>
        <label>
          <div className={styles.paramLabel}>
            <span>Lyrics:</span>
          </div>
          <textarea
            className={styles.controlSelect}
            placeholder="Enter lyrics..."
            value={params.aceLyrics || ''}
            onChange={(e) => updateParam('aceLyrics', e.target.value)}
            style={{ width: '100%', minHeight: '80px', resize: 'vertical', fontFamily: 'inherit' }}
            disabled={isTranslating}
          />
        </label>
      </div>

      {/* Language Selector */}
      <div className={styles.paramRow}>
        <label>
          <div className={styles.paramLabel}>
            <span>Language:</span>
            {isTranslating && <span className={styles.paramValue}>Translating...</span>}
          </div>
          <CustomDropdown
            value={selectedLanguage}
            onChange={handleLanguageChange}
            options={languages}
          />
        </label>
      </div>

      {/* Advanced Settings */}
      <div className={styles.advancedSettings}>
        <div
          className={styles.advancedHeader}
          onClick={() => setAdvancedExpanded(!advancedExpanded)}
        >
          <i className={`fa-solid fa-chevron-${advancedExpanded ? 'down' : 'right'}`}></i>
          <span>Advanced Settings</span>
        </div>

        {advancedExpanded && (
          <div className={styles.advancedContent}>
            {/* Key Input */}
            <div className={styles.paramRow}>
              <label>
                <div className={styles.paramLabel}>
                  <span>Key:</span>
                </div>
                <select
                  className={styles.controlSelect}
                  value={params.aceKey || 'C'}
                  onChange={(e) => updateParam('aceKey', e.target.value)}
                >
                  {['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'].map(key => (
                    <option key={key} value={key}>{key}</option>
                  ))}
                </select>
              </label>
            </div>

            {/* Steps Input */}
            <div className={styles.paramRow}>
              <label>
                <div className={styles.paramLabel}>
                  <span>Steps:</span>
                  <span className={styles.paramValue}>{params.aceSteps ?? 100}</span>
                </div>
                <input
                  type="number"
                  className={styles.controlSelect}
                  min="10"
                  max="200"
                  step="10"
                  value={params.aceSteps ?? 100}
                  onChange={(e) => updateParam('aceSteps', parseInt(e.target.value))}
                />
              </label>
            </div>

            {/* Seed Input */}
            <div className={styles.paramRow}>
              <label>
                <div className={styles.paramLabel}>
                  <span>Seed:</span>
                  <span className={styles.paramValue}>{params.seed ?? 0}</span>
                </div>
                <input
                  type="number"
                  className={styles.controlSelect}
                  min="0"
                  max="999999"
                  step="1"
                  value={params.seed ?? 0}
                  onChange={(e) => updateParam('seed', parseInt(e.target.value))}
                  placeholder="Random seed (0 for random)"
                />
              </label>
            </div>

            {/* Detailed Mode Checkbox */}
            <div className={styles.checkboxRow}>
              <label>
                <input
                  type="checkbox"
                  checked={params.aceDetailedMode ?? false}
                  onChange={(e) => updateParam('aceDetailedMode', e.target.checked)}
                />
                <span>Detailed Mode</span>
              </label>
            </div>
          </div>
        )}
      </div>
    </>
  );
});

DoVoxControls.displayName = 'DoVoxControls';

// ========== File Upload Component ==========
const FileUpload = React.memo(({ uploadedFile, onFileSelect, onClearFile, isGenerating }) => {
  const [extractMidiFromRecording, setExtractMidiFromRecording] = React.useState(true);
  const [detailedMode, setDetailedMode] = React.useState(false);
  const [convertToSine, setConvertToSine] = React.useState(false);
  const [octaveUp, setOctaveUp] = React.useState(false);
  const [glideTime, setGlideTime] = React.useState(0.05); // Default 50ms glide
  const [recordingSettingsExpanded, setRecordingSettingsExpanded] = React.useState(false);
  const { isRecording, recordingDuration, isProcessing, startRecording, stopRecording, cancelRecording } = useAudioRecorder({
    extractMidi: extractMidiFromRecording,
    detailedMode: detailedMode,
    convertToSine: convertToSine,
    octaveUp: octaveUp,
    glideTime: glideTime
  });

  const handleFileChange = useCallback((e) => {
    const file = e.target.files[0];
    if (file) onFileSelect(file);
  }, [onFileSelect]);

  const handleRecordClick = useCallback(async () => {
    if (isRecording) {
      // Stop recording and upload
      const recordedFile = await stopRecording();
      if (recordedFile) {
        onFileSelect(recordedFile);
      }
    } else {
      // Start recording
      await startRecording();
    }
  }, [isRecording, stopRecording, startRecording, onFileSelect]);

  const handleCancelRecording = useCallback(() => {
    cancelRecording();
  }, [cancelRecording]);

  const formatDuration = useCallback((seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }, []);

  return (
    <div className={styles.uploadContainer}>
      {uploadedFile?.file ? (
        <div className={styles.fileInfo}>
          <div className={styles.fileDetails}>
            <i className={`fa-solid ${uploadedFile.fileType === 'midi' ? 'fa-file-audio' : 'fa-file-waveform'}`}></i>
            <div>
              <div className={styles.fileName}>{uploadedFile.file.name}</div>
              <div className={styles.fileType}>{uploadedFile.fileType?.toUpperCase()} File</div>
            </div>
          </div>
          <button
            className={styles.clearBtn}
            onClick={onClearFile}
            disabled={isGenerating}
            title="Clear conditioning"
          >
            <i className="fa-solid fa-xmark"></i>
          </button>
        </div>
      ) : (
        <>
          <div className={styles.uploadButtons}>
            <div className={styles.uploadLabelWrapper}>
              <label htmlFor="audio-conditioning-input" className={`${styles.uploadLabel} ${isGenerating || isRecording ? styles.disabled : ''}`}>
                <i className="fa-solid fa-upload"></i>
              </label>
              <span className={styles.uploadLabelText}>Upload Audio<br/>or MIDI</span>
            </div>
            <div className={styles.recordBtnWrapper}>
              <button
                className={`${styles.recordBtn} ${isRecording ? styles.recording : ''} ${isGenerating ? styles.disabled : ''}`}
                onClick={handleRecordClick}
                disabled={isGenerating}
                title={isRecording ? "Stop recording" : "Record from microphone"}
              >
                <i className={`fa-solid ${isRecording ? 'fa-stop' : 'fa-microphone'}`}></i>
              </button>
              <span className={styles.recordBtnText}>{isRecording ? 'Stop' : 'Record'}</span>
            </div>
            {!isRecording && !isProcessing && (
              <div className={styles.recordBtnWrapper}>
                <button
                  className={`${styles.recordBtn} ${recordingSettingsExpanded ? styles.active : ''}`}
                  onClick={() => setRecordingSettingsExpanded(!recordingSettingsExpanded)}
                  disabled={isGenerating}
                  title="Recording settings"
                >
                  <i className="fa-solid fa-gear"></i>
                </button>
                <span className={styles.recordBtnText}>Settings</span>
              </div>
            )}
          </div>
          {!isRecording && !isProcessing && recordingSettingsExpanded && (
            <div className={styles.recordingSettings}>
              <div className={styles.checkboxRow}>
                <label>
                  <input
                    type="checkbox"
                    checked={extractMidiFromRecording}
                    onChange={(e) => setExtractMidiFromRecording(e.target.checked)}
                    disabled={isGenerating}
                  />
                  Convert recording to MIDI track
                </label>
              </div>
              {extractMidiFromRecording && (
                <div className={styles.checkboxRow}>
                  <label>
                    <input
                      type="checkbox"
                      checked={detailedMode}
                      onChange={(e) => setDetailedMode(e.target.checked)}
                      disabled={isGenerating}
                    />
                    Detailed mode (slower, better accuracy)
                  </label>
                </div>
              )}
              <div className={styles.checkboxRow}>
                <label>
                  <input
                    type="checkbox"
                    checked={convertToSine}
                    onChange={(e) => setConvertToSine(e.target.checked)}
                    disabled={isGenerating}
                  />
                  Convert to sine wave
                </label>
              </div>
              {convertToSine && (
                <>
                  <div className={styles.checkboxRow}>
                    <label>
                      <input
                        type="checkbox"
                        checked={octaveUp}
                        onChange={(e) => setOctaveUp(e.target.checked)}
                        disabled={isGenerating}
                      />
                      Octave up
                    </label>
                  </div>
                  <div className={styles.sliderRow}>
                    <label>
                      Glide: {(glideTime * 1000).toFixed(0)}ms
                      <input
                        type="range"
                        min="0"
                        max="200"
                        step="5"
                        value={glideTime * 1000}
                        onChange={(e) => setGlideTime(parseFloat(e.target.value) / 1000)}
                        disabled={isGenerating}
                        style={{ width: '100%', marginTop: '4px' }}
                      />
                    </label>
                  </div>
                </>
              )}
            </div>
          )}
          {isRecording && (
            <div className={styles.recordingInfo}>
              <div className={styles.recordingDot}></div>
              <span>Recording: {formatDuration(recordingDuration)}</span>
              <button
                className={styles.clearBtn}
                onClick={handleCancelRecording}
                title="Cancel recording"
                style={{ marginLeft: 'auto' }}
              >
                <i className="fa-solid fa-xmark"></i>
              </button>
            </div>
          )}
          {isProcessing && (
            <div className={styles.recordingInfo}>
              <i className="fa-solid fa-spinner fa-spin"></i>
              <span>Extracting MIDI...</span>
            </div>
          )}
        </>
      )}
      <input
        type="file"
        id="audio-conditioning-input"
        accept="audio/*,.mid,.midi"
        onChange={handleFileChange}
        disabled={isGenerating || isRecording}
        style={{ display: 'none' }}
      />
    </div>
  );
});

FileUpload.displayName = 'FileUpload';

// ========== Main Optimized Component ==========
const GenerationPanelOptimized = React.memo(() => {
  const { state, dispatch } = useApp();

  // Local state for generation mode dropdown
  const [generationMode, setGenerationMode] = useState('do-v1');

  // Local state for folder tabs
  const [activeTab, setActiveTab] = useState('instruments');

  // Use generation hook
  const {
    isGenerating,
    generationProgress,
    generationError,
    elapsedTime,
    generate,
    setIsGenerating,
    setGenerationProgress,
    setGenerationError,
    startTimer,
    stopTimer
  } = useGeneration();

  // Auto-enable/disable MIDI mode when generation mode changes
  useEffect(() => {
    const shouldEnableMidi = generationMode === 'mido';
    if (state.generationParams.midiMode !== shouldEnableMidi) {
      dispatch({
        type: 'UPDATE_GENERATION_PARAMS',
        payload: { midiMode: shouldEnableMidi }
      });
    }
  }, [generationMode, state.generationParams.midiMode, dispatch]);

  const [collapsedSections, setCollapsedSections] = useState({
    conditioning: false,
    instrument: false,
    aceStep: false,
    advanced: true, // Parent section for all advanced parameters
    processing: true,
    generation: true,
    conditioningGains: true,
    extractionFormats: true,
    midiFidelity: true,
    sampleRecreation: true,
    audioDecoder: true,
    output: true,
    tapeSpeed: true
  });
  const [selectedMode, setSelectedMode] = useState('Music');

  // Update generation parameter
  const updateParam = useCallback((param, value) => {
    // Sync t0 and noiseLevel - they should always move together
    if (param === 't0' || param === 'noiseLevel') {
      dispatch({
        type: 'UPDATE_GENERATION_PARAMS',
        payload: {
          t0: value,
          noiseLevel: value
        }
      });
    } else {
      dispatch({
        type: 'UPDATE_GENERATION_PARAMS',
        payload: { [param]: value }
      });
    }
  }, [dispatch]);

  // Toggle section collapse
  const toggleSection = useCallback((section) => {
    setCollapsedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  }, []);

  // Handle file upload
  const handleFileSelect = useCallback(async (file) => {
    const fileType = isMidiFile(file) ? 'midi' : isAudioFile(file) ? 'audio' : 'unknown';

    if (fileType === 'unknown') {
      alert('Unsupported file type. Please upload an audio or MIDI file.');
      return;
    }

    let previewUrl = null;
    if (fileType === 'audio') {
      previewUrl = URL.createObjectURL(file);
    }

    dispatch({
      type: 'SET_UPLOADED_FILE',
      payload: { file, fileType, previewUrl }
    });

    // Auto-enable MIDI mode for MIDI files
    if (fileType === 'midi' && !state.generationParams.midiMode) {
      updateParam('midiMode', true);
    }

    console.log('📁 File uploaded:', {
      name: file.name,
      type: fileType,
      size: (file.size / 1024).toFixed(2) + ' KB'
    });
  }, [dispatch, state.generationParams.midiMode, updateParam]);

  // Handle file clear
  const handleClearFile = useCallback(() => {
    if (state.uploadedFile?.previewUrl) {
      URL.revokeObjectURL(state.uploadedFile.previewUrl);
    }

    dispatch({
      type: 'SET_UPLOADED_FILE',
      payload: { file: null, fileType: null, previewUrl: null }
    });
  }, [dispatch, state.uploadedFile]);

  // Save parameters for current instrument
  const handleSaveParams = useCallback(() => {
    const instrumentKey = state.generationParams.midiMode
      ? `midi_${state.generationParams.midiTarget || 'melody'}`
      : `${state.generationParams.instrumentGroup || 'strings'}_${state.generationParams.instrumentSubgroup || 'violin'}`;

    // Save all generation parameters to localStorage
    const savedParams = {
      ...state.generationParams,
      savedAt: new Date().toISOString()
    };

    localStorage.setItem(`genParams_${instrumentKey}`, JSON.stringify(savedParams));
    console.log(`💾 Saved parameters for ${instrumentKey}:`, savedParams);

    // Show user feedback (optional: you could add a toast notification here)
    alert(`Parameters saved for ${instrumentKey.replace(/_/g, ' ')}`);
  }, [state.generationParams]);

  // Load saved parameters when instrument changes
  useEffect(() => {
    const instrumentKey = state.generationParams.midiMode
      ? `midi_${state.generationParams.midiTarget || 'melody'}`
      : `${state.generationParams.instrumentGroup || 'strings'}_${state.generationParams.instrumentSubgroup || 'violin'}`;

    const savedParamsStr = localStorage.getItem(`genParams_${instrumentKey}`);
    if (savedParamsStr) {
      try {
        const savedParams = JSON.parse(savedParamsStr);
        // Remove metadata before applying
        const { savedAt, ...paramsToApply } = savedParams;

        console.log(`📂 Loading saved parameters for ${instrumentKey}:`, paramsToApply);

        // Apply saved parameters
        dispatch({
          type: 'UPDATE_GENERATION_PARAMS',
          payload: paramsToApply
        });
      } catch (error) {
        console.error('Error loading saved parameters:', error);
      }
    }
  }, [state.generationParams.instrumentGroup, state.generationParams.instrumentSubgroup, state.generationParams.midiTarget, state.generationParams.midiMode, dispatch]);

  // Handle generation
  const handleGenerate = useCallback(async () => {
    console.log('🎵 Starting generation...');

    // Check if dø vox mode is selected
    if (generationMode === 'ace-step') {
      console.log('🎵 dø vox mode detected - calling dø API');

      // Set generating state manually for dø vox
      setIsGenerating(true);
      setGenerationError(null);
      startTimer();

      try {
        // Get uploaded file (if any)
        const inputFile = state.uploadedFile || null;

        // Start dø vox generation with params and input file (uses ace-step backend)
        const startResult = await generationAPI.generateACEStep(state.generationParams, inputFile);
        console.log('🎵 dø generation started:', startResult);

        const taskId = startResult.task_id;
        if (!taskId) {
          throw new Error('No task_id returned from dø generation');
        }

        console.log('⏳ Polling for dø task completion...');

        // Poll until complete with progress updates
        const result = await generationAPI.pollACEStepUntilComplete(
          taskId,
          (progress) => {
            console.log(`  Poll attempt ${progress.attempts + 1}, status: ${progress.status}`);
            setGenerationProgress({
              status: progress.status,
              progress: progress.attempts * 10, // Rough progress indicator
              message: `Processing... (attempt ${progress.attempts + 1})`
            });
          }
        );

        console.log('✅ dø generation complete!', result);

        // Add tracks to DAW
        if (result.file_paths && result.file_paths.length > 0) {
          // Get or create Music bus
          let musicBusId = state.buses.find(b => b.type === 'Music')?.id;
          if (!musicBusId) {
            musicBusId = `music-${Date.now()}`;
            dispatch({
              type: 'CREATE_BUS',
              payload: { id: musicBusId, type: 'Music', name: 'Music 1', expanded: true }
            });
            console.log('🆕 Created Music bus:', musicBusId);
          }

          // Add each track
          result.file_paths.forEach((filePath, index) => {
            const track = {
              id: `ace-step-${Date.now()}-${index}`,
              name: `dø vox ${index + 1}`,
              audioUrl: filePath,
              duration: state.generationParams.duration || 30.0,
              startPosition: 0,
              gain: 1.0,
              isMuted: false,
              isSolo: false,
              cropStart: 0,
              cropEnd: 0
            };

            dispatch({
              type: 'ADD_TRACK',
              payload: { busId: musicBusId, track }
            });
          });

          console.log(`✅ Added ${result.file_paths.length} dø track(s) to Music bus`);
        }

        // Clear generating state on success
        setIsGenerating(false);
        setGenerationProgress(null);
        stopTimer();

      } catch (error) {
        console.error('❌ dø generation error:', error);
        setGenerationError(error.message);
        setIsGenerating(false);
        setGenerationProgress(null);
        stopTimer();
        alert(`dø vox generation failed: ${error.message}`);
      }

      return; // Exit early - don't run normal generation
    }

    // Check if MIDI mode with drums target - use special endpoints
    const midiMode = state.generationParams.midiMode;
    const midiTarget = state.generationParams.midiTarget;
    const drumSubgroup = state.generationParams.drumSubgroup || 'orchestral';

    if (midiMode && midiTarget === 'drums') {
      console.log(`🥁 MIDI mode with drums target detected - generating ${drumSubgroup} drums`);

      try {
        // Calculate scene durations - use video if available, otherwise use default 30s
        let sceneDurations = [];
        if (state.video?.sceneChanges && state.video.sceneChanges.length > 1) {
          // Use video scene changes
          sceneDurations = sceneToDurations(state.video.sceneChanges, state.video.duration || state.totalDuration);
          console.log('📹 Using video scene changes:', sceneDurations);
        } else {
          // No video - use default 30 second duration
          sceneDurations = [30];
          console.log('🎵 No video - using default 30 second duration');
        }

        // Create or get SFX bus for drums/risers
        let sfxBusId = state.buses.find(b => b.type === 'SFX')?.id;
        if (!sfxBusId) {
          sfxBusId = `sfx-${Date.now()}`;
          dispatch({
            type: 'CREATE_BUS',
            payload: { id: sfxBusId, type: 'SFX', name: 'SFX 1', expanded: false }
          });
          console.log('🆕 Created SFX bus (collapsed):', sfxBusId);
        }

        // Call drums endpoint for both orchestral and risers
        let result;
        const pattern = drumSubgroup === 'riser'
          ? (state.generationParams.riserPattern || 4)
          : (state.generationParams.drumPattern || 1);

        if (drumSubgroup === 'riser') {
          const riserTiming = state.generationParams.riserTiming || 'scene_changes';
          console.log(`🎚️ Generating risers with timing mode: ${riserTiming}`);

          result = await generationAPI.generateDrums({
            tempo: (state.generationParams.tempoOverride ?? state.bpm ?? 120),
            pattern: pattern,
            sceneDurations,
            automationData: state.automationWindow.points && state.automationWindow.points.length > 0
              ? { points: state.automationWindow.points }
              : null
          });

          console.log('🎚️ Riser generation result:', result);

          // Handle riser results
          if (result.file_paths && result.start_times && result.start_times.length > 0) {
            const tempo = result.tempo || (state.generationParams.tempoOverride ?? state.bpm ?? 120);
            const sceneTempos = state.video?.sceneTempos || [];
            const sceneChanges = state.video?.sceneChanges || [];
            const hasSceneTempos = sceneTempos.length > 0 && sceneChanges.length > 1;

            console.log(`🎚️ Creating riser tracks with ${result.file_paths.length} samples`);
            console.log(`🎚️ Timing mode: ${riserTiming}`);

            // Calculate riser placement based on timing mode
            const calculateExpectedHitTime = (hitIndex) => {
              // Scene changes mode - place 2s before each scene change
              if (riserTiming === 'scene_changes' && hasSceneTempos) {
                // Skip first scene (start), return time 2s before each subsequent scene change
                if (hitIndex >= sceneChanges.length - 1) {
                  // Beyond available scene changes
                  return null;
                }
                const sceneChangeTime = sceneChanges[hitIndex + 1]; // +1 to skip first scene
                return Math.max(0, sceneChangeTime - 2); // 2s before scene change
              }

              // Bar pattern mode - use bar calculation
              if (!hasSceneTempos) {
                const secondsPerBeat = 60 / tempo;
                const secondsPerBar = secondsPerBeat * 4;
                const secondsPerPattern = secondsPerBar * pattern;
                return hitIndex * secondsPerPattern;
              }

              // Variable BPM - same logic as drums
              let accumulatedBeats = 0;
              let barNumber = 0;
              const targetBarNumber = hitIndex * pattern;

              for (let sceneIdx = 0; sceneIdx < sceneTempos.length; sceneIdx++) {
                const sceneBPM = sceneTempos[sceneIdx];
                const sceneStart = sceneChanges[sceneIdx];
                const sceneEnd = sceneChanges[sceneIdx + 1] || (sceneStart + 30);
                const secondsPerBeat = 60 / sceneBPM;
                const secondsPerBar = secondsPerBeat * 4;

                const beatsIntoFirstBar = accumulatedBeats % 4;
                const beatsUntilNextBar = beatsIntoFirstBar === 0 ? 0 : (4 - beatsIntoFirstBar);
                const firstBarTime = sceneStart + (beatsUntilNextBar * secondsPerBeat);

                let barTime = firstBarTime;
                while (barTime < sceneEnd) {
                  if (barNumber === targetBarNumber) {
                    return barTime;
                  }
                  barNumber++;
                  barTime += secondsPerBar;
                }

                const sceneDuration = sceneEnd - sceneStart;
                accumulatedBeats += sceneDuration / secondsPerBeat;
              }

              const lastBPM = sceneTempos[sceneTempos.length - 1];
              const lastSceneEnd = sceneChanges[sceneChanges.length - 1];
              const secondsPerBeat = 60 / lastBPM;
              const secondsPerBar = secondsPerBeat * 4;
              const barsNeeded = targetBarNumber - barNumber;
              return lastSceneEnd + (barsNeeded * secondsPerBar);
            };

            // Create riser tracks
            result.file_paths.forEach((filePath, index) => {
              const hitTime = calculateExpectedHitTime(index);

              // Skip if no valid time (e.g., beyond available scene changes)
              if (hitTime === null) {
                console.log(`  Riser ${index}: skipped (no scene change available)`);
                return;
              }

              // For scene_changes mode, hitTime is already 2s before scene change
              // For bar_pattern mode, subtract 2s from downbeat time
              const riserStartTime = riserTiming === 'scene_changes' ? hitTime : Math.max(0, hitTime - 2);
              const velocity = result.velocities ? result.velocities[index] : 1.0;

              console.log(`  Riser ${index}: start=${riserStartTime.toFixed(2)}s (mode: ${riserTiming})`);

              const track = {
                id: `riser-${Date.now()}-${index}`,
                name: `Riser ${index + 1}`,
                audioUrl: filePath,
                duration: 2, // Risers are 2 seconds long
                startPosition: riserStartTime,
                gain: velocity,
                isMuted: false,
                isSolo: false,
                cropStart: 0,
                cropEnd: 0
              };

              dispatch({
                type: 'ADD_TRACK',
                payload: { busId: sfxBusId, track }
              });
            });

            console.log(`✅ Added ${result.file_paths.length} risers to SFX bus`);
          }
        } else {
          // orchestral drums
          const drumTiming = state.generationParams.drumTiming || 'scene_changes';
          console.log(`🥁 Generating orchestral drums with timing mode: ${drumTiming}`);

          result = await generationAPI.generateDrums({
            tempo: (state.generationParams.tempoOverride ?? state.bpm ?? 120),
            pattern: pattern,
            sceneDurations,
            automationData: state.automationWindow.points && state.automationWindow.points.length > 0
              ? { points: state.automationWindow.points }
              : null,
            activeSamples: state.generationParams.activeSamples || ['bass_drum']
          });

          console.log('🥁 Drum generation result:', result);

          // Handle drum results
          if (result.file_paths && result.start_times && result.start_times.length > 0) {
            const tempo = result.tempo || (state.generationParams.tempoOverride ?? state.bpm ?? 120);
            // pattern already defined above
            const sceneTempos = state.video?.sceneTempos || [];
            const sceneChanges = state.video?.sceneChanges || [];
            const hasSceneTempos = sceneTempos.length > 0 && sceneChanges.length > 1;

            console.log(`🥁 Creating drum track with ${result.file_paths.length} samples`);
            console.log(`🥁 Sample index: ORCH #${result.sample_number || 'unknown'}`);
            console.log(`🥁 Velocities:`, result.velocities);
            console.log(`🥁 Hit times from backend:`, result.start_times);
            console.log(`🥁 Tempo: ${tempo} BPM (scene tempos: ${hasSceneTempos ? sceneTempos.join(', ') : 'none'})`);
            console.log(`🥁 Timing mode: ${drumTiming}`);

            // Calculate expected hit times based on timing mode
            const calculateExpectedHitTime = (hitIndex) => {
              // Scene changes mode - place at each scene change
              if (drumTiming === 'scene_changes' && hasSceneTempos) {
                // Skip first scene (start), return time at each subsequent scene change
                if (hitIndex >= sceneChanges.length - 1) {
                  // Beyond available scene changes
                  return null;
                }
                return sceneChanges[hitIndex + 1]; // +1 to skip first scene
              }

              // Bar pattern mode - use bar calculation
              if (!hasSceneTempos) {
                // Constant BPM - simple calculation
                const secondsPerBeat = 60 / tempo;
                const secondsPerBar = secondsPerBeat * 4;
                const secondsPerPattern = secondsPerBar * pattern;
                return hitIndex * secondsPerPattern;
              }

              // Variable BPM - walk through scenes maintaining global bar alignment
              // This uses the same logic as Timeline.js for bar placement
              let accumulatedBeats = 0;
              let barNumber = 0;
              const targetBarNumber = hitIndex * pattern;

              for (let sceneIdx = 0; sceneIdx < sceneTempos.length; sceneIdx++) {
                const sceneBPM = sceneTempos[sceneIdx];
                const sceneStart = sceneChanges[sceneIdx];
                const sceneEnd = sceneChanges[sceneIdx + 1] || (sceneStart + 30);
                const secondsPerBeat = 60 / sceneBPM;
                const secondsPerBar = secondsPerBeat * 4;

                // Calculate where the next bar should start based on accumulated beats
                const beatsIntoFirstBar = accumulatedBeats % 4;
                const beatsUntilNextBar = beatsIntoFirstBar === 0 ? 0 : (4 - beatsIntoFirstBar);
                const firstBarTime = sceneStart + (beatsUntilNextBar * secondsPerBeat);

                // Walk through bars in this scene
                let barTime = firstBarTime;
                while (barTime < sceneEnd) {
                  // Check if this is our target bar
                  if (barNumber === targetBarNumber) {
                    return barTime;
                  }

                  barNumber++;
                  barTime += secondsPerBar;
                }

                // Update accumulated beats for next scene
                const sceneDuration = sceneEnd - sceneStart;
                accumulatedBeats += sceneDuration / secondsPerBeat;
              }

              // If we got here, target is beyond all scenes
              // Continue from last scene with last tempo
              const lastBPM = sceneTempos[sceneTempos.length - 1];
              const lastSceneEnd = sceneChanges[sceneChanges.length - 1];
              const secondsPerBeat = 60 / lastBPM;
              const secondsPerBar = secondsPerBeat * 4;

              // Calculate time for remaining bars
              const barsNeeded = targetBarNumber - barNumber;
              return lastSceneEnd + (barsNeeded * secondsPerBar);
            };

            const expectedHitTimes = result.file_paths.map((_, i) => calculateExpectedHitTime(i));
            console.log(`🥁 Expected hit times:`, expectedHitTimes.filter(t => t !== null).map(t => t.toFixed(2) + 's'));

            // Verify backend hit times align with expected times (skip nulls)
            const backendHitsAligned = result.start_times.every((time, idx) => {
              if (expectedHitTimes[idx] === null) return true; // Skip validation for null
              return Math.abs(time - expectedHitTimes[idx]) < 0.01; // Within 10ms tolerance
            });

            if (!backendHitsAligned) {
              console.warn('⚠️ Backend hit times do not align! Recalculating...');
              console.warn('   Backend times:', result.start_times);
              console.warn('   Expected times:', expectedHitTimes);
            }

            // Create INDIVIDUAL tracks for each drum hit
            // Each track can be independently dragged and edited
            result.file_paths.forEach((filePath, index) => {
              const calculatedTime = expectedHitTimes[index];

              // Skip if no valid time (e.g., beyond available scene changes)
              if (calculatedTime === null) {
                console.log(`  Drum ${index}: skipped (no scene change available)`);
                return;
              }

              const backendTime = result.start_times[index] || 0;
              const useTime = backendHitsAligned ? backendTime : calculatedTime;
              const velocity = result.velocities ? result.velocities[index] : 1.0;

              if (!backendHitsAligned && index < 5) {
                console.log(`  Hit ${index}: backend=${backendTime.toFixed(2)}s, calculated=${calculatedTime.toFixed(2)}s, using=${useTime.toFixed(2)}s`);
              }

              const track = {
                id: `drum-hit-${Date.now()}-${index}`,
                name: `Drum ${index + 1}`,
                audioUrl: filePath,
                duration: 1, // Drum hits are typically ~1 second
                startPosition: useTime,
                gain: velocity, // Use velocity as track gain
                isMuted: false,
                isSolo: false,
                cropStart: 0,
                cropEnd: 0
              };

              dispatch({
                type: 'ADD_TRACK',
                payload: { busId: sfxBusId, track }
              });
            });

            console.log(`✅ Added ${result.file_paths.length} individual drum tracks to SFX bus`);
          } else {
            console.warn('⚠️ No drum samples returned');
          }
        }

        console.log('✅ Drum generation complete!');
      } catch (error) {
        console.error('❌ Drum generation error:', error);
        alert(`Drum generation failed: ${error.message}`);
      }

      return; // Exit early - don't run normal generation
    }

    // Normal generation flow (non-drums or non-MIDI mode)
    // Collect all parameters from state

    // Map ensemble subgroups to backend defaults
    let backendSubgroup = state.generationParams.instrumentSubgroup || 'ensemble_strings';
    if (backendSubgroup === 'ensemble_strings') {
      backendSubgroup = 'violin';
    } else if (backendSubgroup === 'ensemble_brass') {
      backendSubgroup = 'trumpet';
    } else if (backendSubgroup === 'ensemble_winds') {
      backendSubgroup = 'sax';
    }

    const params = {
      // Selected mode
      selectedMode,

      // File inputs (handled separately)
      // audioFile: state.uploadedFile?.file,

      // Instrument selection
      instrumentGroup: state.generationParams.instrumentGroup || 'strings',
      instrumentSubgroup: backendSubgroup,
      generationKey: state.generationParams.generationKey || 'C',

      // MIDI target (when in MIDI mode)
      midiTarget: state.generationParams.midiTarget || 'melody',

      // Generation parameters
      seed: state.generationParams.seed ?? 0,
      steps: state.generationParams.steps ?? 20,
      adapterScale: state.generationParams.adapterScale ?? 1.0,
      cfgWeight: state.generationParams.cfgWeight ?? 3.5,
      t0: state.generationParams.t0 ?? 0.0,
      noiseLevel: state.generationParams.noiseLevel ?? 0.7,

      // Variance control parameters
      temperature: state.generationParams.temperature ?? 1.0,
      eta: state.generationParams.eta ?? 0.5,

      // Processing mode
      midiMode: state.generationParams.midiMode || false,
      renderAndExtract: state.generationParams.renderAndExtract || false,
      renderExtractMono: state.generationParams.renderExtractMono || false,
      debugMode: state.generationParams.debugMode || false,
      tempoOverride: (state.generationParams.tempoOverride ?? state.bpm ?? 120),
      tapeSpeed: state.generationParams.tapeSpeed ?? 1.0,
      slowdownMethod: state.generationParams.slowdownMethod || 'stretch',
      upsampleMode: state.generationParams.upsampleMode ?? false,
      upsampleNoiseLevel: state.generationParams.upsampleNoiseLevel ?? 0.3,
      upsampleSteps: state.generationParams.upsampleSteps ?? 20,

      // Conditioning gains
      instrumentStrength: state.generationParams.instrumentStrength ?? 1.0,
      instBoost: state.generationParams.instBoost ?? 2.5,
      pianoRollGain: state.generationParams.pianoRollGain ?? 1.0,
      ampGain: state.generationParams.ampGain ?? 1.0,
      rframeGain: state.generationParams.rframeGain ?? 1.0,
      rbendGain: state.generationParams.rbendGain ?? 1.0,
      encodecGain: state.generationParams.encodecGain ?? 1.0,

      // MIDI fidelity
      pitchFidelityBoost: state.generationParams.pitchFidelityBoost ?? 1.0,
      onsetGuidanceBoost: state.generationParams.onsetGuidanceBoost ?? 0.0,
      pitchSnapStrength: state.generationParams.pitchSnapStrength ?? 0.5,

      // Sample recreation features
      useTimeVaryingNoise: state.generationParams.useTimeVaryingNoise || false,
      onsetPreservation: state.generationParams.onsetPreservation ?? 0.7,
      useMultiresolutionMixing: state.generationParams.useMultiresolutionMixing ?? true,
      useOnsetWeightedEncodec: state.generationParams.useOnsetWeightedEncodec || false,
      encodecOnsetBoost: state.generationParams.encodecOnsetBoost ?? 2.0,

      // Decoder options
      useOverlapDecoder: state.generationParams.useOverlapDecoder ?? true,
      monophonicMode: state.generationParams.monophonicMode ?? true,
      arrangeMode: state.generationParams.arrangeMode || false,
      fattenMode: state.generationParams.fattenMode ?? true,
      fattenType: state.generationParams.fattenType || 'real',

      // Test-time enhancement options
      useBestOfN: state.generationParams.bestOfNMode || false,
      nCandidates: state.generationParams.nCandidates || 12,
      useSelfConsistency: state.generationParams.selfConsistencyMode || false,
      consistencySamples: state.generationParams.consistencySamples || 3,

      // Output options
      // Auto-enable voice separation when monophonic mode is on and audio file is uploaded
      enableVoiceSeparation: state.generationParams.enableVoiceSeparation ||
        (state.generationParams.monophonicMode && state.fileType === 'audio'),
      enableMidiExport: state.generationParams.enableMidiExport ?? true,
      fastModeVariant: state.generationParams.fastModeVariant || '',

      // Extraction formats (for selective feature extraction)
      extractFormats: state.generationParams.extractFormats || ['midi', 'amp', 'rframe', 'rbend', 'encodec'],

      // Scene changes from video
      sceneDurations: state.video?.sceneChanges && state.video.sceneChanges.length > 1
        ? sceneToDurations(state.video.sceneChanges, state.video.duration || state.totalDuration)
        : null,

      // Automation data from automation window
      automationData: state.automationWindow.points && state.automationWindow.points.length > 0
        ? { points: state.automationWindow.points }
        : null
    };

    console.log('📤 Generation parameters:', params);
    console.log('🎬 Scene durations:', params.sceneDurations);

    // Log voice separation auto-enable
    if (params.enableVoiceSeparation && state.fileType === 'audio' && state.generationParams.monophonicMode) {
      console.log('🎵 Voice separation auto-enabled for audio file in monophonic mode');
    }

    // Log automation data if present
    if (params.automationData && params.automationData.points) {
      console.log('🎛️ Including automation data:', params.automationData.points.length, 'points');
      console.log('   First point:', params.automationData.points[0]);
      console.log('   Last point:', params.automationData.points[params.automationData.points.length - 1]);
    } else {
      console.log('ℹ️  No automation data to send');
    }

    // Get audio file if uploaded
    // state.uploadedFile IS the File object directly, not an object with a .file property
    const audioFile = state.uploadedFile || null;

    try {
      // Call generation function from hook
      await generate(params, audioFile);

      console.log('✅ Generation complete!');
    } catch (error) {
      console.error('❌ Generation error:', error);
    }
  }, [state, selectedMode, generate, generationMode, dispatch, setIsGenerating, setGenerationError, setGenerationProgress, startTimer, stopTimer]);


  return (
    <div id="audiodiv" className={styles.panelGrid}>
      {/* Header */}
      <div className={styles.panelHeader}>
        <h4 className={styles.panelTitle}>dø v1</h4>
      </div>

      {/* Folder Tabs */}
      <div className={styles.folderTabs}>
        <button
          className={`${styles.folderTab} ${activeTab === 'instruments' ? styles.active : ''}`}
          onClick={() => {
            setActiveTab('instruments');
            setGenerationMode('do-v1');
          }}
        >
          <i className="fa-solid fa-guitar"></i>
          Instruments
        </button>
        <button
          className={`${styles.folderTab} ${activeTab === 'vocals' ? styles.active : ''}`}
          onClick={() => {
            setActiveTab('vocals');
            setGenerationMode('ace-step');
          }}
        >
          <i className="fa-solid fa-microphone"></i>
          Vocals
        </button>
        <button
          className={`${styles.folderTab} ${activeTab === 'drums' ? styles.active : ''}`}
          onClick={() => {
            setActiveTab('drums');
            setGenerationMode('mido');
          }}
        >
          <i className="fa-solid fa-drum"></i>
          Drums
        </button>
      </div>

      {/* Show content for all tabs */}
      {(activeTab === 'instruments' || activeTab === 'vocals' || activeTab === 'drums') && (
        <>

      {/* Input */}
      <CollapsibleSection
        title="Input"
        isCollapsed={collapsedSections.conditioning}
        onToggle={() => toggleSection('conditioning')}
      >
        <FileUpload
          uploadedFile={{ file: state.uploadedFile, fileType: state.fileType }}
          onFileSelect={handleFileSelect}
          onClearFile={handleClearFile}
          isGenerating={isGenerating}
        />
      </CollapsibleSection>

      {/* Instrument / dø Controls */}
      {generationMode === 'ace-step' ? (
        <CollapsibleSection
          title="Settings"
          isCollapsed={collapsedSections.aceStep}
          onToggle={() => toggleSection('aceStep')}
        >
          <DoVoxControls
            params={state.generationParams}
            updateParam={updateParam}
          />
        </CollapsibleSection>
      ) : (
        <CollapsibleSection
          title="Instrument"
          isCollapsed={collapsedSections.instrument}
          onToggle={() => toggleSection('instrument')}
        >
          {!state.generationParams.midiMode ? (
            <InstrumentSelection
              params={state.generationParams}
              updateParam={updateParam}
            />
          ) : (
            <MIDITargetSelection
              params={state.generationParams}
              updateParam={updateParam}
            />
          )}
        </CollapsibleSection>
      )}

      {/* Hide advanced sections in MIDI mode and dø mode */}
      {generationMode !== 'mido' && generationMode !== 'ace-step' && (
        <CollapsibleSection
          title="Advanced Parameters"
          number="3"
          isCollapsed={collapsedSections.advanced}
          onToggle={() => toggleSection('advanced')}
        >
          {/* 3.1. Processing Mode */}
          <CollapsibleSection
            title="Processing Mode"
            number="3.1"
            isCollapsed={collapsedSections.processing}
            onToggle={() => toggleSection('processing')}
          >
            <ProcessingMode
              params={state.generationParams}
              updateParam={updateParam}
              uploadedFile={state.uploadedFile}
              timelineBpm={state.bpm}
            />
          </CollapsibleSection>

          {/* 3.2. Generation Parameters */}
          <CollapsibleSection
            title="Generation Parameters"
            number="3.2"
            isCollapsed={collapsedSections.generation}
            onToggle={() => toggleSection('generation')}
          >
            <GenerationParameters
              params={state.generationParams}
              updateParam={updateParam}
            />
          </CollapsibleSection>

          {/* 3.3. Conditioning Stream Gains */}
          <CollapsibleSection
            title="Conditioning Stream Gains"
            number="3.3"
            isCollapsed={collapsedSections.conditioningGains}
            onToggle={() => toggleSection('conditioningGains')}
          >
            <ConditioningStreamGains
              params={state.generationParams}
              updateParam={updateParam}
            />
          </CollapsibleSection>

          {/* 3.4 Extraction Formats */}
          <CollapsibleSection
            title="Extraction Formats"
            number="3.4"
            isCollapsed={collapsedSections.extractionFormats}
            onToggle={() => toggleSection('extractionFormats')}
          >
            <ExtractionFormats
              params={state.generationParams}
              updateParam={updateParam}
            />
          </CollapsibleSection>

          {/* 3.5. MIDI Fidelity Enhancements */}
          <CollapsibleSection
            title="MIDI Fidelity Enhancements"
            number="3.5"
            isCollapsed={collapsedSections.midiFidelity}
            onToggle={() => toggleSection('midiFidelity')}
          >
            <MidiFidelityEnhancements
              params={state.generationParams}
              updateParam={updateParam}
            />
          </CollapsibleSection>

          {/* 3.6. Sample Recreation Features */}
          <CollapsibleSection
            title="Sample Recreation Features"
            number="3.6"
            isCollapsed={collapsedSections.sampleRecreation}
            onToggle={() => toggleSection('sampleRecreation')}
          >
            <SampleRecreationFeatures
              params={state.generationParams}
              updateParam={updateParam}
            />
          </CollapsibleSection>

          {/* 3.7. Audio Decoder Options */}
          <CollapsibleSection
            title="Audio Decoder Options"
            number="3.7"
            isCollapsed={collapsedSections.audioDecoder}
            onToggle={() => toggleSection('audioDecoder')}
          >
            <AudioDecoderOptions
              params={state.generationParams}
              updateParam={updateParam}
            />
          </CollapsibleSection>

          {/* 3.8. Output Configuration */}
          <CollapsibleSection
            title="Output Configuration"
            number="3.8"
            isCollapsed={collapsedSections.output}
            onToggle={() => toggleSection('output')}
          >
            <OutputConfiguration
              params={state.generationParams}
              updateParam={updateParam}
            />
          </CollapsibleSection>

          {/* 3.9. Tape Speed Controls */}
          <CollapsibleSection
            title="Tape Speed / Slowdown"
            number="3.9"
            isCollapsed={collapsedSections.tapeSpeed}
            onToggle={() => toggleSection('tapeSpeed')}
          >
            <TapeSpeedControls
              params={state.generationParams}
              updateParam={updateParam}
            />
          </CollapsibleSection>
        </CollapsibleSection>
      )}

      {/* Error Display */}
      {generationError && (
        <div className={styles.errorContainer}>
          <i className="fa-solid fa-triangle-exclamation"></i>
          <span>{generationError}</span>
        </div>
      )}

      {/* Progress Display */}
      {isGenerating && generationProgress && (
        <div className={styles.progressContainer}>
          <div className={styles.progressHeader}>
            <i className="fa-solid fa-spinner fa-spin"></i>
            <span>
              Generating audio...
              {generationProgress.totalVoices > 0 && ` (${generationProgress.completedVoices}/${generationProgress.totalVoices} voices)`}
              {elapsedTime && ` - ${elapsedTime}`}
            </span>
          </div>
          <div className={styles.progressBar}>
            <div
              className={styles.progressFill}
              style={{
                width: `${(generationProgress.progress || 0) * 100}%`
              }}
            ></div>
          </div>
        </div>
      )}

      {/* Master Denoising Strength Slider */}
      <div className={styles.denoisingMaster} style={{ marginTop: '20px', marginBottom: '15px' }}>
        <label>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '14px', fontWeight: 'bold', color: 'var(--color-primary-blue)' }}>
              <i className="fa-solid fa-waveform-lines"></i> Denoising Strength
            </span>
            <span style={{ fontSize: '16px', fontWeight: 'bold', color: 'var(--color-primary-blue)' }}>
              {(state.generationParams.t0 ?? 1.0).toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            className={styles.denoisingRange}
            min="0"
            max="1"
            step="0.05"
            value={state.generationParams.t0 ?? 1.0}
            onChange={(e) => updateParam('t0', parseFloat(e.target.value))}
            style={{
              width: '100%',
              height: '8px'
            }}
          />
          <div style={{ fontSize: '11px', color: '#888', marginTop: '6px', lineHeight: '1.4' }}>
            Controls T0 and Noise Level. Higher = more creative/noise, Lower = more faithful to input.
          </div>
        </label>
      </div>

          {/* Generate and Save Buttons */}
          <div className={styles.buttonGroup}>
            <GlassButtonWrapper
              className={styles.generateButton}
              onClick={handleGenerate}
              disabled={isGenerating}
            >
              {isGenerating ? (
                <>
                  <i className="fa-solid fa-spinner fa-spin"></i>
                  <span>Generating...</span>
                </>
              ) : (
                <>
                  <i className="fa-solid fa-wand-magic-sparkles"></i>
                  <span>Generate</span>
                </>
              )}
            </GlassButtonWrapper>
            <GlassButtonWrapper
              className={styles.saveButton}
              onClick={handleSaveParams}
              disabled={isGenerating}
              title="Save current parameters for this instrument"
            >
              <i className="fa-solid fa-floppy-disk"></i>
            </GlassButtonWrapper>
          </div>
        </>
      )}
    </div>
  );
});

GenerationPanelOptimized.displayName = 'GenerationPanelOptimized';

export default GenerationPanelOptimized;
