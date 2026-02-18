import React, { useState, useRef, useEffect, useCallback } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.js';
import MidiCorrectionGrid from './MidiCorrectionGrid';
import './AudioLabeler.css';

const API_BASE = 'https://doseedo.com/api';

const GROUPS = [
  'guitar', 'piano', 'bass', 'strings', 'brass',
  'winds', 'voice', 'dialogue', 'drums', 'e-drums', 'synth', 'organ', 'percussion', 'plucked', 'mallets',
  'fx', 'ensemble', 'full-track'
];

const SPECIAL_TAGS = ['silent', 'junk', 'mix'];

// Subgroups organized by parent group
const SUBGROUPS = {
  brass: ['french_horn', 'trombone', 'trumpet', 'tuba'],
  strings: ['cello', 'viola', 'violin'],
  winds: ['clarinet', 'flute', 'oboe', 'sax'],
  bass: ['electric_bass', 'upright_bass'],
  guitar: ['acoustic_guitar', 'electric_guitar'],
  piano: ['acoustic_piano', 'keys'],
};

// Colors for different instruments in multi-label mode
const GROUP_COLORS = {
  guitar: 'rgba(231, 76, 60, 0.3)',
  piano: 'rgba(155, 89, 182, 0.3)',
  bass: 'rgba(52, 152, 219, 0.3)',
  strings: 'rgba(26, 188, 156, 0.3)',
  brass: 'rgba(243, 156, 18, 0.3)',
  winds: 'rgba(230, 126, 34, 0.3)',
  voice: 'rgba(255, 107, 157, 0.3)',
  dialogue: 'rgba(245, 158, 11, 0.3)',
  drums: 'rgba(149, 165, 166, 0.3)',
  synth: 'rgba(0, 212, 255, 0.3)',
  organ: 'rgba(168, 85, 247, 0.3)',
  percussion: 'rgba(34, 197, 94, 0.3)',
  plucked: 'rgba(251, 191, 36, 0.3)',
  mallets: 'rgba(236, 72, 153, 0.3)',
  'e-drums': 'rgba(99, 102, 241, 0.3)',
  fx: 'rgba(139, 92, 246, 0.3)',
  ensemble: 'rgba(20, 184, 166, 0.3)',
  'full-track': 'rgba(244, 114, 182, 0.3)',
  silent: 'rgba(148, 163, 184, 0.3)',
  junk: 'rgba(239, 68, 68, 0.3)',
};

// Data sources organized by type
const DATA_SOURCES = [
  // Main data
  { id: 'master', name: '📋 Master Manifest', endpoint: '/master-manifest', category: 'main' },
  { id: 'consolidated', name: '📊 Consolidated Manifest', endpoint: '/consolidated-manifest', category: 'main' },
  // Single-label classifiers
  { id: 'instrument', name: '🎸 Group Classifier', endpoint: '/classifier/instrument/predictions', category: 'classifier', description: 'Single instrument group prediction' },
  { id: 'subgroups', name: '🎺 Subgroup Classifier', endpoint: '/classifier/subgroups', category: 'classifier', description: 'Instrument subtype (trumpet, violin, etc.)' },
  // Multi-label classifiers
  { id: 'multilabel', name: '🎹 Multi-Group Classifier', endpoint: '/classifier/multilabel/predictions', category: 'classifier', description: 'Detects multiple instruments' },
  { id: 'multilabel_other', name: '🎷 Multi-Group Other', endpoint: '/classifier/other/predictions', category: 'classifier', description: 'brass/strings/winds/synth on other stem' },
  // Unified self-improving classifier
  { id: 'unified', name: '🧠 Unified Classifier', endpoint: '/classifier/unified/predictions', category: 'classifier', description: 'Self-improving: group + subgroup + multilabel' },
  // Mix detection
  { id: 'ensemble', name: '👥 Ensemble Detector', endpoint: '/classifier/ensemble', category: 'mix', description: 'Mix vs Isolated (latent-based)' },
  // Stem analysis (ground truth from Demucs)
  { id: 'stem_energy', name: '📊 Stem Energy (GT)', endpoint: '/classifier/stem-energy', category: 'stems', description: 'Demucs stem energy over time (training GT)' },
  { id: 'separated', name: '🔀 Separated Stems', endpoint: '/classifier/separated-stems', category: 'stems', description: 'Demucs stem separation summary' },
  // Silence detection
  { id: 'silence', name: '🔇 Silence Detection', endpoint: '/silence-detection', category: 'analysis', description: 'Temporal silence detection results' },
];

// Views available for each data source - Batch Review Modes
const VIEWS = [
  { id: 'all', name: 'All Entries' },
  { id: 'mismatches', name: '⚠️ Mismatches Only', desc: 'Classifier disagrees with manifest' },
  { id: 'low_confidence', name: '🔍 Low Confidence (<65%)', desc: 'Items needing review' },
  { id: 'boundary', name: '⚖️ Boundary Cases', desc: 'Near decision boundary' },
  { id: 'needs_review', name: '📋 Needs Review' },
  { id: 'corrected', name: '✓ My Corrections' },
  { id: 'mix', name: '🎵 Mix/Room Files' },
  { id: 'isolated', name: '🎸 Isolated Only' },
];

const AudioLabeler = () => {
  // Simplified navigation state
  const [dataSource, setDataSource] = useState('master');
  const [activeView, setActiveView] = useState('all');
  const [activeTab, setActiveTab] = useState('label'); // 'label', 'train', 'stats'

  // Dataset stats
  const [datasetStats, setDatasetStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsExpandedGroup, setStatsExpandedGroup] = useState(null);
  const [statsManifest, setStatsManifest] = useState('consolidated_manifest.json');
  const [availableManifests, setAvailableManifests] = useState([]);

  // Legacy classifier state (for backwards compatibility)
  const [classifiers, setClassifiers] = useState([]);
  const [activeClassifierId, setActiveClassifierId] = useState('instrument');

  // Parse classifier ID to get type and version
  const parseClassifierId = (id) => {
    if (!id) return { type: 'instrument', version: 'current' };
    const parts = id.split(':');
    return {
      type: parts[0],
      version: parts[1] || 'current'
    };
  };

  const { type: activeClassifier, version: activeVersion } = parseClassifierId(activeClassifierId);

  // Training dashboard state
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [isTraining, setIsTraining] = useState(false);

  // Predictions/Flagged items
  const [items, setItems] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isLoadingItems, setIsLoadingItems] = useState(false);
  const [labeledPaths, setLabeledPaths] = useState(new Set());
  const [labeledCount, setLabeledCount] = useState(0);

  // Filters
  const [confidenceFilter, setConfidenceFilter] = useState('low');
  const [flaggedFilter, setFlaggedFilter] = useState('all');
  const [predictedGroupFilter, setPredictedGroupFilter] = useState(''); // For predictions tab
  const [predManifestGroupFilter, setPredManifestGroupFilter] = useState(''); // Manifest group filter for predictions
  const [matchFilter, setMatchFilter] = useState(''); // matches, mismatches
  const [multiFilter, setMultiFilter] = useState(''); // multi, single, or empty for all
  const [mixFilter, setMixFilter] = useState(''); // mix, room, mix_or_room
  const [isolatedFilter, setIsolatedFilter] = useState(''); // isolated, mix, unknown
  const [isolatedStats, setIsolatedStats] = useState({}); // {isolated, mix, unknown, analyzed_rate}
  const [sessionInstrumentFilter, setSessionInstrumentFilter] = useState('');
  const [availableSessionInstruments, setAvailableSessionInstruments] = useState([]);
  const [manifestGroupFilter, setManifestGroupFilter] = useState('');
  const [correctionFilter, setCorrectionFilter] = useState(''); // corrected, multilabel, roomy, bleed
  const [availableGroups, setAvailableGroups] = useState([]);
  const [predictedGroups, setPredictedGroups] = useState([]); // Groups from predictions
  const [manifestGroupsInPredictions, setManifestGroupsInPredictions] = useState([]); // Manifest groups in predictions
  const [matchStats, setMatchStats] = useState({}); // {matches, mismatches, match_rate}
  const [multiStats, setMultiStats] = useState({}); // {multi, single, multi_rate}
  const [manifestTotal, setManifestTotal] = useState(0);
  const [correctionStats, setCorrectionStats] = useState({});
  const [subgroupFilter, setSubgroupFilter] = useState('');
  const [availableSubgroups, setAvailableSubgroups] = useState([]);
  const [sortBy, setSortBy] = useState(''); // confidence_asc, confidence_desc, group, subgroup
  const [sourceFilter, setSourceFilter] = useState(''); // original, classifier, flagged, corrected, mix
  const [sourceCounts, setSourceCounts] = useState({});
  const [selectedSubgroup, setSelectedSubgroup] = useState(''); // For editing subgroup
  const [pendingGroup, setPendingGroup] = useState(''); // Group selected but not yet confirmed (allows subgroup selection)
  const [showFileList, setShowFileList] = useState(false);
  const [fileListGroupFilter, setFileListGroupFilter] = useState('');
  const [fileListSubgroupFilter, setFileListSubgroupFilter] = useState('');
  const [fileListSourceFilter, setFileListSourceFilter] = useState('');
  const [fileListVerifiedFilter, setFileListVerifiedFilter] = useState('');

  // Separated stems state
  const [isStemMode, setIsStemMode] = useState(false);
  const [isTemporalMode, setIsTemporalMode] = useState(false);
  const [stemModeOption, setStemModeOption] = useState('auto'); // auto, normal, temporal
  const [availableStemModes, setAvailableStemModes] = useState({ normal: false, temporal: false });
  const [timelineRegions, setTimelineRegions] = useState([]); // For displaying temporal regions on waveform

  // Demucs stem playback state
  const [selectedStem, setSelectedStem] = useState('mix'); // 'mix', 'drums', 'bass', 'vocals', 'other', 'guitar', 'piano'
  const STEM_NAMES = ['mix', 'drums', 'bass', 'vocals', 'other', 'guitar', 'piano'];

  // Audio state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState('');

  // Multi-label state
  const [multiLabelMode, setMultiLabelMode] = useState(false);
  const [selectedLabels, setSelectedLabels] = useState([]);
  const [regions, setRegions] = useState([]); // {id, start, end, labels: []}
  const [isRoomy, setIsRoomy] = useState(false); // modifier for room mics
  const [hasVerb, setHasVerb] = useState(false); // has reverb
  const [isMidi, setIsMidi] = useState(false); // is MIDI/virtual instrument
  const [hasBleed, setHasBleed] = useState(false); // has instrument bleed
  const [bleedConfirmed, setBleedConfirmed] = useState(false); // bleed selection confirmed, ready for main label
  const [bleedInstruments, setBleedInstruments] = useState([]); // which instruments are bleeding
  const [isolatedOverride, setIsolatedOverride] = useState(null); // null = use model, true = force isolated, false = force mix
  const [selectedSessionInstruments, setSelectedSessionInstruments] = useState([]); // for mix file multi-label editing
  const [isPushingCorrections, setIsPushingCorrections] = useState(false);
  const [pushResult, setPushResult] = useState(null);

  // Undo state - track last labeled item for undo
  const [lastLabeledItem, setLastLabeledItem] = useState(null); // {item, index, label}
  const [isUndoing, setIsUndoing] = useState(false);

  const waveformRef = useRef(null);
  const wavesurferRef = useRef(null);
  const regionsRef = useRef(null);
  const dragSelectionCleanupRef = useRef(null);

  // Dual waveform refs for GT comparison view
  const [isDualWaveformMode, setIsDualWaveformMode] = useState(false);
  const [gtCompareEnabled, setGtCompareEnabled] = useState(false); // Manual toggle
  const gtWaveformRef = useRef(null);
  const gtWavesurferRef = useRef(null);
  const gtRegionsRef = useRef(null);

  const currentItem = items[currentIndex] || null;

  // Fetch dataset stats (includes available manifests list) when stats tab is opened or manifest changes
  useEffect(() => {
    if (activeTab !== 'stats') return;
    setStatsLoading(true);
    setDatasetStats(null);
    fetch(`${API_BASE}/consolidated-manifest/stats?manifest=${encodeURIComponent(statsManifest)}`, { credentials: 'include' })
      .then(r => r.json())
      .then(data => {
        if (data.status === 'ok') {
          setDatasetStats(data);
          if (data.available_manifests) setAvailableManifests(data.available_manifests);
        } else console.error('Stats error:', data.message);
      })
      .catch(e => console.error('Failed to fetch stats:', e))
      .finally(() => setStatsLoading(false));
  }, [activeTab, statsManifest]);

  // Initialize WaveSurfer
  useEffect(() => {
    if (waveformRef.current && !wavesurferRef.current) {
      // Create regions plugin
      regionsRef.current = RegionsPlugin.create();

      wavesurferRef.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: 'rgba(100, 149, 237, 0.6)',
        progressColor: 'rgba(65, 105, 225, 0.9)',
        cursorColor: '#ff6b6b',
        cursorWidth: 2,
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        height: 128,
        responsive: true,
        normalize: false,
        backend: 'WebAudio',
        plugins: [regionsRef.current],
      });

      wavesurferRef.current.on('ready', () => {
        setDuration(wavesurferRef.current.getDuration());
        setIsLoading(false);
        wavesurferRef.current.play();
        // Prefetch next audio once current is loaded
        if (prefetchNext) prefetchNext();
      });

      wavesurferRef.current.on('audioprocess', () => {
        setCurrentTime(wavesurferRef.current.getCurrentTime());
      });

      wavesurferRef.current.on('seek', () => {
        setCurrentTime(wavesurferRef.current.getCurrentTime());
      });

      wavesurferRef.current.on('play', () => setIsPlaying(true));
      wavesurferRef.current.on('pause', () => setIsPlaying(false));
      wavesurferRef.current.on('finish', () => setIsPlaying(false));

      wavesurferRef.current.on('interaction', () => {
        if (!isPlaying) {
          wavesurferRef.current.play();
        }
      });

      // Handle decode errors - skip to next (but ignore abort errors from navigation)
      wavesurferRef.current.on('error', (err) => {
        // Ignore abort errors - these happen when switching tracks quickly
        if (err?.name === 'AbortError' || err?.message?.includes('abort')) {
          return;
        }
        console.error('WaveSurfer error:', err);
        setIsLoading(false);
        setStatus('Audio decode error - skipping...');
        // Auto-skip after a moment
        setTimeout(() => {
          setCurrentIndex(prev => prev + 1);
          setStatus('');
        }, 500);
      });

      // Region events
      regionsRef.current.on('region-created', (region) => {
        // Will be handled by our state
      });

      regionsRef.current.on('region-clicked', (region, e) => {
        e.stopPropagation();
        region.play();
      });
    }

    return () => {
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
        wavesurferRef.current = null;
      }
    };
  }, []);

  // Initialize GT WaveSurfer for dual waveform mode
  useEffect(() => {
    if (isDualWaveformMode && gtWaveformRef.current && !gtWavesurferRef.current) {
      gtRegionsRef.current = RegionsPlugin.create();

      gtWavesurferRef.current = WaveSurfer.create({
        container: gtWaveformRef.current,
        waveColor: 'rgba(34, 197, 94, 0.6)',  // Green for GT
        progressColor: 'rgba(22, 163, 74, 0.9)',
        cursorColor: '#ff6b6b',
        cursorWidth: 2,
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        height: 100,
        responsive: true,
        normalize: false,
        backend: 'WebAudio',
        plugins: [gtRegionsRef.current],
      });

      // Sync GT waveform with main waveform
      gtWavesurferRef.current.on('ready', () => {
        // Start playing automatically if main is playing
        if (wavesurferRef.current?.isPlaying()) {
          gtWavesurferRef.current.play();
        }
      });

      gtWavesurferRef.current.on('error', (err) => {
        if (err?.name === 'AbortError' || err?.message?.includes('abort')) return;
        console.error('GT WaveSurfer error:', err);
      });
    }

    return () => {
      if (gtWavesurferRef.current) {
        gtWavesurferRef.current.destroy();
        gtWavesurferRef.current = null;
      }
    };
  }, [isDualWaveformMode]);

  // Sync playback between waveforms in dual mode
  useEffect(() => {
    if (!isDualWaveformMode || !wavesurferRef.current || !gtWavesurferRef.current) return;

    const syncPlay = () => {
      if (gtWavesurferRef.current && !gtWavesurferRef.current.isPlaying()) {
        gtWavesurferRef.current.seekTo(wavesurferRef.current.getCurrentTime() / wavesurferRef.current.getDuration());
        gtWavesurferRef.current.play();
      }
    };
    const syncPause = () => {
      if (gtWavesurferRef.current) gtWavesurferRef.current.pause();
    };
    const syncSeek = () => {
      if (gtWavesurferRef.current && wavesurferRef.current) {
        const progress = wavesurferRef.current.getCurrentTime() / wavesurferRef.current.getDuration();
        gtWavesurferRef.current.seekTo(progress);
      }
    };

    wavesurferRef.current.on('play', syncPlay);
    wavesurferRef.current.on('pause', syncPause);
    wavesurferRef.current.on('seek', syncSeek);

    return () => {
      if (wavesurferRef.current) {
        wavesurferRef.current.un('play', syncPlay);
        wavesurferRef.current.un('pause', syncPause);
        wavesurferRef.current.un('seek', syncSeek);
      }
    };
  }, [isDualWaveformMode]);

  // Load audio in GT waveform when dual mode and item changes
  useEffect(() => {
    if (!isDualWaveformMode || !currentItem || !gtWavesurferRef.current) return;

    const audioPath = currentItem.original_path || currentItem.audio_path || currentItem.path;
    if (audioPath) {
      const audioUrl = `${API_BASE}/audio?path=${encodeURIComponent(audioPath)}&format=opus`;
      gtWavesurferRef.current.load(audioUrl);
    }
  }, [isDualWaveformMode, currentItem?.path]);

  // Render GT regions on GT waveform in dual mode
  useEffect(() => {
    if (!isDualWaveformMode || !gtWavesurferRef.current || !gtRegionsRef.current || !currentItem) return;

    const duration = gtWavesurferRef.current.getDuration();
    if (!duration || duration === 0) return;

    gtRegionsRef.current.clearRegions();

    // GT regions can come from different sources
    const gtRegions = currentItem.gt_regions || currentItem.regions || [];

    // If no temporal regions but has group label, show as full-duration region
    if (gtRegions.length === 0 && currentItem.current_label && currentItem.current_label !== 'undefined') {
      const label = currentItem.current_label;
      const color = GROUP_COLORS[label]
        ? GROUP_COLORS[label].replace('0.3', '0.5')
        : 'rgba(34, 197, 94, 0.5)';

      gtRegionsRef.current.addRegion({
        id: 'gt-full',
        start: 0,
        end: duration,
        color: color,
        drag: false,
        resize: false,
        content: `GT: ${label}`,
      });
    } else {
      gtRegions.forEach((region, idx) => {
        const firstLabel = region.labels?.[0] || region.label;
        const color = firstLabel && GROUP_COLORS[firstLabel]
          ? GROUP_COLORS[firstLabel].replace('0.3', '0.5')
          : 'rgba(34, 197, 94, 0.5)';

        gtRegionsRef.current.addRegion({
          id: `gt-${idx}`,
          start: region.start,
          end: region.end,
          color: color,
          drag: false,
          resize: false,
          content: region.labels?.join(', ') || region.label || '',
        });
      });
    }
  }, [isDualWaveformMode, currentItem?.path, currentItem?.gt_regions, currentItem?.regions, currentItem?.current_label]);

  // Render predicted regions on main waveform in dual mode
  useEffect(() => {
    if (!isDualWaveformMode || !wavesurferRef.current || !regionsRef.current || !currentItem) return;

    const duration = wavesurferRef.current.getDuration();
    if (!duration || duration === 0) return;

    regionsRef.current.clearRegions();

    const predRegions = currentItem.predicted_regions || currentItem.estimated_regions || [];

    // If no temporal regions but has predicted label, show as full-duration region
    if (predRegions.length === 0 && currentItem.predicted_class && currentItem.predicted_class !== 'undefined') {
      const label = currentItem.predicted_class;
      const confidence = currentItem.confidence;
      const color = GROUP_COLORS[label]
        ? GROUP_COLORS[label].replace('0.3', '0.5')
        : 'rgba(59, 130, 246, 0.5)';

      regionsRef.current.addRegion({
        id: 'pred-full',
        start: 0,
        end: duration,
        color: color,
        drag: false,
        resize: false,
        content: `Pred: ${label}${confidence ? ` (${Math.round(confidence * 100)}%)` : ''}`,
      });
    } else {
      predRegions.forEach((region, idx) => {
        const firstLabel = region.labels?.[0] || region.label;
        const color = firstLabel && GROUP_COLORS[firstLabel]
          ? GROUP_COLORS[firstLabel].replace('0.3', '0.5')
          : 'rgba(59, 130, 246, 0.5)';

        regionsRef.current.addRegion({
          id: `pred-${idx}`,
          start: region.start,
          end: region.end,
          color: color,
          drag: false,
          resize: false,
          content: `${region.labels?.join(', ') || region.label} ${region.confidence ? `(${Math.round(region.confidence * 100)}%)` : ''}`,
        });
      });
    }
  }, [isDualWaveformMode, currentItem?.path, currentItem?.predicted_regions, currentItem?.estimated_regions, currentItem?.predicted_class, currentItem?.confidence]);

  // Enable/disable drag selection based on multiLabelMode
  useEffect(() => {
    if (regionsRef.current && wavesurferRef.current) {
      // Cleanup previous drag selection
      if (dragSelectionCleanupRef.current) {
        dragSelectionCleanupRef.current();
        dragSelectionCleanupRef.current = null;
      }

      // Enable drag selection if in multi-label mode with labels selected
      if (multiLabelMode && selectedLabels.length > 0) {
        dragSelectionCleanupRef.current = regionsRef.current.enableDragSelection({
          color: GROUP_COLORS[selectedLabels[0]] || 'rgba(100, 149, 237, 0.3)',
        });
      }
    }
  }, [multiLabelMode, selectedLabels]);

  // Handle new regions created by drag
  useEffect(() => {
    if (!regionsRef.current) return;

    const handleRegionCreated = (region) => {
      if (multiLabelMode && selectedLabels.length > 0) {
        const newRegion = {
          id: region.id,
          start: region.start,
          end: region.end,
          labels: [...selectedLabels],
        };
        setRegions(prev => [...prev, newRegion]);

        // Update region color based on labels
        const color = selectedLabels.length === 1
          ? GROUP_COLORS[selectedLabels[0]]
          : 'rgba(100, 149, 237, 0.4)';
        region.setOptions({ color });
      }
    };

    regionsRef.current.on('region-created', handleRegionCreated);

    return () => {
      regionsRef.current.un('region-created', handleRegionCreated);
    };
  }, [multiLabelMode, selectedLabels]);

  // Clear regions and modifiers when loading new audio
  useEffect(() => {
    if (currentItem && regionsRef.current) {
      regionsRef.current.clearRegions();
      setRegions([]);
      setSelectedLabels([]);
      setBleedInstruments([]);
      setBleedConfirmed(false);
      setHasBleed(false);
      setHasVerb(false);
      setIsMidi(false);
      setIsolatedOverride(null);
    }
  }, [currentItem?.path]);

  // Fetch classifiers, labeled paths, and training status on mount
  useEffect(() => {
    fetchClassifiers();
    fetchLabeledPaths();
    fetchTrainingStatus();
  }, []);

  const fetchLabeledPaths = async () => {
    try {
      const response = await fetch(`${API_BASE}/labeled-paths`);
      const data = await response.json();
      setLabeledPaths(new Set(data.paths || []));
      setLabeledCount(data.count || 0);
    } catch (err) {
      console.error('Failed to fetch labeled paths:', err);
    }
  };

  // Track initial load state
  const [initialLoadDone, setInitialLoadDone] = useState(false);

  // Fetch items when tab/filter changes (not when labeledPaths changes during session)
  useEffect(() => {
    if (activeClassifier) {
      fetchItems();
      setInitialLoadDone(true);
    }
  }, [activeClassifierId, activeClassifier, activeVersion, activeTab, confidenceFilter, flaggedFilter, predictedGroupFilter, predManifestGroupFilter, matchFilter, multiFilter, mixFilter, isolatedFilter, sessionInstrumentFilter, manifestGroupFilter, correctionFilter, stemModeOption, subgroupFilter, sortBy, sourceFilter, dataSource, activeView]);

  // Load audio when current item changes
  useEffect(() => {
    if (currentItem && wavesurferRef.current) {
      loadCurrentAudio();
    }
    // Reset stem selection when item changes
    setSelectedStem('mix');
  }, [currentItem?.path]);

  // Reload audio when stem selection changes
  useEffect(() => {
    if (currentItem && wavesurferRef.current && activeClassifier === 'demucs_other') {
      loadCurrentAudio();
    }
  }, [selectedStem]);

  // Restore correction settings when navigating to a corrected item
  useEffect(() => {
    if (!currentItem) return;

    // Initialize session instruments for mix files
    if (currentItem.session_instruments?.length > 0) {
      setSelectedSessionInstruments([...currentItem.session_instruments]);
    } else {
      setSelectedSessionInstruments([]);
    }

    // Reset subgroup and pending group selection when item changes
    setSelectedSubgroup('');
    setPendingGroup('');

    // If item has correction data, restore the settings
    if (currentItem.has_correction) {
      setIsRoomy(currentItem.roomy || false);
      setHasVerb(currentItem.has_verb || false);
      setIsMidi(currentItem.is_midi || false);
      setHasBleed(currentItem.has_bleed || false);
      setBleedInstruments(currentItem.bleed_instruments || []);

      if (currentItem.multi_label && currentItem.regions?.length > 0) {
        setMultiLabelMode(true);
        setRegions(currentItem.regions.map((r, i) => ({
          id: `region-${i}`,
          start: r.start,
          end: r.end,
          labels: r.labels || [],
        })));
      } else {
        setMultiLabelMode(false);
        setRegions([]);
      }
    } else if (currentItem.is_comparison) {
      // For comparison mode, display both estimated and GT regions
      setMultiLabelMode(true);
      const allRegions = [];

      // Add estimated regions (from temporal analysis) - blue tint
      (currentItem.estimated_regions || []).forEach((r, i) => {
        allRegions.push({
          id: `estimated-${i}`,
          start: r.start,
          end: r.end,
          labels: r.labels || [],
          isEstimated: true,
          confidences: r.confidences || {},
        });
      });

      // Add GT regions (from corrections) - green tint
      (currentItem.gt_regions || []).forEach((r, i) => {
        allRegions.push({
          id: `gt-${i}`,
          start: r.start,
          end: r.end,
          labels: r.labels || [],
          isGT: true,
        });
      });

      setRegions(allRegions);
      setTimelineRegions(allRegions);
    } else if (currentItem.is_temporal && currentItem.timeline?.length > 0) {
      // For temporal stem results, display timeline as regions
      setMultiLabelMode(true); // Use multi-label mode for display
      const timelineRegions = currentItem.timeline.map((t, i) => ({
        id: `timeline-${i}`,
        start: t.start,
        end: t.end,
        labels: t.instruments.map(inst => inst.instrument),
        isTimeline: true, // Mark as read-only timeline region
      }));
      setRegions(timelineRegions);
      setTimelineRegions(timelineRegions);
    } else {
      // Reset for uncorrected items
      setIsRoomy(false);
      setHasVerb(false);
      setIsMidi(false);
      setHasBleed(false);
      setBleedConfirmed(false);
      setBleedInstruments([]);
      setIsolatedOverride(null);
      setMultiLabelMode(false);
      setRegions([]);
      setTimelineRegions([]);
    }
  }, [currentItem?.path, currentItem?.has_correction, currentItem?.is_temporal, currentItem?.is_comparison]);

  // Render comparison/timeline regions on the waveform
  useEffect(() => {
    if (!regionsRef.current || !wavesurferRef.current || !currentItem) return;

    // Only render for comparison or temporal modes with regions
    if (!currentItem.is_comparison && !currentItem.is_temporal) return;
    if (regions.length === 0) return;

    // Wait for audio to be ready
    const duration = wavesurferRef.current.getDuration();
    if (!duration || duration === 0) return;

    // Clear existing wavesurfer regions first
    regionsRef.current.clearRegions();

    // Add regions to wavesurfer with appropriate colors
    regions.forEach(region => {
      let color;
      if (region.isEstimated) {
        color = 'rgba(59, 130, 246, 0.3)'; // Blue for estimated
      } else if (region.isGT) {
        color = 'rgba(34, 197, 94, 0.3)'; // Green for GT
      } else if (region.isTimeline) {
        color = 'rgba(100, 149, 237, 0.3)'; // Default blue for timeline
      } else {
        // Use instrument color if available
        const firstLabel = region.labels[0];
        color = firstLabel && GROUP_COLORS[firstLabel]
          ? GROUP_COLORS[firstLabel].replace(')', ', 0.3)').replace('rgb', 'rgba')
          : 'rgba(100, 149, 237, 0.3)';
      }

      regionsRef.current.addRegion({
        id: region.id,
        start: region.start,
        end: region.end,
        color: color,
        drag: false,
        resize: false,
        content: region.labels.join(', '),
      });
    });
  }, [regions, currentItem?.is_comparison, currentItem?.is_temporal, currentItem?.path]);

  // Render silent regions overlay on waveform when in silence detection mode
  useEffect(() => {
    if (!regionsRef.current || !wavesurferRef.current || !currentItem) return;
    if (dataSource !== 'silence') return;

    const silentRegions = currentItem.silent_regions;
    if (!silentRegions || silentRegions.length === 0) return;

    // Wait for audio to be ready (duration state is set on wavesurfer 'ready' event)
    if (!duration || duration === 0) return;

    // Clear existing regions
    regionsRef.current.clearRegions();

    // Add each silent region as a semi-transparent overlay
    silentRegions.forEach((region, i) => {
      const start = Math.max(0, region.start_sec);
      const end = Math.min(duration, region.end_sec);
      if (end <= start) return;

      regionsRef.current.addRegion({
        id: `silent-${i}`,
        start,
        end,
        color: 'rgba(255, 60, 60, 0.2)',
        drag: false,
        resize: false,
        content: `${region.duration_sec.toFixed(1)}s`,
      });
    });
  }, [dataSource, duration, currentItem?.path, currentItem?.silent_regions]);

  const fetchClassifiers = async () => {
    try {
      const response = await fetch(`${API_BASE}/classifiers`);
      const data = await response.json();
      const classifierList = data.classifiers || [];

      // Add separated stems as a pseudo-classifier
      classifierList.push({
        id: 'separated_stems',
        name: 'Separated Stems (Mix Analysis)',
        type: 'separated_stems',
        has_model: true,
        predictions_count: 0,
        flagged_count: 0,
      });

      // Add multilabel comparison as a pseudo-classifier
      classifierList.push({
        id: 'multilabel_comparison',
        name: 'Multi-Label Comparison (Est vs GT)',
        type: 'multilabel_comparison',
        has_model: true,
        predictions_count: 0,
        flagged_count: 0,
      });

      // Add demucs other classifier (brass/strings/winds/synth on separated 'other' stems)
      classifierList.push({
        id: 'demucs_other',
        name: 'Demucs Other Classifier (brass/strings/winds/synth)',
        type: 'demucs_other',
        has_model: true,
        predictions_count: 0,
        flagged_count: 0,
      });

      // Add mix classifier v2 (brass/strings/winds directly from mix)
      classifierList.push({
        id: 'mix_classifier_v2',
        name: 'Mix Classifier V2 (brass/strings/winds from mix)',
        type: 'mix_classifier_v2',
        has_model: true,
        predictions_count: 0,
        flagged_count: 0,
      });

      // Add mix classifier v3 (segment-trained, better temporal accuracy)
      classifierList.push({
        id: 'mix_classifier_v3',
        name: 'Mix Classifier V3 (segment-trained, temporal)',
        type: 'mix_classifier_v3',
        has_model: true,
        predictions_count: 0,
        flagged_count: 0,
      });

      // Add mix classifier v3 comparison (dual waveform GT vs Predicted)
      classifierList.push({
        id: 'mix_classifier_v3_compare',
        name: 'Mix V3 Comparison (Dual Waveform: GT vs Pred)',
        type: 'mix_classifier_v3_compare',
        has_model: true,
        predictions_count: 0,
        flagged_count: 0,
      });

      // Add merged manifest v2 (with subgroups, corrections, flagged fixes, mix labels)
      classifierList.push({
        id: 'merged_manifest_v2',
        name: 'Merged Manifest V2 (Subgroups + Corrections + Mix)',
        type: 'merged_manifest_v2',
        has_model: true,
        predictions_count: 0,
        flagged_count: 0,
      });

      // Add subgroup classifiers (brass->trumpet/trombone, strings->violin/cello, etc.)
      classifierList.push({
        id: 'subgroups',
        name: 'Subgroup Classifiers (brass/strings/winds/bass/guitar/piano)',
        type: 'subgroups',
        has_model: true,
        predictions_count: 25444,
        flagged_count: 0,
      });

      setClassifiers(classifierList);
    } catch (err) {
      console.error('Failed to fetch classifiers:', err);
    }
  };

  // Fetch training status for Training Dashboard
  const fetchTrainingStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/training/status`);
      const data = await response.json();
      setTrainingStatus(data);

      // Check if any jobs are running
      const history = data.training_history || [];
      const hasRunning = history.some(job => job.status === 'running');
      setIsTraining(hasRunning);
    } catch (err) {
      console.error('Failed to fetch training status:', err);
    }
  };

  // Check for training completion
  const checkTrainingCompletion = async () => {
    try {
      const response = await fetch(`${API_BASE}/training/check-all`);
      const data = await response.json();
      if (data.updated > 0) {
        setStatus(`${data.updated} training job(s) completed!`);
        fetchTrainingStatus();
      }
      return data;
    } catch (err) {
      console.error('Failed to check training:', err);
    }
  };

  // Poll for training status when training is active
  useEffect(() => {
    if (isTraining && activeTab === 'train') {
      const interval = setInterval(() => {
        checkTrainingCompletion();
      }, 5000); // Check every 5 seconds
      return () => clearInterval(interval);
    }
  }, [isTraining, activeTab]);

  // Trigger retraining
  const triggerRetrain = async (classifierType) => {
    setIsTraining(true);
    try {
      const response = await fetch(`${API_BASE}/training/retrain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ classifier: classifierType }),
      });
      const data = await response.json();
      if (data.status === 'ok') {
        setStatus(`Training started for ${classifierType}. Monitoring progress...`);
      } else {
        setStatus(`Training error: ${data.message}`);
        setIsTraining(false);
      }
      // Refresh status after a delay
      setTimeout(fetchTrainingStatus, 2000);
    } catch (err) {
      setStatus(`Training error: ${err.message}`);
      setIsTraining(false);
    }
  };

  const fetchItems = async () => {
    setIsLoadingItems(true);
    try {
      let url;

      // Common query params for all classifier endpoints
      const fetchLimit = predictedGroupFilter ? 50000 : 5000;
      const commonParams = `limit=${fetchLimit}${predictedGroupFilter ? `&group=${predictedGroupFilter}` : ''}${subgroupFilter ? `&subgroup=${subgroupFilter}` : ''}${sortBy ? `&sort_by=${sortBy}` : ''}${isolatedFilter ? `&isolated_filter=${isolatedFilter}` : ''}`;

      // Build URL based on data source
      if (dataSource === 'master') {
        url = `${API_BASE}/master-manifest?${commonParams}&view=${activeView}${sourceFilter ? `&source=${sourceFilter}` : ''}`;
      } else if (dataSource === 'consolidated') {
        const manifestParam = statsManifest && statsManifest !== 'consolidated_manifest.json' ? `&manifest=${statsManifest}` : '';
        url = `${API_BASE}/consolidated-manifest?${commonParams}&view=${activeView}${manifestParam}${sourceFilter ? `&source=${sourceFilter}` : ''}`;
      } else if (dataSource === 'instrument') {
        url = `${API_BASE}/classifier/instrument/predictions?version=${activeVersion}&confidence=${confidenceFilter}&${commonParams}${predManifestGroupFilter ? `&manifest_group=${predManifestGroupFilter}` : ''}${matchFilter ? `&match_filter=${matchFilter}` : ''}${multiFilter ? `&multi_filter=${multiFilter}` : ''}${mixFilter ? `&mix_filter=${mixFilter}` : ''}${sessionInstrumentFilter ? `&session_instrument=${sessionInstrumentFilter}` : ''}`;
      } else if (dataSource === 'subgroups') {
        url = `${API_BASE}/classifier/subgroups?${commonParams}`;
      } else if (dataSource === 'multilabel') {
        url = `${API_BASE}/classifier/multilabel/predictions?${commonParams}`;
      } else if (dataSource === 'multilabel_other') {
        url = `${API_BASE}/classifier/other/predictions?${commonParams}`;
      } else if (dataSource === 'unified') {
        url = `${API_BASE}/classifier/unified/predictions?confidence=${confidenceFilter}&${commonParams}${predManifestGroupFilter ? `&manifest_group=${predManifestGroupFilter}` : ''}${matchFilter ? `&match_filter=${matchFilter}` : ''}${multiFilter ? `&multi_filter=${multiFilter}` : ''}`;
      } else if (dataSource === 'ensemble') {
        url = `${API_BASE}/classifier/ensemble?${commonParams}`;
      } else if (dataSource === 'stem_energy') {
        url = `${API_BASE}/classifier/stem-energy?${commonParams}`;
      } else if (dataSource === 'separated') {
        url = `${API_BASE}/classifier/separated-stems?${commonParams}&mode=${stemModeOption}`;
      } else if (dataSource === 'silence') {
        url = `${API_BASE}/silence-detection?${commonParams}${sourceFilter ? `&classification=${sourceFilter}` : ''}&sort_by=silence_ratio`;
      } else if (activeTab === 'label') {
        // Fallback for legacy activeClassifier pattern
        url = `${API_BASE}/classifier/${activeClassifier}/predictions?version=${activeVersion}&confidence=${confidenceFilter}&${commonParams}`;
      } else if (activeTab === 'flagged') {
        url = `${API_BASE}/classifier/${activeClassifier}/flagged?version=${activeVersion}&flag_type=${flaggedFilter}&limit=2000`;
      } else if (activeTab === 'manifest') {
        url = `${API_BASE}/manifest/entries?limit=2000${manifestGroupFilter ? `&group=${manifestGroupFilter}` : ''}${correctionFilter ? `&correction_filter=${correctionFilter}` : ''}`;
      }

      const response = await fetch(url);
      const data = await response.json();

      // Check if this is separated stems mode or comparison mode
      const isStemData = activeClassifier === 'separated_stems';
      const isComparisonMode = activeClassifier === 'multilabel_comparison' || activeClassifier === 'demucs_other' || activeClassifier === 'mix_classifier_v2' || activeClassifier === 'mix_classifier_v3' || activeClassifier === 'mix_classifier_v3_compare';
      const isDualWaveform = activeClassifier === 'mix_classifier_v3_compare' || data.is_dual_waveform || gtCompareEnabled;
      const isTemporal = data.is_temporal || false;
      setIsStemMode(isStemData);
      setIsTemporalMode(isTemporal || isComparisonMode);
      setIsDualWaveformMode(isDualWaveform || gtCompareEnabled);
      if (data.available_modes) {
        setAvailableStemModes(data.available_modes);
      }

      let entries = (data.entries || data.items || []).map(e => ({
        ...e,
        predicted_class: e.predicted_group || e.predicted_label || e.predicted_class || e.predicted_subgroup,
        predicted_labels: e.predicted_labels || [], // For multilabel classifier
        is_multilabel_prediction: e.is_multilabel || false,
        is_multi: e.is_multi || false, // From binary multi classifier
        multi_probability: e.multi_probability || 0,
        current_label: e.true_label || e.current_label || e.current_group || e.current_subgroup,
        // Subgroup specific
        current_subgroup: e.current_subgroup,
        predicted_subgroup: e.predicted_subgroup,
        is_mismatch: e.is_mismatch || false,
        manifest_group: e.manifest_group || e.original_group || '', // Original group from manifest
        matches_manifest: e.matches_manifest || false, // Whether prediction matches manifest
        session_name: e.session_name || '',
        session_instruments: e.session_instruments || [], // Instruments detected in same session
        // Separated stems data
        stems: e.stems || [], // Array of stem data
        stems_detail: e.stems_detail || {}, // Detailed stem info
        detected_instruments: e.detected_instruments || e.estimated_instruments || [], // List of instruments detected
        is_separated_stems: isStemData,
        is_temporal: isTemporal,
        timeline: e.timeline || [], // Temporal timeline entries
        original_duration: e.original_duration || e.duration || 0,
        silent_stems: e.silent_stems || 0,
        active_stems: e.active_stems || 0,
        // Multilabel comparison data
        is_comparison: isComparisonMode || e.is_comparison || false,
        estimated_regions: e.estimated_regions || [], // From temporal analysis
        gt_regions: e.gt_regions || [], // From corrections (ground truth)
        estimated_instruments: e.estimated_instruments || e.detected_instruments || [],
        gt_instruments: e.gt_instruments || [],
        has_gt: e.has_gt || false,
        stem_segments: e.stem_segments || {},
      }));

      // Always populate group/subgroup options from response
      if (data.available_groups) {
        setPredictedGroups(data.available_groups);
        setAvailableGroups(data.available_groups);
      }
      if (data.available_subgroups) setAvailableSubgroups(data.available_subgroups);

      // Store match stats from predictions
      if (activeTab === 'predictions') {
        if (data.available_manifest_groups) setManifestGroupsInPredictions(data.available_manifest_groups);
        if (data.match_stats) setMatchStats(data.match_stats);
        if (data.multi_stats) setMultiStats(data.multi_stats);
        if (data.isolated_stats) setIsolatedStats(data.isolated_stats);
        if (data.available_session_instruments) setAvailableSessionInstruments(data.available_session_instruments);
        if (data.source_counts) setSourceCounts(data.source_counts);
      }

      // Store correction stats for manifest filter
      if (activeTab === 'manifest') {
        setManifestTotal(data.total || 0);
        if (data.correction_stats) setCorrectionStats(data.correction_stats);
      }

      // Filter out already-labeled items (except for manifest review)
      if (activeTab !== 'manifest') {
        entries = entries.filter(e => !labeledPaths.has(e.path));
      } else {
        // For manifest, mark items that have been corrected (using has_correction from backend)
        entries = entries.map(e => ({
          ...e,
          is_corrected: e.has_correction || labeledPaths.has(e.path),
        }));
      }

      // Apply view-based batch filtering
      if (activeView === 'mismatches') {
        entries = entries.filter(e => !e.matches_manifest && e.manifest_group && e.manifest_group !== 'unknown');
      } else if (activeView === 'low_confidence') {
        entries = entries.filter(e => (e.confidence || 0) < 0.65);
      } else if (activeView === 'boundary') {
        // Items where top two classes are close (within 10%)
        entries = entries.filter(e => {
          if (!e.all_probabilities) return false;
          const probs = Object.values(e.all_probabilities).sort((a, b) => b - a);
          return probs.length >= 2 && (probs[0] - probs[1]) < 0.1;
        });
      } else if (activeView === 'corrected') {
        entries = entries.filter(e => e.is_corrected || labeledPaths.has(e.path));
      } else if (activeView === 'mix') {
        entries = entries.filter(e => e.is_isolated === false || /mix|room/i.test(e.filename || ''));
      } else if (activeView === 'isolated') {
        entries = entries.filter(e => e.is_isolated === true);
      }

      setItems(entries);
      setCurrentIndex(0);
    } catch (err) {
      console.error('Failed to fetch items:', err);
      setItems([]);
    } finally {
      setIsLoadingItems(false);
    }
  };

  // Prefetch cache: stores pre-fetched audio blobs by path
  const prefetchCacheRef = useRef({});
  const prefetchingRef = useRef(null);

  const getAudioUrl = useCallback((item, stem = 'mix') => {
    if (!item) return null;
    let audioPath;
    if (stem !== 'mix' && item.stem_audio_paths?.[stem]) {
      audioPath = item.stem_audio_paths[stem];
    } else {
      audioPath = item.original_path || item.path;
    }
    return `${API_BASE}/audio?path=${encodeURIComponent(audioPath)}&format=opus`;
  }, []);

  // Prefetch the next item's audio
  const prefetchNext = useCallback(() => {
    if (!items.length) return;
    const nextIdx = currentIndex + 1;
    if (nextIdx >= items.length) return;
    const nextItem = items[nextIdx];
    const nextUrl = getAudioUrl(nextItem);
    if (!nextUrl || prefetchCacheRef.current[nextUrl]) return;
    if (prefetchingRef.current === nextUrl) return;
    prefetchingRef.current = nextUrl;

    fetch(nextUrl)
      .then(r => r.blob())
      .then(blob => {
        prefetchCacheRef.current[nextUrl] = blob;
        prefetchingRef.current = null;
      })
      .catch(() => { prefetchingRef.current = null; });
  }, [items, currentIndex, getAudioUrl]);

  const loadCurrentAudio = useCallback(() => {
    if (!currentItem) return;
    setIsLoading(true);

    const audioUrl = getAudioUrl(currentItem, selectedStem);

    // Check prefetch cache first
    const cached = prefetchCacheRef.current[audioUrl];
    if (cached) {
      const blobUrl = URL.createObjectURL(cached);
      wavesurferRef.current.load(blobUrl);
      delete prefetchCacheRef.current[audioUrl];
      // Start prefetching next
      prefetchNext();
    } else {
      wavesurferRef.current.load(audioUrl);
    }
  }, [currentItem, selectedStem, getAudioUrl, prefetchNext]);

  const togglePlayPause = useCallback(() => {
    if (wavesurferRef.current) {
      wavesurferRef.current.playPause();
    }
  }, []);

  const handleStop = useCallback(() => {
    if (wavesurferRef.current) {
      wavesurferRef.current.stop();
      setCurrentTime(0);
    }
  }, []);

  // Toggle label selection in multi-label mode
  const toggleLabelSelection = useCallback((label) => {
    setSelectedLabels(prev => {
      if (prev.includes(label)) {
        return prev.filter(l => l !== label);
      } else {
        return [...prev, label];
      }
    });
  }, []);

  // Toggle bleed instrument selection
  const toggleBleedInstrument = useCallback((instrument) => {
    setBleedInstruments(prev => {
      if (prev.includes(instrument)) {
        return prev.filter(i => i !== instrument);
      } else {
        return [...prev, instrument];
      }
    });
  }, []);

  // Toggle session instrument selection (for mix file multi-label)
  const toggleSessionInstrument = useCallback((instrument) => {
    setSelectedSessionInstruments(prev => {
      if (prev.includes(instrument)) {
        return prev.filter(i => i !== instrument);
      } else {
        return [...prev, instrument];
      }
    });
  }, []);

  // Save multi-instrument label for mix files (non-temporal)
  const saveMultiInstrumentLabel = useCallback(async () => {
    if (!currentItem || selectedSessionInstruments.length === 0) {
      setStatus('Select at least one instrument');
      setTimeout(() => setStatus(''), 2000);
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/label`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          path: currentItem.path,
          group: selectedSessionInstruments.length === 1 ? selectedSessionInstruments[0] : 'ensemble',
          multi_instruments: selectedSessionInstruments,
          is_multi_instrument: selectedSessionInstruments.length > 1,
        }),
      });

      if (response.ok) {
        setStatus(`Saved: ${selectedSessionInstruments.join(' + ')}`);
        // Mark as labeled and move to next
        setLabeledPaths(prev => new Set([...prev, currentItem.path]));
        setLabeledCount(prev => prev + 1);

        // Store for undo
        setLastLabeledItem({
          item: { ...currentItem },
          index: currentIndex,
          label: selectedSessionInstruments.join(' + '),
          tab: activeTab
        });

        // Move to next
        if (currentIndex < items.length - 1) {
          setCurrentIndex(prev => prev + 1);
        }
      } else {
        const data = await response.json();
        setStatus(`Error: ${data.message || 'Failed to save'}`);
      }
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
    setTimeout(() => setStatus(''), 3000);
  }, [currentItem, selectedSessionInstruments, currentIndex, items.length, activeTab]);

  // Save multi-label regions
  const saveMultiLabel = useCallback(async () => {
    if (!currentItem || regions.length === 0) {
      setStatus('No regions to save');
      setTimeout(() => setStatus(''), 2000);
      return;
    }

    const label = {
      path: currentItem.path,
      multi_label: true,
      regions: regions.map(r => ({
        start: r.start,
        end: r.end,
        labels: r.labels,
      })),
      duration: duration,
      source: activeTab === 'predictions' ? 'classifier_prediction' : 'classifier_flagged',
    };

    try {
      const response = await fetch(`${API_BASE}/label`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(label),
      });

      if (response.ok) {
        setStatus(`Saved ${regions.length} region(s)`);
        if (currentIndex < items.length - 1) {
          setCurrentIndex(prev => prev + 1);
        } else {
          setStatus('All items reviewed!');
        }
      } else {
        setStatus('Save failed');
      }
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }

    setTimeout(() => setStatus(''), 2000);
  }, [currentItem, regions, duration, currentIndex, items.length, activeTab]);

  // Single label confirm - saves the label to backend
  const handleConfirm = useCallback(async (group) => {
    if (!currentItem) return;

    // Use pending group if set, otherwise use provided group
    const finalGroup = group || pendingGroup;
    if (!finalGroup) return;

    // Determine final isolated status: override if set, otherwise use model prediction
    const finalIsIsolated = isolatedOverride !== null ? isolatedOverride : currentItem.is_isolated;

    const label = {
      path: currentItem.path,
      group: isRoomy ? `${finalGroup}_roomy` : finalGroup,
      previous_group: currentItem.current_label || currentItem.predicted_class,
      previous_subgroup: currentItem.subgroup || currentItem.original_subgroup,
      subgroup: selectedSubgroup || currentItem.subgroup || undefined,
      source: activeTab === 'predictions' ? 'classifier_prediction' : (activeTab === 'flagged' ? 'classifier_flagged' : 'manifest_review'),
      classifier: activeClassifier, // Route corrections to per-classifier file
      roomy: isRoomy,
      has_verb: hasVerb,
      is_midi: isMidi,
      has_bleed: hasBleed,
      bleed_instruments: hasBleed ? bleedInstruments : [],
      is_isolated: finalIsIsolated,
      is_isolated_override: isolatedOverride !== null, // Track if user manually set this
      ensemble_probability: currentItem.ensemble_probability || 0,
    };

    try {
      const response = await fetch(`${API_BASE}/label`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(label),
      });

      if (response.ok) {
        let labelText = isRoomy ? `${group} (roomy)` : group;
        if (hasBleed && bleedInstruments.length > 0) {
          labelText += ` + bleed: ${bleedInstruments.join(', ')}`;
        }
        setStatus(`Labeled as "${labelText}"`);

        // Store for undo before modifying state
        setLastLabeledItem({
          item: { ...currentItem },
          index: currentIndex,
          label: group,
          tab: activeTab
        });

        // Add to labeled paths
        setLabeledPaths(prev => new Set([...prev, currentItem.path]));
        setLabeledCount(prev => prev + 1);
        // Reset modifiers after labeling
        setIsRoomy(false);
        setHasVerb(false);
        setIsMidi(false);
        setHasBleed(false);
        setBleedConfirmed(false);
        setBleedInstruments([]);
        setIsolatedOverride(null);
        setSelectedSubgroup('');
        setPendingGroup('');

        if (activeTab === 'manifest') {
          // For manifest, mark as corrected and move to next
          setItems(prev => prev.map((item, i) =>
            i === currentIndex ? { ...item, is_corrected: true, current_label: group } : item
          ));
          if (currentIndex < items.length - 1) {
            setCurrentIndex(prev => prev + 1);
          } else {
            setStatus('All items reviewed!');
          }
        } else {
          // Remove the labeled item from the list by path (more reliable than index)
          const labeledPath = currentItem.path;
          const newItems = items.filter(item => item.path !== labeledPath);

          if (newItems.length === 0) {
            setItems([]);
            setStatus('All items reviewed!');
          } else {
            setItems(newItems);
            // Adjust index if we were at or past the end
            if (currentIndex >= newItems.length) {
              setCurrentIndex(newItems.length - 1);
            }
            // Index stays same - next item slides into this position
          }
        }
      } else {
        setStatus('Save failed');
      }
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }

    setTimeout(() => setStatus(''), 2000);
  }, [currentItem, currentIndex, items, activeTab, multiLabelMode, toggleLabelSelection, isRoomy, hasBleed, bleedInstruments, isolatedOverride, selectedSubgroup, pendingGroup]);

  // Handle group button click - sets pending if has subgroups, otherwise confirms
  const handleGroupClick = useCallback((group) => {
    if (!currentItem) return;

    // In multi-label mode, toggle selection instead
    if (multiLabelMode) {
      toggleLabelSelection(group);
      return;
    }

    // If bleed mode but not confirmed, toggle bleed instrument selection
    if (hasBleed && !bleedConfirmed) {
      toggleBleedInstrument(group);
      return;
    }
    // If bleed confirmed, continue to label selection (bleed instruments are locked in)

    // If group has subgroups and not already selected, set as pending
    if (SUBGROUPS[group]) {
      if (pendingGroup === group) {
        // Clicking same group again = confirm
        handleConfirm(group);
      } else {
        // Set as pending, show subgroup options
        setPendingGroup(group);
        // Pre-select existing subgroup if it belongs to this group
        const existingSubgroup = currentItem.current_subgroup || currentItem.predicted_subgroup || '';
        if (existingSubgroup && SUBGROUPS[group].includes(existingSubgroup)) {
          setSelectedSubgroup(existingSubgroup);
          setStatus(`Selected ${group} → ${existingSubgroup} - click Confirm or change subgroup`);
        } else {
          setSelectedSubgroup('');
          setStatus(`Selected ${group} - now pick a subgroup or click Confirm`);
        }
      }
    } else {
      // No subgroups, confirm immediately
      setPendingGroup('');
      handleConfirm(group);
    }
  }, [currentItem, multiLabelMode, hasBleed, bleedConfirmed, pendingGroup, toggleLabelSelection, toggleBleedInstrument, handleConfirm]);

  const handleSkip = useCallback(() => {
    if (currentIndex < items.length - 1) {
      setCurrentIndex(prev => prev + 1);
    }
  }, [currentIndex, items.length]);

  const handlePrevious = useCallback(() => {
    if (currentIndex > 0) {
      setCurrentIndex(prev => prev - 1);
    }
  }, [currentIndex]);

  // Undo last label
  const handleUndo = useCallback(async () => {
    if (!lastLabeledItem) {
      setStatus('Nothing to undo');
      setTimeout(() => setStatus(''), 2000);
      return;
    }

    setIsUndoing(true);
    try {
      // Call backend to delete the correction
      const response = await fetch(
        `${API_BASE}/correction?path=${encodeURIComponent(lastLabeledItem.item.path)}`,
        { method: 'DELETE' }
      );

      if (response.ok) {
        // Remove from labeled paths
        setLabeledPaths(prev => {
          const newSet = new Set(prev);
          newSet.delete(lastLabeledItem.item.path);
          return newSet;
        });
        setLabeledCount(prev => Math.max(0, prev - 1));

        // Restore item to list
        if (lastLabeledItem.tab !== 'manifest') {
          // For predictions/flagged, re-insert the item at its original position
          setItems(prev => {
            const newItems = [...prev];
            const insertIndex = Math.min(lastLabeledItem.index, newItems.length);
            newItems.splice(insertIndex, 0, lastLabeledItem.item);
            return newItems;
          });
          setCurrentIndex(lastLabeledItem.index);
        } else {
          // For manifest, just mark as uncorrected and go back
          setItems(prev => prev.map((item, i) =>
            item.path === lastLabeledItem.item.path
              ? { ...item, is_corrected: false }
              : item
          ));
          // Find the item's index and go to it
          const itemIndex = items.findIndex(item => item.path === lastLabeledItem.item.path);
          if (itemIndex >= 0) {
            setCurrentIndex(itemIndex);
          }
        }

        setStatus(`Undid label for "${lastLabeledItem.item.path?.split('/').pop()}"`);
        setLastLabeledItem(null);
      } else {
        const data = await response.json();
        setStatus(`Undo failed: ${data.message || 'Unknown error'}`);
      }
    } catch (err) {
      setStatus(`Undo error: ${err.message}`);
    }

    setIsUndoing(false);
    setTimeout(() => setStatus(''), 3000);
  }, [lastLabeledItem, items]);

  // Delete a region
  const deleteRegion = useCallback((regionId) => {
    setRegions(prev => prev.filter(r => r.id !== regionId));
    const allRegions = regionsRef.current.getRegions();
    const region = allRegions.find(r => r.id === regionId);
    if (region) {
      region.remove();
    }
  }, []);

  // Push corrections to manifest
  const pushCorrectionsToManifest = useCallback(async () => {
    setIsPushingCorrections(true);
    setPushResult(null);
    try {
      const response = await fetch(`${API_BASE}/push-corrections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const result = await response.json();
      if (result.status === 'ok') {
        setPushResult({ success: true, applied: result.applied, notFound: result.not_found });
        setStatus(`Pushed ${result.applied} corrections to manifest`);
      } else {
        setPushResult({ success: false, error: result.message });
        setStatus(`Push failed: ${result.message}`);
      }
    } catch (err) {
      setPushResult({ success: false, error: err.message });
      setStatus(`Push error: ${err.message}`);
    }
    setIsPushingCorrections(false);
    setTimeout(() => setStatus(''), 3000);
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT') return;

      if (e.code === 'Space') {
        e.preventDefault();
        togglePlayPause();
      } else if (e.code === 'Enter' && currentItem) {
        e.preventDefault();
        if (multiLabelMode && regions.length > 0) {
          saveMultiLabel();
        } else if (!multiLabelMode) {
          handleConfirm(currentItem.predicted_class || currentItem.current_label);
        }
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        handleSkip();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        handlePrevious();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [togglePlayPause, handleConfirm, handleSkip, handlePrevious, currentItem, multiLabelMode, regions, saveMultiLabel]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getConfidenceClass = (confidence) => {
    if (confidence >= 0.85) return 'high';
    if (confidence >= 0.65) return 'medium';
    return 'low';
  };

  const activeClassifierData = classifiers.find(c => c.id === activeClassifierId);

  return (
    <div className="audio-labeler">
      <div className="labeler-header">
        <h1>Audio Labeler</h1>
        <div className="header-actions">
          <button
            className={`push-corrections-btn ${isPushingCorrections ? 'pushing' : ''} ${pushResult?.success ? 'success' : ''}`}
            onClick={pushCorrectionsToManifest}
            disabled={isPushingCorrections}
          >
            {isPushingCorrections ? 'Pushing...' : 'Push Corrections'}
          </button>
          {pushResult && (
            <span className={`push-result ${pushResult.success ? 'success' : 'error'}`}>
              {pushResult.success ? `Applied ${pushResult.applied}` : pushResult.error}
            </span>
          )}
        </div>
      </div>

      {/* Main Tabs: Label vs Train */}
      <div className="main-tabs">
        <button
          className={`main-tab ${activeTab === 'label' ? 'active' : ''}`}
          onClick={() => setActiveTab('label')}
        >
          Label
        </button>
        <button
          className={`main-tab ${activeTab === 'train' ? 'active' : ''}`}
          onClick={() => setActiveTab('train')}
        >
          Training Dashboard
          {trainingStatus?.corrections_pending > 0 && (
            <span className="pending-badge">{trainingStatus.corrections_pending}</span>
          )}
        </button>
        <button
          className={`main-tab ${activeTab === 'midi' ? 'active' : ''}`}
          onClick={() => setActiveTab('midi')}
        >
          MIDI Corrections
        </button>
        <button
          className={`main-tab ${activeTab === 'stats' ? 'active' : ''}`}
          onClick={() => setActiveTab('stats')}
        >
          Dataset Stats
        </button>
      </div>

      {/* Training Dashboard */}
      {activeTab === 'train' && (
        <div className="training-dashboard">
          <div className="training-header">
            <h2>Classifier Training</h2>
            <p>Label corrections, retrain classifiers, view updated predictions</p>
            {trainingStatus?.corrections_pending > 0 && (
              <span className="corrections-total">{trainingStatus.corrections_pending} total corrections</span>
            )}
          </div>

          {/* Classifier Cards */}
          <div className="pipeline-diagram" style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
            {[
              { key: 'instrument', name: 'Group Classifier', desc: 'Instrument group: guitar, piano, drums, brass, strings, etc.', icon: '🎸' },
              { key: 'subgroup', name: 'Subgroup Classifier', desc: 'Subtypes: trumpet vs trombone, violin vs cello, etc.', icon: '🎺' },
              { key: 'mix_v3', name: 'Mix Classifier', desc: 'Temporal brass/strings/winds detection on mix recordings', icon: '🎵' },
              { key: 'stem_predictor', name: 'Stem Predictor', desc: 'Predicts 6 stems (drums/bass/vocals/guitar/piano/other) from mix latents', icon: '📊' },
            ].map(clf => {
              const info = trainingStatus?.classifiers?.[clf.key];
              return (
                <div key={clf.key} className={`classifier-card ${info?.has_model ? 'has-model' : 'no-model'}`} style={{ flex: '1 1 280px' }}>
                  <div className="clf-header">
                    <span className="clf-name">{clf.icon} {clf.name}</span>
                    <span className={`clf-status ${info?.has_model ? 'ready' : 'pending'}`}>
                      {info?.has_model ? '● Ready' : '○ No Model'}
                    </span>
                  </div>
                  <div className="clf-description">{clf.desc}</div>
                  <div className="clf-stats">
                    {info?.predictions_count > 0 && <span className="stat">{info.predictions_count.toLocaleString()} predictions</span>}
                    {info?.validation_accuracy && <span className="stat accuracy">{(info.validation_accuracy * 100).toFixed(1)}%</span>}
                    {info?.model_count > 0 && <span className="stat">{info.model_count} models</span>}
                    {info?.model_date && <span className="stat">Updated {new Date(info.model_date).toLocaleDateString()}</span>}
                  </div>
                  {info?.new_corrections > 0 && (
                    <div className="clf-stats" style={{ marginTop: '4px' }}>
                      <span className="stat pending" style={{ color: '#fbbf24', fontWeight: 'bold' }}>
                        {info.new_corrections} new corrections since last train
                      </span>
                    </div>
                  )}
                  <button className="retrain-btn" onClick={() => triggerRetrain(clf.key)} disabled={isTraining}>
                    {isTraining ? 'Training...' : 'Retrain'}
                  </button>
                </div>
              );
            })}
          </div>

          {trainingStatus?.training_history?.length > 0 && (
            <div className="training-history">
              <h3>Recent Training</h3>
              <div className="history-list">
                {trainingStatus.training_history.map((entry, idx) => (
                  <div key={idx} className={`history-item ${entry.status}`}>
                    <span className="history-clf">{entry.classifier}</span>
                    <span className="history-date">{new Date(entry.started_at).toLocaleString()}</span>
                    <span className={`history-status ${entry.status}`}>
                      {entry.status === 'running' && '⏳ '}
                      {entry.status === 'completed' && '✓ '}
                      {entry.status === 'failed' && '✗ '}
                      {entry.status}
                    </span>
                    {entry.validation_accuracy && (
                      <span className="history-accuracy">
                        {(entry.validation_accuracy * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Refresh Actions */}
          <div className="training-actions">
            <button
              className="refresh-manifest-btn"
              onClick={async () => {
                setStatus('Refreshing master manifest...');
                try {
                  const response = await fetch(`${API_BASE}/master-manifest/refresh`, { method: 'POST' });
                  const data = await response.json();
                  if (data.status === 'ok') {
                    setStatus(`Manifest refreshed: ${data.total?.toLocaleString()} entries`);
                  } else {
                    setStatus(`Refresh error: ${data.message}`);
                  }
                } catch (err) {
                  setStatus(`Refresh error: ${err.message}`);
                }
              }}
            >
              Refresh Manifest
            </button>
            <button
              className="refresh-status-btn"
              onClick={fetchTrainingStatus}
            >
              Refresh Status
            </button>
          </div>
        </div>
      )}

      {/* MIDI Corrections Interface */}
      {activeTab === 'midi' && (
        <>
          {/* Reuse data source selector */}
          <div className="data-nav">
            <div className="nav-row">
              <div className="nav-group">
                <label>Data Source:</label>
                <select
                  value={dataSource}
                  onChange={(e) => setDataSource(e.target.value)}
                  className="nav-dropdown"
                >
                  {DATA_SOURCES.map(ds => (
                    <option key={ds.id} value={ds.id}>{ds.name}</option>
                  ))}
                </select>
              </div>
              <div className="nav-group">
                <label>View:</label>
                <select
                  value={activeView}
                  onChange={(e) => setActiveView(e.target.value)}
                  className="nav-dropdown"
                >
                  <option value="all">All Entries</option>
                  <option value="has_midi">With MIDI Only</option>
                  {VIEWS.filter(v => v.id !== 'all').map(v => (
                    <option key={v.id} value={v.id}>{v.name}</option>
                  ))}
                </select>
              </div>
              <div className="nav-stats">
                <span className="stat-chip">{items.length} items</span>
                <span className="stat-chip">{items.filter(i => i.has_midi).length} with MIDI</span>
              </div>
            </div>
          </div>

          {/* Filters */}
          <div className="filter-bar">
            <div className="filter-group">
              <label>Group:</label>
              <select value={predictedGroupFilter} onChange={e => setPredictedGroupFilter(e.target.value)}>
                <option value="">All Groups</option>
                {predictedGroups.map(g => (
                  <option key={g} value={g}>{g}</option>
                ))}
              </select>
            </div>
            {availableSubgroups.length > 0 && (
              <div className="filter-group">
                <label>Subgroup:</label>
                <select value={subgroupFilter} onChange={e => setSubgroupFilter(e.target.value)}>
                  <option value="">All Subgroups</option>
                  {availableSubgroups.map(sg => (
                    <option key={sg} value={sg}>{sg}</option>
                  ))}
                </select>
              </div>
            )}
            <div className="filter-group">
              <label>Sort:</label>
              <select value={sortBy} onChange={e => setSortBy(e.target.value)}>
                <option value="">Default</option>
                <option value="group">By Group</option>
                <option value="subgroup">By Subgroup</option>
              </select>
            </div>
            <div className="progress-info">
              {items.length > 0 && (
                <span>{currentIndex + 1} / {items.length}</span>
              )}
            </div>
          </div>

          {/* Current Item Info */}
          {currentItem && (
            <div className="current-item-info">
              <div className="item-path">{currentItem.path?.split('/').pop()}</div>
              <div className="item-details">
                <span className={`current-label`}>
                  <strong>{currentItem.current_label || currentItem.group || 'unlabeled'}</strong>
                </span>
                {currentItem.subgroup && currentItem.subgroup !== 'undefined' && (
                  <span className="subgroup-badge">{currentItem.subgroup}</span>
                )}
                <span className={`midi-availability ${currentItem.has_midi ? 'has-midi' : 'no-midi'}`}>
                  {currentItem.has_midi ? 'MIDI available' : 'No MIDI'}
                </span>
              </div>
            </div>
          )}

          {/* Audio Player + Navigation */}
          {currentItem && (
            <div className="midi-tab-audio">
              <audio
                id="midi-tab-audio"
                src={`${API_BASE}/audio?path=${encodeURIComponent(currentItem.path)}`}
                controls
                style={{ width: '100%', height: '40px', borderRadius: '8px' }}
                onTimeUpdate={(e) => {
                  // Update time for playhead sync with MIDI grid
                  const audioEl = e.target;
                  setCurrentTime(audioEl.currentTime);
                  setDuration(audioEl.duration || 0);
                }}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onEnded={() => setIsPlaying(false)}
                autoPlay
              />
              <div className="transport-controls" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
                <button className="control-btn nav-btn" onClick={() => setCurrentIndex(prev => Math.max(0, prev - 1))} disabled={currentIndex === 0}>
                  Prev
                </button>
                <button className="control-btn nav-btn" onClick={() => setCurrentIndex(prev => Math.min(items.length - 1, prev + 1))} disabled={currentIndex >= items.length - 1}>
                  Next
                </button>
                <span className="progress-info">{currentIndex + 1} / {items.length}</span>
              </div>
            </div>
          )}

          {/* MIDI Correction Grid */}
          <MidiCorrectionGrid
            audioPath={currentItem?.path}
            isDrumMode={currentItem?.group === 'drums' || currentItem?.group === 'e-drums' || currentItem?.group === 'percussion'}
            currentTime={currentTime}
            duration={duration}
            isPlaying={isPlaying}
          />
        </>
      )}

      {/* Dataset Stats */}
      {activeTab === 'stats' && (
        <div className="dataset-stats-panel">
          <div className="stats-header">
            <h2>Dataset Statistics</h2>
            <div className="stats-manifest-selector">
              <label>Manifest:</label>
              <select
                value={statsManifest}
                onChange={(e) => setStatsManifest(e.target.value)}
                className="manifest-dropdown"
              >
                {availableManifests.length > 0 ? (
                  availableManifests.map(m => (
                    <option key={m.filename} value={m.filename}>
                      {m.filename.replace('.json', '')} ({m.size_mb}MB, {m.modified_date})
                    </option>
                  ))
                ) : (
                  <option value="consolidated_manifest.json">consolidated_manifest</option>
                )}
              </select>
              <button
                className="load-in-labeler-btn"
                title="Load this manifest in the labeler"
                onClick={() => {
                  setActiveTab('label');
                  setDataSource('consolidated');
                }}
              >Load in Labeler</button>
            </div>
            {datasetStats && (
              <div className="stats-meta">
                <span>{datasetStats.total_entries?.toLocaleString()} total entries</span>
                {datasetStats.generated_at && (
                  <span>Generated: {new Date(datasetStats.generated_at).toLocaleDateString()}</span>
                )}
                <button className="refresh-btn" onClick={() => {
                  setDatasetStats(null);
                  setStatsLoading(true);
                  fetch(`${API_BASE}/consolidated-manifest/stats?manifest=${encodeURIComponent(statsManifest)}`, { credentials: 'include' })
                    .then(r => r.json())
                    .then(data => { if (data.status === 'ok') setDatasetStats(data); })
                    .catch(e => console.error(e))
                    .finally(() => setStatsLoading(false));
                }}>Refresh</button>
              </div>
            )}
          </div>

          {statsLoading && <div className="stats-loading">Loading stats...</div>}

          {datasetStats && (
            <>
              {/* Summary cards */}
              <div className="stats-summary-cards">
                <div className="stat-card">
                  <div className="stat-value">{datasetStats.total_entries?.toLocaleString()}</div>
                  <div className="stat-label">Total Files</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{datasetStats.stats?.has_latent?.toLocaleString()}</div>
                  <div className="stat-label">Has Latent</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{datasetStats.stats?.mix_files?.toLocaleString()}</div>
                  <div className="stat-label">Mix Files</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{datasetStats.stats?.corrected?.toLocaleString()}</div>
                  <div className="stat-label">Corrected</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{datasetStats.stats?.flagged?.toLocaleString()}</div>
                  <div className="stat-label">Flagged</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{datasetStats.stats?.still_undefined?.toLocaleString()}</div>
                  <div className="stat-label">Still Undefined</div>
                </div>
              </div>

              {/* Group source breakdown */}
              {datasetStats.group_source_distribution && (
                <div className="stats-section">
                  <h3>Label Sources</h3>
                  <div className="stats-bar-chart">
                    {Object.entries(datasetStats.group_source_distribution)
                      .sort((a, b) => b[1] - a[1])
                      .map(([source, count]) => (
                        <div key={source} className="bar-row">
                          <span className="bar-label">{source}</span>
                          <div className="bar-track">
                            <div className="bar-fill" style={{
                              width: `${(count / datasetStats.total_entries) * 100}%`,
                              backgroundColor: source === 'filename' ? '#4a9eff' : source === 'classifier' ? '#ff9f43' : source === 'correction' ? '#2ed573' : '#aaa'
                            }} />
                          </div>
                          <span className="bar-count">{count.toLocaleString()}</span>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Group distribution with expandable subgroups */}
              <div className="stats-section">
                <h3>Group Distribution</h3>
                <div className="stats-group-table">
                  <div className="group-table-header">
                    <span>Group</span>
                    <span>Count</span>
                    <span>%</span>
                  </div>
                  {Object.entries(datasetStats.group_distribution || {})
                    .sort((a, b) => b[1] - a[1])
                    .map(([group, count]) => {
                      const pct = ((count / datasetStats.total_entries) * 100).toFixed(1);
                      const subgroups = datasetStats.subgroup_distribution?.[group];
                      const isExpanded = statsExpandedGroup === group;
                      return (
                        <div key={group}>
                          <div
                            className={`group-table-row ${subgroups ? 'expandable' : ''} ${isExpanded ? 'expanded' : ''}`}
                            onClick={() => subgroups && setStatsExpandedGroup(isExpanded ? null : group)}
                          >
                            <span className="group-name">
                              {subgroups && <span className="expand-icon">{isExpanded ? '\u25BC' : '\u25B6'}</span>}
                              {group}
                            </span>
                            <span className="group-count">{count.toLocaleString()}</span>
                            <span className="group-pct">{pct}%</span>
                          </div>
                          {isExpanded && subgroups && (
                            <div className="subgroup-rows">
                              {Object.entries(subgroups)
                                .sort((a, b) => b[1] - a[1])
                                .map(([sub, subCount]) => (
                                  <div key={sub} className="subgroup-row">
                                    <span className="subgroup-name">{sub}</span>
                                    <span className="subgroup-count">{subCount.toLocaleString()}</span>
                                    <span className="subgroup-pct">{((subCount / count) * 100).toFixed(1)}%</span>
                                  </div>
                                ))}
                            </div>
                          )}
                        </div>
                      );
                    })}
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Label Interface */}
      {activeTab === 'label' && (
        <>
          {/* Unified Data Source + View Selector */}
          <div className="data-nav">
            <div className="nav-row">
              <div className="nav-group">
                <label>Data Source:</label>
                <select
                  value={dataSource}
                  onChange={(e) => setDataSource(e.target.value)}
                  className="nav-dropdown"
                >
                  {DATA_SOURCES.map(ds => (
                    <option key={ds.id} value={ds.id}>{ds.name}</option>
                  ))}
                </select>
              </div>
              <div className="nav-group">
                <label>View:</label>
                <select
                  value={activeView}
                  onChange={(e) => setActiveView(e.target.value)}
                  className="nav-dropdown"
                >
                  {VIEWS.map(v => (
                    <option key={v.id} value={v.id}>{v.name}</option>
                  ))}
                </select>
              </div>
              <div className="nav-stats">
                <span className="stat-chip">{items.length} items</span>
                <span className="stat-chip">{labeledCount} labeled</span>
              </div>
              <button
                className={`file-list-toggle ${showFileList ? 'active' : ''}`}
                onClick={() => { if (!showFileList) { setFileListGroupFilter(predictedGroupFilter || ''); } setShowFileList(!showFileList); }}
              >File List</button>
            </div>
          </div>

          {/* File List Panel */}
          {showFileList && (
            <div className="file-list-panel">
              <div className="file-list-filters">
                <select
                  value={fileListGroupFilter}
                  onChange={(e) => {
                    const g = e.target.value;
                    setFileListGroupFilter(g);
                    setFileListSubgroupFilter('');
                    setFileListSourceFilter('');
                    setFileListVerifiedFilter('');
                    // Sync with main group filter to trigger refetch with full data
                    setPredictedGroupFilter(g);
                  }}
                >
                  <option value="">All Groups</option>
                  {(predictedGroups.length > 0 ? predictedGroups : [...new Set(items.map(i => i.group).filter(Boolean))].sort()).map(g => (
                    <option key={g} value={g}>{g}</option>
                  ))}
                </select>
                <select
                  value={fileListSubgroupFilter}
                  onChange={(e) => setFileListSubgroupFilter(e.target.value)}
                >
                  <option value="">All Subgroups</option>
                  {[...new Set(items
                    .filter(i => !fileListGroupFilter || i.group === fileListGroupFilter)
                    .map(i => i.subgroup || i.predicted_subgroup)
                    .filter(s => s && s !== 'undefined' && s !== 'unspecified')
                  )].sort().map(sg => (
                    <option key={sg} value={sg}>{sg}</option>
                  ))}
                </select>
                <select
                  value={fileListSourceFilter}
                  onChange={(e) => setFileListSourceFilter(e.target.value)}
                >
                  <option value="">All Sources</option>
                  {[...new Set(items
                    .filter(i => !fileListGroupFilter || i.group === fileListGroupFilter)
                    .map(i => i.group_source)
                    .filter(Boolean)
                  )].sort().map(s => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
                <select
                  value={fileListVerifiedFilter}
                  onChange={(e) => setFileListVerifiedFilter(e.target.value)}
                >
                  <option value="">All Verify</option>
                  <option value="verified">Filename Verified</option>
                  <option value="mismatch">Filename Mismatch</option>
                  <option value="none">No Filename Match</option>
                </select>
                <span className="file-list-count">
                  {items.filter(i =>
                    (!fileListGroupFilter || i.group === fileListGroupFilter) &&
                    (!fileListSubgroupFilter || (i.subgroup || i.predicted_subgroup) === fileListSubgroupFilter) &&
                    (!fileListSourceFilter || i.group_source === fileListSourceFilter) &&
                    (!fileListVerifiedFilter ||
                      (fileListVerifiedFilter === 'verified' && i.filename_verified) ||
                      (fileListVerifiedFilter === 'mismatch' && !i.filename_verified && i.filename_group) ||
                      (fileListVerifiedFilter === 'none' && !i.filename_verified && !i.filename_group))
                  ).length}{manifestTotal > items.length ? ` / ${manifestTotal.toLocaleString()} total` : ''} files
                </span>
                <button className="file-list-close" onClick={() => setShowFileList(false)}>×</button>
              </div>
              <div className="file-list-scroll">
                {items
                  .map((item, realIdx) => ({ item, realIdx }))
                  .filter(({ item }) =>
                    (!fileListGroupFilter || item.group === fileListGroupFilter) &&
                    (!fileListSubgroupFilter || (item.subgroup || item.predicted_subgroup) === fileListSubgroupFilter) &&
                    (!fileListSourceFilter || item.group_source === fileListSourceFilter) &&
                    (!fileListVerifiedFilter ||
                      (fileListVerifiedFilter === 'verified' && item.filename_verified) ||
                      (fileListVerifiedFilter === 'mismatch' && !item.filename_verified && item.filename_group) ||
                      (fileListVerifiedFilter === 'none' && !item.filename_verified && !item.filename_group))
                  )
                  .map(({ item, realIdx }) => {
                    const group = item.group || item.predicted_class || 'unknown';
                    const subgroup = item.subgroup || item.predicted_subgroup || '';
                    const groupSource = item.group_source || '';
                    const fname = item.filename || (item.path ? item.path.split('/').pop() : `item-${realIdx}`);
                    const color = GROUP_COLORS[group] || 'rgba(150,150,150,0.3)';
                    const solidColor = color.replace(/[\d.]+\)$/, '0.85)');
                    const sourceLabel = groupSource === 'filename' ? 'FN' : groupSource === 'classifier' ? 'ML' : groupSource === 'correction' ? 'LBL' : groupSource === 'none' ? '?' : '';
                    const sourceClass = groupSource === 'filename' ? 'src-filename' : groupSource === 'classifier' ? 'src-classifier' : groupSource === 'correction' ? 'src-correction' : 'src-none';
                    return (
                      <div
                        key={item.path || realIdx}
                        className={`file-list-row ${realIdx === currentIndex ? 'current' : ''}`}
                        onClick={() => {
                          setShowFileList(false);
                          // Delay index change so layout settles before waveform renders
                          setTimeout(() => setCurrentIndex(realIdx), 50);
                        }}
                      >
                        <span className="file-list-label" style={{ backgroundColor: color, borderLeft: `3px solid ${solidColor}` }}>
                          {group}{subgroup && subgroup !== 'undefined' ? ` / ${subgroup}` : ''}
                        </span>
                        {sourceLabel && <span className={`file-list-source ${sourceClass}`}>{sourceLabel}</span>}
                        {item.filename_verified !== undefined && (
                          <span className={`file-list-source ${item.filename_verified ? 'src-fn-verified' : item.filename_group ? 'src-fn-mismatch' : 'src-fn-none'}`}>
                            {item.filename_verified ? 'FV' : item.filename_group ? `FN:${item.filename_group}` : '--'}
                          </span>
                        )}
                        {item.classification && (
                          <span className={`file-list-source ${item.classification === 'fully_silent' ? 'src-silent-full' : item.classification === 'mostly_silent' ? 'src-silent-mostly' : 'src-silent-regions'}`}>
                            {item.classification === 'fully_silent' ? 'SILENT' : item.classification === 'mostly_silent' ? 'MOSTLY' : 'REGIONS'}
                            {item.silence_ratio != null ? ` ${Math.round(item.silence_ratio * 100)}%` : ''}
                          </span>
                        )}
                        <span className="file-list-name">{fname}</span>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {/* Compact Filter Bar */}
          <div className="filter-bar">
            <div className="filter-group">
              <label>Group:</label>
              <select value={predictedGroupFilter} onChange={e => setPredictedGroupFilter(e.target.value)}>
                <option value="">All Groups</option>
                {predictedGroups.map(g => (
                  <option key={g} value={g}>{g}</option>
                ))}
              </select>
            </div>
            {availableSubgroups.length > 0 && (
              <div className="filter-group">
                <label>Subgroup:</label>
                <select value={subgroupFilter} onChange={e => setSubgroupFilter(e.target.value)}>
                  <option value="">All Subgroups</option>
                  {availableSubgroups.map(sg => (
                    <option key={sg} value={sg}>{sg}</option>
                  ))}
                </select>
              </div>
            )}
            {/* Sort dropdown - always show for classifier data sources */}
            <div className="filter-group">
              <label>Sort:</label>
              <select value={sortBy} onChange={e => setSortBy(e.target.value)}>
                <option value="">Confidence ↑ (default)</option>
                <option value="confidence_desc">Confidence ↓</option>
                <option value="group">By Group</option>
                <option value="subgroup">By Subgroup</option>
                <option value="isolated_first">Isolated First</option>
                <option value="mix_first">Mix First</option>
                <option value="ensemble_prob_desc">Ensemble Prob ↓</option>
                <option value="ensemble_prob_asc">Ensemble Prob ↑</option>
              </select>
            </div>
            {(dataSource === 'master' || dataSource === 'consolidated' || dataSource === 'merged_manifest_v2') && (
              <div className="filter-group">
                <label>Source:</label>
                <select value={sourceFilter} onChange={e => setSourceFilter(e.target.value)}>
                  <option value="">All Sources</option>
                  <option value="original">Original (filename)</option>
                  <option value="classifier">Classifier</option>
                  <option value="manual">Corrections</option>
                </select>
              </div>
            )}
            {dataSource === 'silence' && (
              <div className="filter-group">
                <label>Classification:</label>
                <select value={sourceFilter} onChange={e => setSourceFilter(e.target.value)}>
                  <option value="">All Classifications</option>
                  <option value="fully_silent">Fully Silent</option>
                  <option value="mostly_silent">Mostly Silent</option>
                  <option value="noise_hiss">Noise / Hiss</option>
                  <option value="has_silent_regions">Has Silent Regions</option>
                </select>
              </div>
            )}
            {isolatedStats && (isolatedStats.isolated > 0 || isolatedStats.mix > 0) && (
              <div className="filter-group">
                <label>Mix/Isolated:</label>
                <select value={isolatedFilter} onChange={e => setIsolatedFilter(e.target.value)}>
                  <option value="">All ({isolatedStats.isolated + isolatedStats.mix + isolatedStats.unknown})</option>
                  <option value="isolated">Isolated ({isolatedStats.isolated})</option>
                  <option value="mix">Mix ({isolatedStats.mix})</option>
                  <option value="unknown">Unknown ({isolatedStats.unknown})</option>
                </select>
              </div>
            )}
            <div className="progress-info">
              {items.length > 0 && (
                <span>{currentIndex + 1} / {items.length}</span>
              )}
            </div>
          </div>

      {/* Current Item Info */}
      {currentItem && (
        <div className="current-item-info">
          <div className="item-path">{currentItem.path?.split('/').pop()}</div>
          <div className="item-details">
            {/* Universal label display - always show predicted vs current clearly */}
            <div className="label-comparison">
              {/* Current/Ground Truth Label (green) */}
              {(() => {
                const currentLabel = currentItem.current_subgroup || currentItem.current_label || currentItem.manifest_group;
                const hasCurrentLabel = currentLabel && currentLabel !== 'unknown' && currentLabel !== 'undefined';
                return (
                  <span className={`current-label ${!hasCurrentLabel ? 'unknown' : ''}`}>
                    <strong>{hasCurrentLabel ? currentLabel : 'unlabeled'}</strong>
                  </span>
                );
              })()}

              {/* Arrow separator when both exist */}
              {(currentItem.predicted_class || currentItem.predicted_subgroup || currentItem.predicted_labels?.length > 0) && (
                <span className="label-arrow">←</span>
              )}

              {/* Predicted/AI Label (blue) */}
              {currentItem.predicted_labels?.length > 1 ? (
                <span className="predicted-class multilabel-prediction">
                  <strong>{currentItem.predicted_labels.join(' + ')}</strong>
                  <span className="multilabel-indicator">multi</span>
                </span>
              ) : (currentItem.predicted_class || currentItem.predicted_subgroup) && (
                <span className={`predicted-class ${currentItem.is_mismatch ? 'mismatch' : ''}`}>
                  <strong>{currentItem.predicted_subgroup || currentItem.predicted_class}</strong>
                  {currentItem.is_mismatch && ' ⚠'}
                </span>
              )}
            </div>

            {/* Confidence badge */}
            {currentItem.confidence !== undefined && (
              <span className={`confidence-badge ${getConfidenceClass(currentItem.confidence)}`}>
                {Math.round((currentItem.confidence || 0) * 100)}%
              </span>
            )}

            {/* Subgroup display */}
            {dataSource === 'subgroups' && (
              <span className="group-badge">
                Group: {currentItem.group}
              </span>
            )}

            {/* Mix/Isolated status */}
            {currentItem.is_isolated !== undefined && currentItem.is_isolated !== null && (
              <span className={`isolated-badge ${currentItem.is_isolated ? 'isolated' : 'mix'}`}>
                {currentItem.is_isolated ? 'Isolated' : `Mix (${Math.round((currentItem.ensemble_probability || 0) * 100)}%)`}
              </span>
            )}

            {/* Multi-instrument badge */}
            {currentItem.is_multi && (
              <span className="multi-badge">
                Multi ({Math.round((currentItem.multi_probability || 0) * 100)}%)
              </span>
            )}

            {/* Source & original group for manifest data sources */}
            {(dataSource === 'master' || dataSource === 'consolidated') && (
              <>
                {currentItem.source && (
                  <span className={`source-badge source-${currentItem.source}`}>
                    {currentItem.source}
                  </span>
                )}
                {currentItem.original_group && currentItem.original_group !== currentItem.group && (
                  <span className="original-group-badge">
                    was: {currentItem.original_group}
                  </span>
                )}
              </>
            )}

            {/* Silence detection info */}
            {dataSource === 'silence' && currentItem.classification && (
              <>
                <span className={`source-badge ${currentItem.classification === 'fully_silent' ? 'source-silent-full' : currentItem.classification === 'mostly_silent' ? 'source-silent-mostly' : 'source-silent-regions'}`}>
                  {currentItem.classification.replace(/_/g, ' ')}
                </span>
                <span style={{color: '#aaa', fontSize: '0.75rem', marginLeft: '0.5rem'}}>
                  silence: {Math.round((currentItem.silence_ratio || 0) * 100)}% | energy: {(currentItem.energy_mean || 0).toFixed(3)} | regions: {currentItem.num_silent_regions || 0}
                </span>
                {currentItem.source && (
                  <span className={`source-badge source-${currentItem.source}`}>
                    {currentItem.source}
                  </span>
                )}
              </>
            )}

            {/* Merged manifest v2 specific info */}
            {activeClassifier === 'merged_manifest_v2' && (
              <>
                {currentItem.subgroup && currentItem.subgroup !== 'undefined' && (
                  <span className="subgroup-badge">
                    Sub: <strong>{currentItem.subgroup}</strong>
                    {currentItem.subgroup_source === 'classifier' && (
                      <span className="source-indicator classifier">
                        ({Math.round((currentItem.subgroup_confidence || 0) * 100)}%)
                      </span>
                    )}
                  </span>
                )}
                {currentItem.mix && (
                  <span className="mix-badge">
                    Mix ({currentItem.mix_source})
                  </span>
                )}
                {currentItem.group_source === 'classifier_correction' && (
                  <span className="flagged-correction-badge">
                    Flagged: {currentItem.flagged_original} → {currentItem.current_label}
                  </span>
                )}
              </>
            )}

            {/* Flagged tab specific */}
            {activeTab === 'flagged' && (
              <span className={`flag-badge flag-${currentItem.flag_type}`}>
                {currentItem.flag_type}
              </span>
            )}

            {/* Manifest tab specific attributes */}
            {activeTab === 'manifest' && (
              <>
                {currentItem.is_corrected && (
                  <span className="corrected-badge">✓ Corrected</span>
                )}
                {currentItem.roomy && (
                  <span className="attribute-badge roomy-badge">Roomy</span>
                )}
                {currentItem.has_bleed && (
                  <span className="attribute-badge bleed-badge">Bleed</span>
                )}
                {currentItem.multi_label && (
                  <span className="attribute-badge multilabel-badge">Multi</span>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {/* Session Instruments (for mix files) */}
      {currentItem && currentItem.session_instruments?.length > 0 && (
        <div className="session-instruments-panel">
          <div className="session-instruments-header">
            <span className="session-name">Session: {currentItem.session_name}</span>
            <span className="session-count">{selectedSessionInstruments.length} selected</span>
          </div>
          <div className="session-instruments-grid">
            {currentItem.session_instruments.map(inst => (
              <button
                key={inst}
                className={`session-inst-btn ${selectedSessionInstruments.includes(inst) ? 'selected' : ''}`}
                onClick={() => toggleSessionInstrument(inst)}
                style={{
                  backgroundColor: selectedSessionInstruments.includes(inst)
                    ? GROUP_COLORS[inst] || 'rgba(100, 149, 237, 0.3)'
                    : 'rgba(255, 255, 255, 0.1)'
                }}
              >
                {inst}
              </button>
            ))}
          </div>
          <div className="session-instruments-actions">
            <button
              className="save-multi-btn"
              onClick={saveMultiInstrumentLabel}
              disabled={selectedSessionInstruments.length === 0}
            >
              Save Multi-Instrument ({selectedSessionInstruments.length})
            </button>
            <button
              className="clear-session-btn"
              onClick={() => setSelectedSessionInstruments([])}
            >
              Clear
            </button>
            <button
              className="reset-session-btn"
              onClick={() => setSelectedSessionInstruments([...currentItem.session_instruments])}
            >
              Reset All
            </button>
          </div>
        </div>
      )}

      {/* Subgroup Editor Panel (for merged_manifest_v2) */}
      {currentItem && activeClassifier === 'merged_manifest_v2' && (
        <div className="subgroup-editor-panel">
          <div className="subgroup-editor-header">
            <span className="subgroup-title">Subgroup</span>
            <span className="subgroup-current">
              Current: <strong>{currentItem.subgroup || 'undefined'}</strong>
              {currentItem.subgroup_source && (
                <span className="subgroup-source">({currentItem.subgroup_source})</span>
              )}
            </span>
          </div>
          <div className="subgroup-selector">
            <select
              value={selectedSubgroup || currentItem.subgroup || ''}
              onChange={(e) => setSelectedSubgroup(e.target.value)}
              className="subgroup-dropdown"
            >
              <option value="">Keep current</option>
              <option value="undefined">undefined</option>
              {GROUPS.map(g => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
            {selectedSubgroup && selectedSubgroup !== currentItem.subgroup && (
              <span className="subgroup-changed">
                Will change to: <strong>{selectedSubgroup}</strong>
              </span>
            )}
          </div>
          {currentItem.original_group !== currentItem.current_label && (
            <div className="group-change-info">
              Group changed: {currentItem.original_group} → {currentItem.current_label}
              {currentItem.group_source && <span> ({currentItem.group_source})</span>}
            </div>
          )}
        </div>
      )}

      {/* Separated Stems Panel (for separated_stems classifier) */}
      {currentItem && currentItem.is_separated_stems && currentItem.stems?.length > 0 && (
        <div className="separated-stems-panel">
          <div className="stems-header">
            <span className="stems-title">
              {currentItem.is_temporal ? '🕐 Temporal Stem Analysis' : '🎛️ Stem Analysis'}
            </span>
            <div className="stems-mode-selector">
              {availableStemModes.normal && (
                <button
                  className={`mode-btn ${stemModeOption === 'normal' ? 'active' : ''}`}
                  onClick={() => setStemModeOption('normal')}
                >
                  Normal
                </button>
              )}
              {availableStemModes.temporal && (
                <button
                  className={`mode-btn ${stemModeOption === 'temporal' ? 'active' : ''}`}
                  onClick={() => setStemModeOption('temporal')}
                >
                  Temporal
                </button>
              )}
            </div>
          </div>
          <div className="stems-summary">
            <span className="detected-label">
              Detected: <strong>{currentItem.detected_instruments?.join(', ') || 'none'}</strong>
            </span>
            <span className="stems-stats">
              {currentItem.active_stems || 0} active / {currentItem.silent_stems || 0} silent
            </span>
          </div>

          {/* All stems grid */}
          <div className="stems-grid">
            {currentItem.stems.map(stem => {
              const isSilent = stem.is_silent;
              const isTemporal = currentItem.is_temporal;

              if (isTemporal) {
                // Temporal mode: show segments info
                const segments = stem.segments || [];
                const instruments = [...new Set(segments.map(s => s.instrument))];
                return (
                  <div
                    key={stem.stem}
                    className={`stem-item ${isSilent ? 'silent' : segments.length > 0 ? 'has-segments' : 'no-segments'}`}
                    style={{
                      borderLeftColor: instruments.length > 0
                        ? GROUP_COLORS[instruments[0]] || 'rgba(100, 149, 237, 0.5)'
                        : 'rgba(100, 100, 100, 0.3)'
                    }}
                  >
                    <span className="stem-name">{stem.stem}</span>
                    {isSilent ? (
                      <span className="stem-prediction silent-label">silent</span>
                    ) : (
                      <>
                        <span className="stem-prediction">{instruments.join(', ') || 'no detection'}</span>
                        <span className="stem-duration">
                          {stem.active_duration?.toFixed(1)}s / {stem.total_duration?.toFixed(1)}s
                        </span>
                      </>
                    )}
                  </div>
                );
              } else {
                // Normal mode: show single classification
                return (
                  <div
                    key={stem.stem}
                    className={`stem-item ${isSilent ? 'silent' : stem.confidence > 0.7 ? 'high-conf' : stem.confidence > 0.4 ? 'med-conf' : 'low-conf'}`}
                    style={{
                      borderLeftColor: isSilent
                        ? 'rgba(100, 100, 100, 0.3)'
                        : GROUP_COLORS[stem.predicted_group] || 'rgba(100, 149, 237, 0.5)'
                    }}
                  >
                    <span className="stem-name">{stem.stem}</span>
                    <span className="stem-prediction">{isSilent ? 'silent' : stem.predicted_group}</span>
                    {!isSilent && (
                      <span className="stem-confidence">{(stem.confidence * 100).toFixed(0)}%</span>
                    )}
                  </div>
                );
              }
            })}
          </div>

          {/* Timeline legend for temporal mode */}
          {currentItem.is_temporal && currentItem.timeline?.length > 0 && (
            <div className="timeline-legend">
              <span className="legend-title">Timeline regions shown on waveform</span>
              <span className="legend-hint">({currentItem.timeline.length} segments)</span>
            </div>
          )}
        </div>
      )}

      {/* Stem Selector for demucs_other mode */}
      {currentItem && activeClassifier === 'demucs_other' && (
        <div className="stem-selector-panel">
          <div className="stem-selector-header">
            <span className="stem-selector-title">🎛️ Audio Stem</span>
            <span className="stem-selector-hint">Select stem to play</span>
          </div>
          <div className="stem-buttons">
            {STEM_NAMES.map(stem => {
              const isAvailable = stem === 'mix' || currentItem.stem_audio_paths?.[stem];
              const isActive = selectedStem === stem;
              return (
                <button
                  key={stem}
                  className={`stem-btn ${isActive ? 'active' : ''} ${!isAvailable ? 'unavailable' : ''}`}
                  onClick={() => isAvailable && setSelectedStem(stem)}
                  disabled={!isAvailable}
                  title={isAvailable ? `Play ${stem}` : `${stem} not available`}
                >
                  {stem}
                  {stem === 'other' && <span className="stem-badge">🎯</span>}
                </button>
              );
            })}
          </div>
          {selectedStem !== 'mix' && (
            <div className="stem-status">
              Playing: <strong>{selectedStem}</strong> stem
            </div>
          )}
        </div>
      )}

      {/* Comparison Panel (for multilabel_comparison classifier) */}
      {currentItem && currentItem.is_comparison && (
        <div className="comparison-panel">
          <div className="comparison-header">
            <span className="comparison-title">Multi-Label Comparison</span>
            <span className="comparison-subtitle">Estimated (blue) vs Ground Truth (green)</span>
          </div>

          <div className="comparison-sections">
            {/* Estimated Section */}
            <div className="comparison-section estimated-section">
              <div className="section-header">
                <span className="section-icon">🤖</span>
                <span className="section-title">Estimated</span>
                <span className="section-count">{currentItem.estimated_regions?.length || 0} regions</span>
              </div>
              <div className="section-instruments">
                {(currentItem.estimated_instruments || []).map(inst => (
                  <span key={inst} className="instrument-tag" style={{ background: GROUP_COLORS[inst] }}>
                    {inst}
                  </span>
                ))}
                {(!currentItem.estimated_instruments || currentItem.estimated_instruments.length === 0) && (
                  <span className="no-instruments">No instruments detected</span>
                )}
              </div>
            </div>

            {/* GT Section */}
            <div className="comparison-section gt-section">
              <div className="section-header">
                <span className="section-icon">✓</span>
                <span className="section-title">Ground Truth</span>
                <span className="section-count">{currentItem.gt_regions?.length || 0} regions</span>
              </div>
              <div className="section-instruments">
                {(currentItem.gt_instruments || []).map(inst => (
                  <span key={inst} className="instrument-tag" style={{ background: GROUP_COLORS[inst] }}>
                    {inst}
                  </span>
                ))}
                {(!currentItem.gt_instruments || currentItem.gt_instruments.length === 0) && (
                  <span className="no-instruments">No GT labels</span>
                )}
              </div>
            </div>
          </div>

          {/* Legend */}
          <div className="comparison-legend">
            <span className="legend-item estimated-legend">■ Estimated (top track)</span>
            <span className="legend-item gt-legend">■ Ground Truth (bottom track)</span>
          </div>
        </div>
      )}

      {/* Waveform Display - Dual mode for comparison */}
      {isDualWaveformMode ? (
        <div className="dual-waveform-container">
          {(isLoading || isLoadingItems) && <div className="loading-overlay">Loading...</div>}
          {!currentItem && !isLoadingItems && (
            <div className="empty-state">
              {items.length === 0 ? 'No items to review' : 'Select an item to begin'}
            </div>
          )}
          {/* Predicted Waveform (Top) */}
          <div className="waveform-section predicted-waveform">
            <div className="waveform-label">
              <span className="waveform-icon">🤖</span>
              <span className="waveform-title">Model Predictions</span>
              <span className="waveform-instruments">
                {(currentItem?.predicted_instruments || []).map(inst => (
                  <span key={inst} className="instrument-chip" style={{ background: GROUP_COLORS[inst] }}>
                    {inst}
                  </span>
                ))}
              </span>
            </div>
            <div ref={waveformRef} className="waveform" />
          </div>
          {/* GT Waveform (Bottom) */}
          <div className="waveform-section gt-waveform">
            <div className="waveform-label">
              <span className="waveform-icon">✓</span>
              <span className="waveform-title">Ground Truth</span>
              <span className="waveform-instruments">
                {(currentItem?.gt_instruments || []).map(inst => (
                  <span key={inst} className="instrument-chip" style={{ background: GROUP_COLORS[inst] }}>
                    {inst}
                  </span>
                ))}
              </span>
            </div>
            <div ref={gtWaveformRef} className="waveform" />
          </div>
          {/* Region counts comparison */}
          <div className="dual-waveform-stats">
            <span className="stat-item pred-stat">
              {currentItem?.predicted_regions?.length || 0} predicted regions
            </span>
            <span className="stat-item gt-stat">
              {currentItem?.gt_regions?.length || 0} GT regions
            </span>
          </div>
        </div>
      ) : (
        <div className={`waveform-container ${multiLabelMode ? 'multi-label-active' : ''}`}>
          {(isLoading || isLoadingItems) && <div className="loading-overlay">Loading...</div>}
          {!currentItem && !isLoadingItems && (
            <div className="empty-state">
              {items.length === 0 ? 'No items to review' : 'Select an item to begin'}
            </div>
          )}
          <div ref={waveformRef} className="waveform" />
          {multiLabelMode && selectedLabels.length > 0 && (
            <div className="drag-hint">Drag on waveform to create region with: {selectedLabels.join(', ')}</div>
          )}
        </div>
      )}

      {/* Regions List (multi-label mode) */}
      {multiLabelMode && regions.length > 0 && (
        <div className="regions-list">
          <h4>{currentItem?.is_comparison ? 'Comparison Regions:' : 'Labeled Regions:'}</h4>
          {regions.map((region, idx) => (
            <div
              key={region.id}
              className={`region-item ${region.isEstimated ? 'estimated-region' : ''} ${region.isGT ? 'gt-region' : ''} ${region.isTimeline ? 'timeline-region' : ''}`}
            >
              <span className="region-type-badge">
                {region.isEstimated ? '🤖' : region.isGT ? '✓' : ''}
              </span>
              <span className="region-time">
                {formatTime(region.start)} - {formatTime(region.end)}
              </span>
              <span className="region-labels">
                {region.labels.map(l => (
                  <span key={l} className="region-label-tag" style={{ background: GROUP_COLORS[l] }}>
                    {l}
                    {region.confidences && region.confidences[l] && (
                      <span className="confidence-badge">{(region.confidences[l] * 100).toFixed(0)}%</span>
                    )}
                  </span>
                ))}
              </span>
              {!region.isEstimated && !region.isGT && !region.isTimeline && (
                <button className="region-delete" onClick={() => deleteRegion(region.id)}>×</button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Playback Controls */}
      <div className="playback-controls">
        <button
          className={`control-btn undo-btn ${lastLabeledItem ? 'has-undo' : ''}`}
          onClick={handleUndo}
          disabled={!lastLabeledItem || isUndoing}
          title={lastLabeledItem ? `Undo: ${lastLabeledItem.item?.path?.split('/').pop()}` : 'Nothing to undo'}
        >
          {isUndoing ? '...' : '↩ Undo'}
        </button>
        <button className="control-btn nav-btn" onClick={handlePrevious} disabled={currentIndex === 0}>
          ← Prev
        </button>
        <button
          className={`control-btn ${isPlaying ? 'playing' : ''}`}
          onClick={togglePlayPause}
          disabled={!currentItem}
        >
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>
        <button className="control-btn" onClick={handleStop} disabled={!currentItem}>
          ⏹ Stop
        </button>
        <button className="control-btn nav-btn" onClick={handleSkip} disabled={currentIndex >= items.length - 1}>
          Skip →
        </button>
        <div className="time-display">
          <span>{formatTime(currentTime)}</span>
          <span> / </span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {/* Mode Toggles */}
      <div className="mode-toggles">
        <label className="toggle-label">
          <input
            type="checkbox"
            checked={multiLabelMode}
            onChange={(e) => {
              setMultiLabelMode(e.target.checked);
              if (!e.target.checked) {
                setSelectedLabels([]);
                if (dragSelectionCleanupRef.current) {
                  dragSelectionCleanupRef.current();
                  dragSelectionCleanupRef.current = null;
                }
                regionsRef.current?.clearRegions();
                setRegions([]);
              }
            }}
          />
          <span className="toggle-text">Multi-label</span>
        </label>
        <label className="toggle-label gt-toggle">
          <input
            type="checkbox"
            checked={gtCompareEnabled}
            onChange={(e) => {
              setGtCompareEnabled(e.target.checked);
              setIsDualWaveformMode(e.target.checked);
            }}
          />
          <span className="toggle-text">GT Compare</span>
          <span className="toggle-hint">(show ground truth below)</span>
        </label>
        {multiLabelMode && regions.length > 0 && (
          <button className="save-regions-btn" onClick={saveMultiLabel}>
            Save {regions.length} Region(s)
          </button>
        )}
      </div>

      {/* Unified Label Section */}
      <div className="label-section">
        {/* Header with mode toggles */}
        <div className="label-header">
          <h3>{multiLabelMode ? 'Select Instruments for Region:' : (hasBleed && !bleedConfirmed) ? 'Select Bleed Instruments:' : (hasBleed && bleedConfirmed) ? 'Select Main Label (with bleed):' : 'Confirm or Change Label:'}</h3>
          <div className="label-modes">
            <label className="mode-toggle">
              <input
                type="checkbox"
                checked={isRoomy}
                onChange={(e) => setIsRoomy(e.target.checked)}
              />
              <span className="mode-text">Roomy</span>
            </label>
            <label className="mode-toggle">
              <input
                type="checkbox"
                checked={hasVerb}
                onChange={(e) => setHasVerb(e.target.checked)}
              />
              <span className="mode-text">Verb</span>
            </label>
            <label className="mode-toggle">
              <input
                type="checkbox"
                checked={isMidi}
                onChange={(e) => setIsMidi(e.target.checked)}
              />
              <span className="mode-text">MIDI</span>
            </label>
            <label className="mode-toggle bleed-toggle">
              <input
                type="checkbox"
                checked={hasBleed}
                onChange={(e) => {
                  setHasBleed(e.target.checked);
                  if (!e.target.checked) {
                    setBleedInstruments([]);
                    setBleedConfirmed(false);
                  }
                }}
              />
              <span className="mode-text">+ Bleed</span>
            </label>
            {hasBleed && bleedInstruments.length > 0 && (
              <>
                <span className="bleed-summary-inline">
                  Bleed: {bleedInstruments.join(', ')}
                  <button className="clear-bleed" onClick={() => { setBleedInstruments([]); setBleedConfirmed(false); }}>×</button>
                </span>
                <label className={`mode-toggle confirm-bleed-toggle ${bleedConfirmed ? 'confirmed' : ''}`}>
                  <input
                    type="checkbox"
                    checked={bleedConfirmed}
                    onChange={(e) => setBleedConfirmed(e.target.checked)}
                  />
                  <span className="mode-text">{bleedConfirmed ? '✓ Bleed Locked' : 'Lock Bleed'}</span>
                </label>
              </>
            )}
            {/* Isolated/Mix Override - shows model prediction with option to correct */}
            <div className="isolated-toggle-group">
              <span className={`model-prediction ${currentItem?.is_isolated ? 'isolated' : currentItem?.is_isolated === false ? 'mix' : 'unknown'}`}>
                Model: {currentItem?.is_isolated ? 'Isolated' : currentItem?.is_isolated === false ? `Mix (${Math.round((currentItem?.ensemble_probability || 0) * 100)}%)` : '?'}
              </span>
              <div className="isolated-override-btns">
                <button
                  className={`override-btn ${isolatedOverride === true ? 'active isolated' : ''}`}
                  onClick={() => setIsolatedOverride(isolatedOverride === true ? null : true)}
                  title="Override: Mark as Isolated"
                >
                  Isolated
                </button>
                <button
                  className={`override-btn ${isolatedOverride === false ? 'active mix' : ''}`}
                  onClick={() => setIsolatedOverride(isolatedOverride === false ? null : false)}
                  title="Override: Mark as Mix"
                >
                  Mix
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Label Buttons - unified for both label and bleed selection */}
        <div className="label-buttons">
          {GROUPS.map(group => (
            <button
              key={group}
              className={`label-btn
                ${currentItem?.predicted_class === group ? 'predicted' : ''}
                ${currentItem?.current_label === group ? 'current' : ''}
                ${multiLabelMode && selectedLabels.includes(group) ? 'selected-multi' : ''}
                ${hasBleed && bleedInstruments.includes(group) ? 'bleed-selected' : ''}
                ${pendingGroup === group ? 'pending' : ''}`}
              onClick={() => handleGroupClick(group)}
              disabled={!currentItem && !hasBleed}
            >
              {group}
              {currentItem?.predicted_class === group && !multiLabelMode && !hasBleed && <span className="badge">pred</span>}
              {currentItem?.current_label === group && activeTab === 'flagged' && !multiLabelMode && !hasBleed && <span className="badge current">curr</span>}
              {multiLabelMode && selectedLabels.includes(group) && <span className="badge selected">✓</span>}
              {hasBleed && bleedInstruments.includes(group) && <span className="badge bleed">bleed</span>}
              {pendingGroup === group && <span className="badge pending">●</span>}
            </button>
          ))}
        </div>

        {/* Subgroup Buttons - show based on pending group first, then current item's group */}
        {currentItem && (SUBGROUPS[pendingGroup] || SUBGROUPS[currentItem.current_label] || SUBGROUPS[currentItem.predicted_class] || SUBGROUPS[currentItem.group]) && (
          <div className="subgroup-buttons">
            <span className="subgroup-label">Subgroup:</span>
            {(SUBGROUPS[pendingGroup] || SUBGROUPS[currentItem.current_label] || SUBGROUPS[currentItem.predicted_class] || SUBGROUPS[currentItem.group]).map(sub => (
              <button
                key={sub}
                className={`label-btn subgroup-btn
                  ${selectedSubgroup === sub ? 'selected' : ''}
                  ${currentItem.current_subgroup === sub ? 'current' : ''}
                  ${currentItem.predicted_subgroup === sub ? 'predicted' : ''}`}
                onClick={() => setSelectedSubgroup(selectedSubgroup === sub ? '' : sub)}
              >
                {sub.replace('_', ' ')}
                {currentItem.predicted_subgroup === sub && <span className="badge">pred</span>}
                {currentItem.current_subgroup === sub && <span className="badge current">curr</span>}
              </button>
            ))}
            {/* Confirm button when group is pending */}
            {pendingGroup && (
              <button
                className="label-btn confirm-btn"
                onClick={() => handleConfirm(pendingGroup)}
              >
                ✓ Confirm {pendingGroup}{selectedSubgroup ? ` → ${selectedSubgroup}` : ''}
              </button>
            )}
          </div>
        )}

        {/* Special Tags */}
        <div className="special-tags">
          {SPECIAL_TAGS.map(tag => (
            <button
              key={tag}
              className={`label-btn special-tag ${tag} ${multiLabelMode && selectedLabels.includes(tag) ? 'selected-multi' : ''}`}
              onClick={() => handleConfirm(tag)}
              disabled={!currentItem || (hasBleed && !bleedConfirmed)}
            >
              {tag}
              {multiLabelMode && selectedLabels.includes(tag) && <span className="badge selected">✓</span>}
            </button>
          ))}
        </div>

        {/* Submit hint when bleed is selected */}
        {hasBleed && bleedInstruments.length > 0 && !bleedConfirmed && (
          <div className="bleed-hint">
            Select bleed instruments, then check "Lock Bleed" to confirm main label.
          </div>
        )}
        {hasBleed && bleedConfirmed && (
          <div className="bleed-hint confirmed">
            Bleed locked ({bleedInstruments.join(', ')}). Now click a main instrument to confirm.
          </div>
        )}
      </div>

      {/* Status Message */}
      {status && <div className="status-message">{status}</div>}

      {/* Keyboard Shortcuts */}
      <div className="shortcuts">
        <span><kbd>Space</kbd> Play/Pause</span>
        <span><kbd>Enter</kbd> {multiLabelMode ? 'Save Regions' : 'Confirm & Next'}</span>
        <span><kbd>←</kbd><kbd>→</kbd> Navigate</span>
      </div>
        </>
      )}
    </div>
  );
};

export default AudioLabeler;
