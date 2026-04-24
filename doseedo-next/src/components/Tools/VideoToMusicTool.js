import React, { useState, useRef, useCallback } from 'react';
import {
  uploadVideo,
  pollVideoUntilComplete,
  collapseSceneChanges,
  computeBestTempos,
} from '../../services/videoAPI';
import { startGeneration, pollUntilComplete } from '../../services/generationAPI';
import ToolWaveform from './ToolWaveform';
import { C, P, Ic, ToolShell, Panel, DropZone, FieldLabel } from './toolShell';

const INSTRUMENT_GROUPS = [
  { id: 'piano', label: 'Piano' },
  { id: 'guitar', label: 'Guitar' },
  { id: 'strings', label: 'Strings' },
  { id: 'brass', label: 'Brass' },
  { id: 'electronic', label: 'Electronic' },
];

/**
 * Video to Music — uploads a video, detects scenes, then calls
 * startGeneration with scene durations + tempos as conditioning. Backend
 * wiring (uploadVideo, pollVideoUntilComplete, startGeneration,
 * pollUntilComplete) preserved.
 */
const VideoToMusicTool = ({ tool, onBack }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [sceneData, setSceneData] = useState(null);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [progress, setProgress] = useState(0);

  const [useWhisper, setUseWhisper] = useState(false);
  const [instrumentGroup, setInstrumentGroup] = useState('strings');
  const videoRef = useRef(null);

  const handleFileUpload = useCallback((file) => {
    if (!file) return;
    setUploadedFile(file);
    setVideoPreviewUrl(URL.createObjectURL(file));
    setSceneData(null);
    setGeneratedAudioUrl(null);
    setStatusMessage(`Loaded: ${file.name}`);
  }, []);

  const handleRemove = useCallback(() => {
    setUploadedFile(null);
    setVideoPreviewUrl(null);
    setSceneData(null);
    setGeneratedAudioUrl(null);
    setStatusMessage('');
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!uploadedFile) {
      setStatusMessage('Upload a video first.');
      return;
    }
    setIsProcessing(true);
    setStatusMessage('Uploading video for analysis…');
    setProgress(0);
    try {
      const uploadResult = await uploadVideo(uploadedFile, useWhisper);
      const taskId = uploadResult.task_id;
      setStatusMessage('Detecting scenes…');
      const result = await pollVideoUntilComplete(
        taskId,
        (progressData) => {
          setProgress(progressData.attempts / 200);
          setStatusMessage(`Analyzing video… (${progressData.attempts}s)`);
        },
        3000,
        600,
      );
      if (result.scene_changes) {
        const collapsedScenes = collapseSceneChanges(result.scene_changes, 3);
        const tempos = computeBestTempos(collapsedScenes);
        setSceneData({
          sceneChanges: collapsedScenes,
          tempos,
          duration: result.video_duration,
          audioUrl: result.audio_url,
        });
        setStatusMessage(`Found ${collapsedScenes.length} scenes. Ready to generate.`);
      } else {
        setStatusMessage('Analysis completed but no scenes detected.');
      }
    } catch (err) {
      console.error('Video analysis failed:', err);
      setStatusMessage(`Error: ${err.message}`);
    } finally {
      setIsProcessing(false);
      setProgress(0);
    }
  }, [uploadedFile, useWhisper]);

  const handleGenerate = useCallback(async () => {
    if (!sceneData) {
      setStatusMessage('Analyze the video first.');
      return;
    }
    setIsGenerating(true);
    setStatusMessage('Generating music for video…');
    setProgress(0);
    try {
      const params = {
        instrumentGroup,
        instrumentSubgroup:
          instrumentGroup === 'strings' ? 'ensemble_strings' : `${instrumentGroup}_main`,
        monophonicMode: true,
        arrangeMode: true,
        sceneDurations: sceneData.sceneChanges,
        sceneTempos: sceneData.tempos,
        steps: 100,
        seed: Math.floor(Math.random() * 1000000),
      };
      let audioFile = null;
      if (sceneData.audioUrl) {
        const response = await fetch(sceneData.audioUrl);
        const blob = await response.blob();
        audioFile = new File([blob], 'video_audio.wav', { type: 'audio/wav' });
      }
      const startResult = await startGeneration(params, audioFile);
      const taskId = startResult.task_id;
      const result = await pollUntilComplete(
        taskId,
        (progressData) => {
          setProgress(progressData.progress || 0);
          setStatusMessage(`Generating… ${Math.round((progressData.progress || 0) * 100)}%`);
        },
        null,
        1800,
      );
      if (result.file_paths && result.file_paths.length > 0) {
        setGeneratedAudioUrl(result.file_paths[0]);
        setStatusMessage('Music generation complete.');
      } else {
        setStatusMessage('Completed but no audio returned.');
      }
    } catch (err) {
      console.error('Music generation failed:', err);
      setStatusMessage(`Error: ${err.message}`);
    } finally {
      setIsGenerating(false);
      setProgress(0);
    }
  }, [sceneData, instrumentGroup]);

  const handleDownload = useCallback(() => {
    if (!generatedAudioUrl) return;
    const a = document.createElement('a');
    a.href = generatedAudioUrl;
    a.download = `video_music_${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [generatedAudioUrl]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // ---------- Input ----------
  const InputPanel = (
    <Panel
      title="Input · video"
      marker="◆"
      status={
        uploadedFile ? (
          <span style={{ color: C.ok, display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: C.ok }} />
            loaded
          </span>
        ) : (
          <span style={{ color: C.inkMute }}>empty</span>
        )
      }
    >
      <DropZone
        file={uploadedFile}
        onPick={handleFileUpload}
        onRemove={handleRemove}
        accept="video/*"
        hint="MP4 · MOV · AVI"
        icon="VID"
      />
      {videoPreviewUrl && (
        <div style={{ padding: '0 16px 14px' }}>
          <FieldLabel style={{ marginBottom: 8 }}>Preview</FieldLabel>
          <div
            style={{
              background: '#000',
              border: `1px solid ${C.rule}`,
              position: 'relative',
              aspectRatio: '16 / 9',
            }}
          >
            <video
              ref={videoRef}
              src={videoPreviewUrl}
              controls
              muted
              style={{ width: '100%', height: '100%', display: 'block' }}
            />
          </div>
        </div>
      )}
      <div
        style={{
          padding: '10px 16px 14px',
          borderTop: `1px solid ${C.rule}`,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <input
          id="useWhisper"
          type="checkbox"
          checked={useWhisper}
          onChange={(e) => setUseWhisper(e.target.checked)}
        />
        <label
          htmlFor="useWhisper"
          style={{
            fontFamily: C.mono,
            fontSize: 10,
            letterSpacing: 0.5,
            color: C.inkSoft,
            textTransform: 'uppercase',
          }}
        >
          Use Whisper for speech transcription
        </label>
      </div>
    </Panel>
  );

  // ---------- Controls ----------
  const ControlsPanel = (
    <Panel title="Controls · scenes & style" marker="◇">
      <div style={{ padding: '14px 16px 0' }}>
        <FieldLabel style={{ marginBottom: 8 }}>Scene detection</FieldLabel>
        <button
          type="button"
          onClick={handleAnalyze}
          disabled={!uploadedFile || isProcessing || isGenerating}
          style={{
            width: '100%',
            padding: '9px 12px',
            background: sceneData ? C.bg : C.ink,
            color: sceneData ? C.ink : C.bg,
            border: `1px solid ${sceneData ? C.rule : C.ink}`,
            fontFamily: C.mono,
            fontSize: 10,
            fontWeight: 600,
            letterSpacing: 0.8,
            textTransform: 'uppercase',
            cursor: !uploadedFile || isProcessing ? 'not-allowed' : 'pointer',
            opacity: !uploadedFile ? 0.5 : 1,
          }}
        >
          {isProcessing ? 'Analyzing…' : sceneData ? 'Re-analyze' : 'Analyze video'}
        </button>
      </div>

      {sceneData && (
        <div style={{ padding: '12px 16px' }}>
          <FieldLabel style={{ marginBottom: 8 }}>
            {sceneData.sceneChanges.length} scenes · {formatTime(sceneData.duration)}
          </FieldLabel>
          <div
            style={{
              position: 'relative',
              height: 36,
              background: C.bg,
              border: `1px solid ${C.rule}`,
            }}
          >
            {sceneData.sceneChanges.map((t, i) => (
              <div
                key={i}
                title={`Scene ${i + 1}: ${formatTime(t)} · ${sceneData.tempos[i] || 120} BPM`}
                style={{
                  position: 'absolute',
                  left: `${(t / sceneData.duration) * 100}%`,
                  top: 4,
                  bottom: 4,
                  width: 2,
                  background: C.purple,
                }}
              />
            ))}
          </div>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              marginTop: 6,
              fontFamily: C.mono,
              fontSize: 9,
              color: C.inkMute,
              letterSpacing: 0.4,
              textTransform: 'uppercase',
            }}
          >
            <span>Avg BPM</span>
            <span style={{ color: C.ink }}>
              {Math.round(
                sceneData.tempos.reduce((a, b) => a + b, 0) / (sceneData.tempos.length || 1),
              )}
            </span>
          </div>
        </div>
      )}

      <div style={{ padding: '12px 16px 14px', borderTop: `1px solid ${C.rule}`, marginTop: 'auto' }}>
        <FieldLabel style={{ marginBottom: 8 }}>Music style</FieldLabel>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 4 }}>
          {INSTRUMENT_GROUPS.map((g) => {
            const active = instrumentGroup === g.id;
            return (
              <button
                key={g.id}
                type="button"
                onClick={() => setInstrumentGroup(g.id)}
                style={{
                  padding: '8px 10px',
                  background: active ? C.ink : C.bg,
                  color: active ? C.bg : C.ink,
                  border: `1px solid ${active ? C.ink : C.rule}`,
                  fontFamily: C.mono,
                  fontSize: 10,
                  letterSpacing: 0.6,
                  textTransform: 'uppercase',
                  cursor: 'pointer',
                }}
              >
                {g.label}
              </button>
            );
          })}
        </div>
      </div>
    </Panel>
  );

  // ---------- Output ----------
  const OutputPanel = (
    <Panel
      title="Output · scored audio"
      marker="●"
      status={generatedAudioUrl ? <span style={{ color: C.ok }}>ready</span> : null}
    >
      <div style={{ padding: 16 }}>
        <div style={{ background: C.bg, border: `1px solid ${C.rule}`, padding: 8 }}>
          <FieldLabel style={{ padding: '0 0 8px' }}>Preview</FieldLabel>
          <ToolWaveform audioUrl={generatedAudioUrl} height={120} color={C.accent} />
        </div>
      </div>
      <div style={{ flex: 1 }} />
      {generatedAudioUrl && (
        <div style={{ padding: 14, borderTop: `1px solid ${C.rule}` }}>
          <button
            type="button"
            onClick={handleDownload}
            style={{
              width: '100%',
              padding: '10px 12px',
              background: C.ink,
              color: C.bg,
              border: 'none',
              fontFamily: C.mono,
              fontSize: 10,
              fontWeight: 600,
              letterSpacing: 0.8,
              textTransform: 'uppercase',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
              cursor: 'pointer',
            }}
          >
            <Ic d={P.dl} size={12} color={C.bg} /> Download score
          </button>
        </div>
      )}
    </Panel>
  );

  const running = isProcessing || isGenerating;
  return (
    <ToolShell
      tool={{ ...tool, sku: 'T-02', category: 'Video tool', version: 'v1.3.0' }}
      subtitle={sceneData ? `${sceneData.sceneChanges.length} scenes` : 'scene-aware scoring'}
      description="Drop a clip. We detect scene cuts, pick per-scene tempos, then score it with the instrument family you choose."
      meta={[{ k: 'Avg time', v: '~90s per minute' }]}
      running={running}
      progress={progress}
      statusMessage={statusMessage}
      primaryLabel="Generate music"
      onPrimary={handleGenerate}
      primaryDisabled={!sceneData || isProcessing}
      onBack={onBack}
      left={InputPanel}
      center={ControlsPanel}
      right={OutputPanel}
    />
  );
};

export default VideoToMusicTool;
