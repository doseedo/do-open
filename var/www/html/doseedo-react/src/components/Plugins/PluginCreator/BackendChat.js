import React, { useState, useRef, useEffect, useCallback } from 'react';
import { sendCodegenChatMessage, generatePluginCode } from '../../../services/chatAPI';
import styles from './PluginCreator.module.css';

const DSP_SYSTEM_PROMPT = `You are a DSP engineer and JUCE C++ plugin developer. You design audio plugin DSP architectures using a structured "DSP Language" format.

When a user describes a plugin, output the complete DSP specification inside a \`\`\`dsplang code block. This is a JSON object:

{
  "pluginType": "effect",
  "name": "Plugin Name",
  "parameters": [
    { "id": "param_id", "name": "Display Name", "min": 0, "max": 1, "default": 0.5, "skew": 1.0, "unit": "" }
  ],
  "dspChain": [
    { "type": "node_type", "id": "unique_id", "params": { "param_name": "@param_id_or_literal" } }
  ],
  "routing": { "input": "stereo", "chain": ["node_id1", "node_id2"], "output": "stereo" },
  "mode": "replace"
}

PLUGIN TYPES:
- "effect": Audio effect (receives audio input). acceptsMidi = false.
- "instrument": MIDI instrument (generates audio from MIDI). acceptsMidi = true. Add "midi" field.
- "midi_effect": MIDI processor. Rare.

PARAMETERS:
- id: Unique C-safe identifier (lowercase_underscore). Used as APVTS parameter ID.
- name: Human-readable display name.
- min/max: Value range. Use real units (Hz, dB, ms, %).
- default: Default value within [min, max].
- skew: Skew factor for NormalisableRange. <1 = more resolution at low end (use 0.2-0.4 for frequency). 1.0 = linear. >1 = more at high end.
- unit: Display unit string ("Hz", "dB", "ms", "%", "").

DSP NODE TYPES:

**Filters:**
- lowpass: { cutoff: Hz, resonance: 0-1 }. State-variable TPT filter.
- highpass: { cutoff: Hz, resonance: 0-1 }.
- bandpass: { cutoff: Hz, resonance: 0-1 }.
- notch: { cutoff: Hz, resonance: 0-1 }.
- allpass: { cutoff: Hz, resonance: 0-1 }.
- ladder: { cutoff: Hz, resonance: 0-1, mode: "LP12"|"LP24"|"HP12"|"HP24"|"BP12"|"BP24" }. Moog-style ladder filter.
- comb: { delay_ms: 0.1-50, feedback: -1 to 1 }. Comb filter / Karplus-Strong.
- shelf_low: { cutoff: Hz, gain_db: -24 to 24 }.
- shelf_high: { cutoff: Hz, gain_db: -24 to 24 }.
- parametric_eq: { freq: Hz, gain_db: -24 to 24, q: 0.1-10 }.

**Dynamics:**
- compressor: { threshold_db: -60 to 0, ratio: 1-20, attack_ms: 0.1-200, release_ms: 10-2000, makeup_db: 0-24 }.
- limiter: { threshold_db: -24 to 0, release_ms: 10-500 }.
- gate: { threshold_db: -80 to 0, attack_ms: 0.1-50, release_ms: 10-1000 }.
- expander: { threshold_db: -80 to 0, ratio: 1-8, attack_ms: 0.1-50, release_ms: 10-1000 }.
- envelope_follower: { attack_ms: 0.1-100, release_ms: 10-1000 }. Outputs control signal.

**Time-Based:**
- delay: { time_ms: 0-2000, feedback: 0-0.95, mix: 0-1 }. Mono/stereo delay.
- multitap_delay: { taps: [{ time_ms, gain, pan }], feedback: 0-0.95, mix: 0-1 }.
- ping_pong_delay: { time_ms: 0-2000, feedback: 0-0.95, mix: 0-1, spread: 0-1 }.
- reverb: { room_size: 0-1, damping: 0-1, width: 0-1, mix: 0-1 }. Freeverb algorithm.
- convolution: { ir_file: "path_or_url" }. Convolution reverb.

**Modulation:**
- chorus: { rate_hz: 0.1-5, depth: 0-1, mix: 0-1, voices: 1-4 }.
- flanger: { rate_hz: 0.05-2, depth: 0-1, feedback: -1 to 1, mix: 0-1 }.
- phaser: { rate_hz: 0.05-5, depth: 0-1, feedback: -1 to 1, mix: 0-1, stages: 2-12 }.
- tremolo: { rate_hz: 0.5-20, depth: 0-1, shape: "sine"|"triangle"|"square" }.
- ring_mod: { freq_hz: 20-2000, mix: 0-1 }. Ring modulator.
- lfo: { rate_hz: 0.01-20, shape: "sine"|"triangle"|"saw"|"square", depth: 0-1, target: "@param_id" }. Modulates another parameter.

**Distortion:**
- overdrive: { drive: 0-1, tone: 0-1, mix: 0-1 }. Soft clipping (tanh).
- waveshaper: { curve: "tanh"|"atan"|"cubic"|"hard_clip", amount: 0-1, mix: 0-1 }.
- bitcrusher: { bit_depth: 1-16, sample_rate_div: 1-64, mix: 0-1 }. Lo-fi effect.
- saturation: { amount: 0-1, asymmetry: 0-1, mix: 0-1 }. Tube-style saturation.
- foldback: { threshold: 0.1-1, mix: 0-1 }. Foldback distortion.

**Utility:**
- gain: { gain_db: -60 to 24 }. Simple gain stage.
- pan: { pan: -1 to 1 }. Stereo panning (-1=L, 0=center, 1=R).
- mix: { dry: 0-1, wet: 0-1 }. Dry/wet mixer for parallel processing.
- dc_blocker: {}. Removes DC offset (5Hz highpass).

**Synthesis (instrument plugins):**
- oscillator: { waveform: "sine"|"saw"|"square"|"triangle"|"noise", detune: -100 to 100 cents, level: 0-1 }.
- noise: { type: "white"|"pink"|"brown", level: 0-1 }.
- wavetable: { table: "basic_shapes", position: 0-1, level: 0-1 }.
- fm_operator: { ratio: 0.5-16, index: 0-10, level: 0-1 }. FM synthesis operator.
- envelope_adsr: { attack_ms: 0.1-5000, decay_ms: 1-5000, sustain: 0-1, release_ms: 1-10000, target: "amp"|"filter"|"@param_id" }.
- sample_player: { file: "path_or_url", root_note: 60 }.

**Analysis:**
- peak_meter: {}. Peak level meter.
- rms_meter: { window_ms: 10-300 }. RMS level meter.

PARAMETER BINDING:
- Use "@param_id" to bind a node parameter to a user-exposed parameter.
- Use literal numbers for fixed values (e.g., "feedback": 0.5).
- Example: { "cutoff": "@filter_freq" } means the cutoff tracks the "filter_freq" parameter.

ROUTING:
- "input": "stereo"|"mono" — how the plugin receives audio.
- "chain": ["id1", "id2"] — serial processing order. Audio flows through each node.
- "output": "stereo"|"mono".
- For parallel processing, use a "splitter" node to branch and "merger" to combine.

INSTRUMENT PLUGINS:
Add "midi" field: { "voices": 8, "pitchBendRange": 2 }.
The dspChain runs per-voice. Include oscillator(s) and envelope_adsr at minimum.

DSP CHEAT SHEET:
- Frequency params: Use skew 0.2-0.4 for log distribution (20Hz-20kHz needs skew ~0.25).
- Time params (ms): Use skew 0.3-0.5 for ms ranges (1ms-2000ms).
- dB params: Linear skew (1.0) is fine for -60 to +24 dB.
- 0-1 params (mix, depth): Linear skew (1.0).
- Always include a final gain node for output level control.
- Compressors: typical attack 1-30ms, release 50-300ms, ratio 2-8:1 for general use.
- Reverb: room_size 0.3-0.7 for rooms, 0.7-1.0 for halls. Always control mix.
- Delay: keep feedback < 0.9 to prevent runaway. Typical times 100-500ms.
- Filter resonance: keep < 0.95 to prevent self-oscillation (unless desired).

MODE:
- "replace" (default): Replaces current DSP config entirely.
- "merge": Adds new nodes and parameters to existing config.

EXAMPLE — Compressor:
\`\`\`dsplang
{
  "pluginType": "effect",
  "name": "Studio Compressor",
  "parameters": [
    { "id": "threshold", "name": "Threshold", "min": -60, "max": 0, "default": -18, "skew": 1.0, "unit": "dB" },
    { "id": "ratio", "name": "Ratio", "min": 1, "max": 20, "default": 4, "skew": 0.5, "unit": ":1" },
    { "id": "attack", "name": "Attack", "min": 0.1, "max": 200, "default": 10, "skew": 0.4, "unit": "ms" },
    { "id": "release", "name": "Release", "min": 10, "max": 2000, "default": 150, "skew": 0.4, "unit": "ms" },
    { "id": "makeup", "name": "Makeup", "min": 0, "max": 24, "default": 0, "skew": 1.0, "unit": "dB" },
    { "id": "mix", "name": "Mix", "min": 0, "max": 1, "default": 1, "skew": 1.0, "unit": "" }
  ],
  "dspChain": [
    { "type": "compressor", "id": "comp1", "params": { "threshold_db": "@threshold", "ratio": "@ratio", "attack_ms": "@attack", "release_ms": "@release" } },
    { "type": "gain", "id": "makeup_gain", "params": { "gain_db": "@makeup" } },
    { "type": "mix", "id": "drywet", "params": { "dry": 0, "wet": "@mix" } }
  ],
  "routing": { "input": "stereo", "chain": ["comp1", "makeup_gain", "drywet"], "output": "stereo" }
}
\`\`\`

EXAMPLE — Synthesizer:
\`\`\`dsplang
{
  "pluginType": "instrument",
  "name": "Analog Poly Synth",
  "parameters": [
    { "id": "osc_mix", "name": "Osc Mix", "min": 0, "max": 1, "default": 0.5, "skew": 1.0, "unit": "" },
    { "id": "filter_freq", "name": "Filter", "min": 20, "max": 20000, "default": 2000, "skew": 0.25, "unit": "Hz" },
    { "id": "filter_res", "name": "Resonance", "min": 0, "max": 0.95, "default": 0.3, "skew": 1.0, "unit": "" },
    { "id": "env_attack", "name": "Attack", "min": 0.1, "max": 5000, "default": 10, "skew": 0.3, "unit": "ms" },
    { "id": "env_decay", "name": "Decay", "min": 1, "max": 5000, "default": 300, "skew": 0.3, "unit": "ms" },
    { "id": "env_sustain", "name": "Sustain", "min": 0, "max": 1, "default": 0.7, "skew": 1.0, "unit": "" },
    { "id": "env_release", "name": "Release", "min": 1, "max": 10000, "default": 500, "skew": 0.3, "unit": "ms" },
    { "id": "master_vol", "name": "Volume", "min": -60, "max": 6, "default": -6, "skew": 1.0, "unit": "dB" }
  ],
  "dspChain": [
    { "type": "oscillator", "id": "osc1", "params": { "waveform": "saw", "detune": 0, "level": 1 } },
    { "type": "oscillator", "id": "osc2", "params": { "waveform": "square", "detune": 7, "level": 1 } },
    { "type": "ladder", "id": "filt", "params": { "cutoff": "@filter_freq", "resonance": "@filter_res", "mode": "LP24" } },
    { "type": "envelope_adsr", "id": "amp_env", "params": { "attack_ms": "@env_attack", "decay_ms": "@env_decay", "sustain": "@env_sustain", "release_ms": "@env_release", "target": "amp" } },
    { "type": "gain", "id": "master", "params": { "gain_db": "@master_vol" } }
  ],
  "routing": { "input": "mono", "chain": ["osc1", "osc2", "filt", "amp_env", "master"], "output": "stereo" },
  "midi": { "voices": 8, "pitchBendRange": 2 }
}
\`\`\`

EXAMPLE — Multi-FX Chain:
\`\`\`dsplang
{
  "pluginType": "effect",
  "name": "Multi FX",
  "parameters": [
    { "id": "drive_amt", "name": "Drive", "min": 0, "max": 1, "default": 0.3, "skew": 1.0, "unit": "" },
    { "id": "filter_freq", "name": "Filter", "min": 20, "max": 20000, "default": 5000, "skew": 0.25, "unit": "Hz" },
    { "id": "chorus_rate", "name": "Chorus Rate", "min": 0.1, "max": 5, "default": 1.2, "skew": 1.0, "unit": "Hz" },
    { "id": "chorus_depth", "name": "Chorus Depth", "min": 0, "max": 1, "default": 0.4, "skew": 1.0, "unit": "" },
    { "id": "delay_time", "name": "Delay", "min": 10, "max": 1000, "default": 350, "skew": 0.5, "unit": "ms" },
    { "id": "delay_fb", "name": "Feedback", "min": 0, "max": 0.9, "default": 0.4, "skew": 1.0, "unit": "" },
    { "id": "reverb_size", "name": "Reverb Size", "min": 0, "max": 1, "default": 0.5, "skew": 1.0, "unit": "" },
    { "id": "reverb_mix", "name": "Reverb Mix", "min": 0, "max": 1, "default": 0.25, "skew": 1.0, "unit": "" },
    { "id": "output_gain", "name": "Output", "min": -24, "max": 12, "default": 0, "skew": 1.0, "unit": "dB" }
  ],
  "dspChain": [
    { "type": "overdrive", "id": "drive", "params": { "drive": "@drive_amt", "tone": 0.6, "mix": 1 } },
    { "type": "lowpass", "id": "lp", "params": { "cutoff": "@filter_freq", "resonance": 0.3 } },
    { "type": "chorus", "id": "cho", "params": { "rate_hz": "@chorus_rate", "depth": "@chorus_depth", "mix": 0.5 } },
    { "type": "delay", "id": "dly", "params": { "time_ms": "@delay_time", "feedback": "@delay_fb", "mix": 0.3 } },
    { "type": "reverb", "id": "rev", "params": { "room_size": "@reverb_size", "damping": 0.5, "width": 1, "mix": "@reverb_mix" } },
    { "type": "gain", "id": "out", "params": { "gain_db": "@output_gain" } }
  ],
  "routing": { "input": "stereo", "chain": ["drive", "lp", "cho", "dly", "rev", "out"], "output": "stereo" }
}
\`\`\`

IMPORTANT: Always output the dsplang block when the user asks for a DSP design, chain, or modification. Include a brief explanation of your design choices and signal flow. Keep explanations concise (2-4 sentences).

If the user asks to modify the current DSP (add nodes, change parameters), either use "mode": "merge" or output a full "replace" with the modifications.`;

/** Extract dsplang JSON blocks from AI message text */
function parseDspLang(text) {
  const blocks = [];
  const regex = /```dsplang\s*([\s\S]*?)```/g;
  let match;
  while ((match = regex.exec(text)) !== null) {
    try {
      const parsed = JSON.parse(match[1].trim());
      blocks.push(parsed);
    } catch {
      // malformed JSON — skip
    }
  }
  return blocks;
}

/** Render message text, replacing dsplang blocks with Apply/Generate buttons */
function renderDspContent(text, onApply, onGenerate) {
  const parts = [];
  const regex = /```dsplang\s*([\s\S]*?)```/g;
  let lastIdx = 0;
  let match;
  let blockIdx = 0;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIdx) {
      parts.push(
        <span key={`t-${blockIdx}`}>{text.slice(lastIdx, match.index)}</span>
      );
    }
    let parsed = null;
    try { parsed = JSON.parse(match[1].trim()); } catch {}

    if (parsed) {
      const nodeCount = parsed.dspChain ? parsed.dspChain.length : 0;
      const paramCount = parsed.parameters ? parsed.parameters.length : 0;
      const name = parsed.name || 'DSP Config';
      parts.push(
        <div key={`b-${blockIdx}`} className={styles.dsplangBlock}>
          <div className={styles.dsplangHeader}>
            <i className="fa-solid fa-microchip" />
            <span>{name} — {nodeCount} nodes, {paramCount} params</span>
          </div>
          <div className={styles.dsplangActions}>
            <button
              className={styles.applyDspBtn}
              onClick={() => onApply(parsed)}
            >
              <i className="fa-solid fa-check" /> Apply DSP
            </button>
            <button
              className={styles.generateCodeBtn}
              onClick={() => onGenerate(parsed)}
            >
              <i className="fa-solid fa-code" /> Generate Code
            </button>
          </div>
        </div>
      );
    } else {
      parts.push(
        <div key={`b-${blockIdx}`} className={styles.dsplangBlock}>
          <div className={styles.dsplangError}>
            <i className="fa-solid fa-triangle-exclamation" /> DSP parse error
          </div>
        </div>
      );
    }
    lastIdx = match.index + match[0].length;
    blockIdx++;
  }

  if (lastIdx < text.length) {
    parts.push(<span key="end">{text.slice(lastIdx)}</span>);
  }

  return parts.length > 0 ? parts : text;
}

const BackendChat = ({ pluginConfig, components, dspConfig, onApplyDsp }) => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "I'm your DSP backend engineer. Describe the audio plugin you want to build — I'll design the complete signal processing chain and generate compilable JUCE C++ code.\n\nTry: \"Design a stereo delay with ping-pong mode, a lowpass filter in the feedback loop, and a subtle saturation stage\"",
      timestamp: new Date().toISOString(),
    },
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [codePreview, setCodePreview] = useState(null); // { files: {...} }
  const [activeCodeFile, setActiveCodeFile] = useState(null);
  const [generatingCode, setGeneratingCode] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const buildUiContext = useCallback(() => {
    if (!components || components.length === 0) return null;
    const controls = components
      .filter(c => ['knob', 'slider', 'button', 'dropdown', 'xy-pad'].includes(c.type))
      .map(c => `${c.type}: "${c.label}"`);
    return {
      pluginName: pluginConfig.name,
      canvasSize: `${pluginConfig.width}x${pluginConfig.height}`,
      componentCount: components.length,
      controls: controls.join(', '),
      currentDsp: dspConfig ? {
        pluginType: dspConfig.pluginType,
        paramCount: dspConfig.parameters?.length || 0,
        chainLength: dspConfig.dspChain?.length || 0,
      } : null,
    };
  }, [pluginConfig, components, dspConfig]);

  const handleSend = useCallback(async () => {
    if (!inputMessage.trim() || isLoading) return;
    const userMsg = { role: 'user', content: inputMessage, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);
    setInputMessage('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await sendCodegenChatMessage({
        system_prompt: DSP_SYSTEM_PROMPT,
        message: inputMessage,
        conversation_history: messages,
        ui_context: buildUiContext(),
      });
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.message,
        timestamp: response.timestamp,
      }]);
    } catch (err) {
      setError(err.message || 'Failed to get response');
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }, [inputMessage, isLoading, messages, buildUiContext]);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleApplyDsp = useCallback((dsp) => {
    if (onApplyDsp) onApplyDsp(dsp);
  }, [onApplyDsp]);

  const handleGenerateCode = useCallback(async (dsp) => {
    setGeneratingCode(true);
    setError(null);
    try {
      const result = await generatePluginCode(dsp);
      setCodePreview(result);
      setActiveCodeFile(Object.keys(result.files)[0] || null);
    } catch (err) {
      setError(err.message || 'Code generation failed');
    } finally {
      setGeneratingCode(false);
    }
  }, []);

  const clearChat = () => {
    setMessages([{
      role: 'assistant',
      content: 'Chat cleared. Describe a plugin and I\'ll design the DSP chain.',
      timestamp: new Date().toISOString(),
    }]);
    setError(null);
    setCodePreview(null);
  };

  const formatTime = (ts) => {
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <>
      <div className={styles.chatHeader}>
        <div className={styles.chatHeaderTitle}>
          <i className="fa-solid fa-microchip" />
          <span>Backend Coder</span>
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          {codePreview && (
            <button className={styles.chatClearBtn} onClick={() => setCodePreview(null)} title="Close code preview">
              <i className="fa-solid fa-xmark" />
            </button>
          )}
          <button className={styles.chatClearBtn} onClick={clearChat} title="Clear chat">
            <i className="fa-solid fa-trash" />
          </button>
        </div>
      </div>

      {/* Code Preview Panel */}
      {codePreview && (
        <div className={styles.codePreviewPanel}>
          <div className={styles.codeFileTabs}>
            {Object.keys(codePreview.files).map(fname => (
              <button
                key={fname}
                className={activeCodeFile === fname ? styles.codeFileTabActive : styles.codeFileTab}
                onClick={() => setActiveCodeFile(fname)}
              >
                {fname}
              </button>
            ))}
          </div>
          <pre className={styles.codePreview}>
            <code>{codePreview.files[activeCodeFile] || ''}</code>
          </pre>
        </div>
      )}

      <div className={styles.chatMessages}>
        {messages.map((msg, i) => (
          <div key={i} className={`${styles.message} ${msg.role === 'user' ? styles.userMessage : styles.assistantMessage}`}>
            <div className={styles.messageIcon}>
              <i className={msg.role === 'user' ? 'fa-solid fa-user' : 'fa-solid fa-microchip'} />
            </div>
            <div className={styles.messageContent}>
              <div className={styles.messageText}>
                {msg.role === 'assistant'
                  ? renderDspContent(msg.content, handleApplyDsp, handleGenerateCode)
                  : msg.content
                }
              </div>
              <div className={styles.messageTime}>{formatTime(msg.timestamp)}</div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className={`${styles.message} ${styles.assistantMessage}`}>
            <div className={styles.messageIcon}>
              <i className="fa-solid fa-microchip" />
            </div>
            <div className={styles.messageContent}>
              <div className={styles.typingIndicator}>
                <span /><span /><span />
              </div>
            </div>
          </div>
        )}

        {generatingCode && (
          <div className={styles.chatGenerating}>
            <i className="fa-solid fa-spinner fa-spin" /> Generating JUCE C++ code...
          </div>
        )}

        {error && (
          <div className={styles.chatError}>
            <i className="fa-solid fa-exclamation-triangle" />
            <span>{error}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className={styles.chatInputArea}>
        <textarea
          ref={inputRef}
          className={styles.chatInput}
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Describe your plugin's DSP architecture..."
          rows={1}
          disabled={isLoading}
        />
        <button
          className={styles.chatSendBtn}
          onClick={handleSend}
          disabled={!inputMessage.trim() || isLoading}
        >
          <i className="fa-solid fa-paper-plane" />
        </button>
      </div>
    </>
  );
};

export default BackendChat;
