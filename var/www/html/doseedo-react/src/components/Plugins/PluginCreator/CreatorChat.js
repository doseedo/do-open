import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as chatAPI from '../../../services/chatAPI';
import styles from './PluginCreator.module.css';

const SYSTEM_PROMPT = `You are an elite VST plugin UI designer who creates stunning, professional audio plugin interfaces. You have deep knowledge of modern UI design styles including glassmorphism, skeumorphism, neumorphism, flat design, and creative themed designs (anime, retro, cyberpunk, etc.).

You output layouts using a structured "Plugin Language" format inside \`\`\`pluginlang code blocks.

## PLUGINLANG SCHEMA

{
  "pluginConfig": {
    "name": "Plugin Name",
    "width": 600, "height": 400,
    "bgColor": "#1a1a2e",
    "titleBarColor": "#2d2d4e",
    "bgImage": {"generate": "DALL-E prompt for background"} OR {"search": "Unsplash search query"} OR "url"
  },
  "components": [ ... ],
  "mode": "replace" OR "merge"
}

## COMPONENT TYPES

| Type | Default Size | Use For |
|------|-------------|---------|
| knob | 60x70 | Rotary control: gain, freq, mix, drive |
| slider | 30x120 | Vertical fader: volume, sends |
| button | 70x28 | Bypass, mode toggle |
| label | 100x24 | Section titles, value readouts |
| led | 12x12 | Status indicator |
| dropdown | 120x28 | Algorithm/mode select |
| image | 80x80 | Custom graphics, logos, artwork |
| panel | 200x150 | Section container/grouping |
| meter | 24x100 | VU/level meter |
| waveform | 180x60 | Oscilloscope display |
| xy-pad | 120x120 | 2D modulation control |

## ALL COMPONENT PROPERTIES

Common: x, y, width, height, color, label, opacity (0-1), rotation (deg), borderRadius (px), fontSize (px), zIndex (int)

Panel-specific: borderColor, bgColor, bgGradient (CSS gradient string like "linear-gradient(135deg, rgba(255,100,200,0.15), rgba(100,50,150,0.05))"), backdropBlur (px, for glass effects), boxShadow (CSS shadow string)

Image-specific: image — can be a URL string, OR {"generate": "DALL-E prompt"} to auto-generate, OR {"search": "query"} to auto-search stock photos. ALWAYS use image generation/search when the user wants custom visuals — never leave image blank.

## AUTO IMAGE GENERATION

When the user wants themed visuals (anime, retro, abstract art, textures, etc.), use the image generation system:
- For backgrounds: set pluginConfig.bgImage to {"generate": "detailed DALL-E prompt"} or {"search": "query"}
- For image components: set image to {"generate": "prompt"} or {"search": "query"}
- Write detailed DALL-E prompts: include style, colors, mood, composition. Example: {"generate": "kawaii anime girl with pink headphones and cat ears, pastel pink and purple gradient background, sparkles and stars, cute chibi style, digital illustration"}
- The system will auto-generate/search and apply the image. DO NOT tell the user to upload their own image.

## DESIGN STYLES — Use these to create stunning UIs:

**Glassmorphism**: Semi-transparent panels with backdrop blur. Use bgColor "rgba(255,255,255,0.08)", backdropBlur 12, borderColor "rgba(255,255,255,0.15)". Layer over a colorful background image.

**Skeumorphism**: Rich textures, realistic shadows, metallic knobs. Use dark grays (#2a2a2a, #1a1a1a), thick borders, boxShadow "0 4px 15px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1)".

**Neumorphism**: Soft raised/pressed elements. Light bg (#e0e0e0 area), boxShadow with dual light/dark. Panel: boxShadow "8px 8px 16px rgba(0,0,0,0.15), -8px -8px 16px rgba(255,255,255,0.05)".

**Cyberpunk/Neon**: Dark backgrounds (#0a0a1a), neon accent colors (#00ff88, #ff00ff, #00ddff), glowing borders, high contrast.

**Anime/Kawaii**: Pastel colors (#ffb7d5, #b5deff, #c5b3ff), rounded corners (borderRadius 12-20), cute fonts, generated anime background image.

**Retro/Vintage**: Warm browns (#3a2a1a), amber (#d4a043), cream (#f5e6c8), textured backgrounds, classic VU meter aesthetics.

**Modern Minimal**: Lots of whitespace, subtle accent color, thin lines, clean typography.

## COORDINATE SYSTEM
Origin (0,0) = top-left. X right, Y down. Title bar is 32px above canvas.

## EDITING RULES — CRITICAL

**Partial edits (user asks to change something specific)**:
- When the user says "change the background", "make it pink", "add a knob", "move the reverb knob" etc., they want you to KEEP everything else and only change what they asked.
- Use "mode": "replace" but include ALL existing components from context, modified as requested.
- Read the current layout from the context carefully. Preserve all components the user didn't mention.
- Only use "mode": "merge" when the user explicitly says "add" something new.

**Full redesign (user asks for a new plugin or complete restyle)**:
- When the user says "design a compressor", "make me a new plugin", "completely redesign" — do a full replace.

**When in doubt, preserve existing components.** Users get frustrated when their work disappears.

## LAYOUT BEST PRACTICES

1. Use panels with subtle borders + glass/gradient effects to group controls — never just float components.
2. Create visual hierarchy: large hero knobs (80x90) for main controls, smaller knobs (50x60) for secondary.
3. Consistent spacing: 80-100px between knob centers, 15-25px margins from edges.
4. Labels ABOVE or BELOW controls, centered with the control, font-size 10-11px, low opacity (0.4-0.6).
5. Color palette: Pick 1-2 accent colors and derive the whole scheme. Warm (amber+rust), cool (blue+purple), neon (cyan+magenta).
6. Use background images for themed plugins — always generate them, don't leave empty.
7. Layer: panels at zIndex 0-1, displays at 2, controls at 3-5, labels at 4-6.
8. Standard sizes: small 400x300, medium 600x400, large 900x500.

## EXAMPLE — Glassmorphic Delay with generated background:
\`\`\`pluginlang
{
  "pluginConfig": {
    "name": "Crystal Delay",
    "width": 650, "height": 420,
    "bgColor": "#0a0a2e",
    "titleBarColor": "rgba(20,20,60,0.8)",
    "bgImage": {"generate": "abstract cosmic nebula with deep blue and purple swirls, stars and light particles, ethereal space atmosphere, digital art, dark background"}
  },
  "components": [
    {"type": "panel", "x": 20, "y": 15, "width": 300, "height": 390, "borderColor": "rgba(130,170,255,0.15)", "bgColor": "rgba(130,170,255,0.06)", "bgGradient": "linear-gradient(180deg, rgba(130,170,255,0.1) 0%, rgba(130,170,255,0.02) 100%)", "backdropBlur": 12, "borderRadius": 16, "zIndex": 0, "label": ""},
    {"type": "panel", "x": 330, "y": 15, "width": 300, "height": 390, "borderColor": "rgba(200,130,255,0.15)", "bgColor": "rgba(200,130,255,0.06)", "bgGradient": "linear-gradient(180deg, rgba(200,130,255,0.1) 0%, rgba(200,130,255,0.02) 100%)", "backdropBlur": 12, "borderRadius": 16, "zIndex": 0, "label": ""},
    {"type": "label", "label": "TIME", "x": 130, "y": 22, "width": 60, "height": 18, "color": "rgba(130,170,255,0.5)", "fontSize": 10, "zIndex": 5},
    {"type": "label", "label": "CHARACTER", "x": 430, "y": 22, "width": 80, "height": 18, "color": "rgba(200,130,255,0.5)", "fontSize": 10, "zIndex": 5},
    {"type": "knob", "label": "Time", "x": 55, "y": 60, "width": 80, "height": 90, "color": "#82aaff", "zIndex": 3},
    {"type": "knob", "label": "Feedback", "x": 175, "y": 60, "width": 80, "height": 90, "color": "#82aaff", "zIndex": 3},
    {"type": "knob", "label": "Tone", "x": 370, "y": 60, "width": 80, "height": 90, "color": "#c882ff", "zIndex": 3},
    {"type": "knob", "label": "Mod", "x": 500, "y": 60, "width": 65, "height": 75, "color": "#c882ff", "zIndex": 3},
    {"type": "waveform", "label": "Delay Shape", "x": 40, "y": 190, "width": 260, "height": 65, "color": "#82aaff", "zIndex": 3},
    {"type": "xy-pad", "label": "Space", "x": 360, "y": 180, "width": 240, "height": 100, "color": "#c882ff", "zIndex": 3, "borderRadius": 12},
    {"type": "slider", "label": "Mix", "x": 70, "y": 290, "width": 25, "height": 90, "color": "#82aaff", "zIndex": 3},
    {"type": "knob", "label": "Spread", "x": 140, "y": 300, "width": 55, "height": 65, "color": "#6a8aee", "zIndex": 3},
    {"type": "dropdown", "label": "Stereo", "x": 230, "y": 315, "width": 70, "height": 26, "color": "#82aaff", "fontSize": 10, "zIndex": 3},
    {"type": "meter", "label": "L", "x": 375, "y": 310, "width": 18, "height": 70, "color": "#4caf50", "zIndex": 3},
    {"type": "meter", "label": "R", "x": 400, "y": 310, "width": 18, "height": 70, "color": "#4caf50", "zIndex": 3},
    {"type": "button", "label": "Ping-Pong", "x": 450, "y": 320, "width": 80, "height": 26, "color": "#c882ff", "borderRadius": 6, "fontSize": 10, "zIndex": 3},
    {"type": "button", "label": "Bypass", "x": 545, "y": 320, "width": 65, "height": 26, "color": "rgba(255,255,255,0.15)", "borderRadius": 6, "fontSize": 10, "zIndex": 3},
    {"type": "led", "label": "", "x": 540, "y": 325, "width": 8, "height": 8, "color": "#4caf50", "zIndex": 4}
  ]
}
\`\`\`

IMPORTANT:
- Always output pluginlang when the user asks for a layout, redesign, or any visual change.
- When modifying, preserve components the user didn't mention — read the context.
- When the user wants themed visuals, ALWAYS use {"generate": "..."} for images — never tell them to upload manually.
- Brief explanation of design choices (2-3 sentences) before/after the block.`;

/** Extract pluginlang JSON blocks from AI message text */
function parsePluginLang(text) {
  const blocks = [];
  const regex = /```pluginlang\s*([\s\S]*?)```/g;
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

/** Render message text, replacing pluginlang blocks with Apply buttons */
function renderMessageContent(text, onApply) {
  const parts = [];
  const regex = /```pluginlang\s*([\s\S]*?)```/g;
  let lastIdx = 0;
  let match;
  let blockIdx = 0;

  while ((match = regex.exec(text)) !== null) {
    // Text before the block
    if (match.index > lastIdx) {
      parts.push(
        <span key={`t-${blockIdx}`}>{text.slice(lastIdx, match.index)}</span>
      );
    }
    // Try to parse the block
    let parsed = null;
    try { parsed = JSON.parse(match[1].trim()); } catch {}

    if (parsed) {
      const compCount = parsed.components ? parsed.components.length : 0;
      const name = parsed.pluginConfig?.name || 'Layout';
      parts.push(
        <div key={`b-${blockIdx}`} className={styles.pluginlangBlock}>
          <div className={styles.pluginlangHeader}>
            <i className="fa-solid fa-puzzle-piece" />
            <span>{name} — {compCount} components</span>
          </div>
          <button
            className={styles.applyLayoutBtn}
            onClick={() => onApply(parsed)}
          >
            <i className="fa-solid fa-wand-magic-sparkles" /> Apply Layout
          </button>
        </div>
      );
    } else {
      parts.push(
        <div key={`b-${blockIdx}`} className={styles.pluginlangBlock}>
          <div className={styles.pluginlangError}>
            <i className="fa-solid fa-triangle-exclamation" /> Layout parse error
          </div>
        </div>
      );
    }
    lastIdx = match.index + match[0].length;
    blockIdx++;
  }

  // Remaining text after last block
  if (lastIdx < text.length) {
    parts.push(<span key="end">{text.slice(lastIdx)}</span>);
  }

  return parts.length > 0 ? parts : text;
}

const CreatorChat = ({ pluginConfig, components, dspContext, onApplyLayout, onOpenImageBrowser }) => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hey! I'm your plugin UI designer. Describe the plugin you want to build — I'll generate a complete layout you can apply with one click.\n\nTry: \"Design a warm analog compressor with input/output knobs, a gain reduction meter, and attack/release controls\"",
      timestamp: new Date().toISOString(),
    },
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const buildContext = useCallback(() => {
    const ctx = {
      pluginName: pluginConfig.name,
      canvasSize: `${pluginConfig.width}x${pluginConfig.height}`,
      bgColor: pluginConfig.bgColor,
      componentCount: components.length,
      components: components.map(c => ({
        type: c.type, label: c.label, x: c.x, y: c.y, width: c.width, height: c.height,
      })),
    };
    if (dspContext) {
      ctx.dspPartnerContext = {
        pluginType: dspContext.pluginType,
        parameters: (dspContext.parameters || []).map(p => `${p.name} (${p.id}: ${p.min}-${p.max} ${p.unit})`),
        chainSummary: (dspContext.dspChain || []).map(n => `${n.type}:${n.id}`).join(' → '),
      };
    }
    return ctx;
  }, [pluginConfig, components, dspContext]);

  const handleSend = useCallback(async () => {
    if (!inputMessage.trim() || isLoading) return;
    const userMsg = { role: 'user', content: inputMessage, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);
    setInputMessage('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await chatAPI.sendChatMessage({
        system_prompt: SYSTEM_PROMPT,
        daw_context: buildContext(),
        message: inputMessage,
        conversation_history: messages,
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
  }, [inputMessage, isLoading, messages, buildContext]);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    setMessages([{
      role: 'assistant',
      content: 'Chat cleared. Describe a plugin and I\'ll design a layout for you.',
      timestamp: new Date().toISOString(),
    }]);
    setError(null);
  };

  const formatTime = (ts) => {
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  const handleApply = useCallback((layout) => {
    if (onApplyLayout) onApplyLayout(layout);
  }, [onApplyLayout]);

  return (
    <>
      <div className={styles.chatHeader}>
        <div className={styles.chatHeaderTitle}>
          <i className="fa-solid fa-wand-magic-sparkles" />
          <span>UI Designer</span>
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          {onOpenImageBrowser && (
            <button className={styles.chatClearBtn} onClick={() => onOpenImageBrowser('new-component')} title="Image browser">
              <i className="fa-solid fa-image" />
            </button>
          )}
          <button className={styles.chatClearBtn} onClick={clearChat} title="Clear chat">
            <i className="fa-solid fa-trash" />
          </button>
        </div>
      </div>

      <div className={styles.chatMessages}>
        {messages.map((msg, i) => (
          <div key={i} className={`${styles.message} ${msg.role === 'user' ? styles.userMessage : styles.assistantMessage}`}>
            <div className={styles.messageIcon}>
              <i className={msg.role === 'user' ? 'fa-solid fa-user' : 'fa-solid fa-wand-magic-sparkles'} />
            </div>
            <div className={styles.messageContent}>
              <div className={styles.messageText}>
                {msg.role === 'assistant'
                  ? renderMessageContent(msg.content, handleApply)
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
              <i className="fa-solid fa-wand-magic-sparkles" />
            </div>
            <div className={styles.messageContent}>
              <div className={styles.typingIndicator}>
                <span /><span /><span />
              </div>
            </div>
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
          placeholder="Describe your plugin layout..."
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

export default CreatorChat;
