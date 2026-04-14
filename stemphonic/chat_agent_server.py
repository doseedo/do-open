"""
Doseedo studio chat agent server.

A standalone Flask + Sock WebSocket server that implements the same agent
protocol as the desktop's `chat_server.py`, adapted for the web studio.

Key differences from the desktop:
- Tools are studio-relevant (generate audio, detect chords, regen stems,
  generate scores, etc.) — they call existing stemphonic_server endpoints
  on http://127.0.0.1:8765 OR emit `client_action` messages that the
  React frontend executes against its redux store.
- No Logic Pro / Tron / Ableton dependencies.
- Runs on port 8766 (alongside stemphonic_server.py:8765).

WebSocket protocol (matches desktop):
  Inbound:
    {type: chat, text}
    {type: set_mode, mode: default|auto|plan}
    {type: set_model, model}
    {type: clear_history}
    {type: cancel}
    {type: permission_response, request_id, approved}
    {type: question_response, request_id, answer}
    {type: daw_context, bpm, key, trackCount}

  Outbound:
    {type: settings, mode, model}
    {type: status, project, summary}
    {type: text_delta, text}
    {type: tool_start, id, name}
    {type: tool_result, id, name, ok, text}
    {type: turn_done, input_tokens, output_tokens, total_cost, context_pct}
    {type: thinking_start} / {type: thinking_done, elapsed_ms}
    {type: client_action, action: {kind, ...args}}
    {type: system, text}
    {type: error, text}
    {type: mode_changed, mode}
    {type: model_changed, model}

Setup:
  pip install flask flask-sock anthropic
  export ANTHROPIC_API_KEY=sk-ant-...
  python chat_agent_server.py

The server requires the ANTHROPIC_API_KEY env var. Without it, the
server still starts but every chat turn returns a friendly error.
"""

from __future__ import annotations

import os
import json
import time
import uuid
import logging
import threading
import traceback
from typing import Any, Dict, List, Optional

import requests
from flask import Flask
from flask_sock import Sock

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chat_agent")

app = Flask(__name__)
sock = Sock(app)

STEMPHONIC_BASE = "http://127.0.0.1:8765"
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

# Lazy Anthropic client
_anthropic = None
def get_client():
    global _anthropic
    if _anthropic is None:
        if not ANTHROPIC_KEY:
            return None
        try:
            import anthropic
            _anthropic = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        except ImportError:
            logger.error("anthropic package not installed; pip install anthropic")
            return None
    return _anthropic


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "generate_audio",
        "description": "Generate audio from a text prompt + optional reference. Calls the stemphonic engine. Use for: 'make a piano loop', 'add drums', etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Style/genre/mood description, e.g. 'lofi piano with rain'"},
                "instrument": {"type": "string", "description": "Optional: piano, guitar, bass, drums, strings, brass, winds, vocals, synth"},
                "duration": {"type": "number", "description": "Seconds (default 16)"},
                "bpm": {"type": "number", "description": "Optional BPM"},
                "key": {"type": "string", "description": "Optional key, e.g. 'Cmin', 'A'"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "generate_score_from_video",
        "description": "Generate a per-scene MIDI score from the timeline's video. Uses the detected scene boundaries and tempos to create a chord progression and master MIDI file. Use when the user wants to score the video on the timeline.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Tonic key, default C"},
                "scale_type": {"type": "string", "enum": ["major", "minor"], "description": "Default minor for cinematic"},
                "genre": {"type": "string", "enum": ["cinematic", "pop", "rock", "jazz", "ambient", "lofi", "edm"], "description": "Default cinematic"},
                "render_audio": {"type": "boolean", "description": "Also render audio via stemphonic"},
            },
        },
    },
    {
        "name": "detect_chords",
        "description": "Detect chord progression and BPM from an audio URL. Use when the user wants to analyze a track's harmony.",
        "input_schema": {
            "type": "object",
            "properties": {
                "audio_url": {"type": "string"},
                "mode": {"type": "string", "enum": ["master", "stems"], "description": "stems mode separates first then analyzes"},
            },
            "required": ["audio_url"],
        },
    },
    {
        "name": "set_bpm",
        "description": "Change the timeline BPM in the studio. Frontend-only mutation.",
        "input_schema": {
            "type": "object",
            "properties": {"bpm": {"type": "number"}},
            "required": ["bpm"],
        },
    },
    {
        "name": "set_chord",
        "description": "Set a chord at a specific beat index on the chord row.",
        "input_schema": {
            "type": "object",
            "properties": {
                "beat": {"type": "number"},
                "chord": {"type": "string", "description": "e.g. 'Cmaj7', 'Am', 'G7'"},
            },
            "required": ["beat", "chord"],
        },
    },
    {
        "name": "set_chord_progression",
        "description": "Replace the entire chord row with a list of chords. Each chord placed at a beat index.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chords": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "beat": {"type": "number"},
                            "chord": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["chords"],
        },
    },
    {
        "name": "clear_chords",
        "description": "Clear the entire chord row.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_meter",
        "description": "Set the time signature beats-per-bar (e.g. 4 for 4/4, 3 for 3/4, 7 for 7/8).",
        "input_schema": {
            "type": "object",
            "properties": {"beats_per_bar": {"type": "number"}},
            "required": ["beats_per_bar"],
        },
    },
    {
        "name": "ask_user_question",
        "description": "Ask the user a clarifying question and wait for their answer. Only use when you genuinely cannot proceed without more info.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
]


# Tools that need user permission in default mode (mutations to studio state)
DESTRUCTIVE_TOOLS = {"set_bpm", "set_chord", "set_chord_progression", "clear_chords", "set_meter"}


# ─────────────────────────────────────────────────────────────────────────────
# Tool execution
# ─────────────────────────────────────────────────────────────────────────────

def execute_tool(name: str, args: Dict[str, Any], session: "ChatSession") -> Dict[str, Any]:
    """Run a tool. Returns {ok, text, client_action} where client_action
    is an optional dict the frontend should execute against the redux store.
    """
    try:
        if name == "generate_audio":
            r = requests.post(
                f"{STEMPHONIC_BASE}/api/generate-stemphonic",
                json={
                    "prompt": args["prompt"],
                    "instrument": args.get("instrument", ""),
                    "duration": float(args.get("duration", 16)),
                    "steps": 50,
                    "cfg": 7.0,
                    "seed": -1,
                },
                timeout=15,
            )
            r.raise_for_status()
            tid = r.json().get("task_id")
            return {"ok": True, "text": f"queued generation task {tid}"}

        if name == "generate_score_from_video":
            # The frontend has the scene_durations + tempos in redux. We
            # ask the frontend to do the call by emitting a client_action.
            return {
                "ok": True,
                "text": "asking studio to generate score from video",
                "client_action": {
                    "kind": "generate_score_from_video",
                    "key": args.get("key", "C"),
                    "scale_type": args.get("scale_type", "minor"),
                    "genre": args.get("genre", "cinematic"),
                    "render_audio": bool(args.get("render_audio", False)),
                },
            }

        if name == "detect_chords":
            audio_url = args["audio_url"]
            mode = args.get("mode", "master")
            audio_resp = requests.get(audio_url, timeout=30)
            files = {"audioFile": ("track.wav", audio_resp.content, "audio/wav")}
            r = requests.post(
                f"{STEMPHONIC_BASE}/api/detect-chords",
                files=files,
                data={"mode": mode},
                timeout=300,
            )
            r.raise_for_status()
            d = r.json()
            return {
                "ok": True,
                "text": f"detected {len(d.get('chords', {}))} chords @ {d.get('bpm')} BPM, meter {d.get('beats_per_bar')}",
                "data": d,
            }

        if name == "set_bpm":
            return {
                "ok": True,
                "text": f"BPM \u2192 {args['bpm']}",
                "client_action": {"kind": "set_bpm", "bpm": args["bpm"]},
            }

        if name == "set_chord":
            return {
                "ok": True,
                "text": f"set chord {args['chord']} at beat {args['beat']}",
                "client_action": {"kind": "set_chord", "beat": args["beat"], "chord": args["chord"]},
            }

        if name == "set_chord_progression":
            return {
                "ok": True,
                "text": f"applied {len(args['chords'])} chord progression",
                "client_action": {"kind": "set_chords", "chords": args["chords"]},
            }

        if name == "clear_chords":
            return {
                "ok": True,
                "text": "cleared chord row",
                "client_action": {"kind": "clear_chords"},
            }

        if name == "set_meter":
            return {
                "ok": True,
                "text": f"meter \u2192 {args['beats_per_bar']}",
                "client_action": {"kind": "set_beats_per_bar", "beats_per_bar": args["beats_per_bar"]},
            }

        if name == "ask_user_question":
            # Block until the user answers via the question_response websocket message
            request_id = uuid.uuid4().hex
            answer_event = threading.Event()
            answer_holder = {"answer": None}
            session.pending_questions[request_id] = (answer_event, answer_holder)
            session.send({"type": "question_request", "request_id": request_id, "question": args["question"]})
            answer_event.wait(timeout=300)
            ans = answer_holder["answer"] or "(no answer)"
            return {"ok": True, "text": f"user answered: {ans}"}

        return {"ok": False, "text": f"unknown tool: {name}"}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "text": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Chat session — one per WebSocket connection
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Dø, an AI music production assistant embedded in the doseedo.com web studio.
You help users create music by interacting with the timeline, chord row, MIDI editor, and generation pipeline.

When the user asks to create or modify music:
- Use the available tools rather than describing what to do
- For "score this video": call generate_score_from_video
- For "make a piano loop": call generate_audio with instrument='piano'
- For "change to 90 BPM": call set_bpm
- For "use a sad chord progression": call set_chord_progression with appropriate chords

Be concise. The studio has: a video track, a chord row, multiple buses (Music/SFX/VO), a MIDI editor, and chord-aware regeneration.
The user is in front of the studio UI; they can see what you do.
"""


class ChatSession:
    def __init__(self, ws):
        self.ws = ws
        self.lock = threading.Lock()
        self.conversation: List[Dict[str, Any]] = []
        self.mode = "default"
        self.model = DEFAULT_MODEL
        self.daw_context: Dict[str, Any] = {}
        self.cancel_event = threading.Event()
        self.pending_questions: Dict[str, Any] = {}
        self.pending_permissions: Dict[str, Any] = {}

    def send(self, msg: Dict[str, Any]):
        try:
            with self.lock:
                self.ws.send(json.dumps(msg))
        except Exception:
            pass

    def request_permission(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        if self.mode == "auto":
            return True
        if tool_name not in DESTRUCTIVE_TOOLS:
            return True
        request_id = uuid.uuid4().hex
        evt = threading.Event()
        result = {"approved": False}
        self.pending_permissions[request_id] = (evt, result)
        self.send({
            "type": "permission_request",
            "request_id": request_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "description": f"Run {tool_name}({json.dumps(tool_input)})",
        })
        evt.wait(timeout=120)
        return result["approved"]

    def handle_chat(self, text: str):
        client = get_client()
        if not client:
            self.send({"type": "error", "text": "ANTHROPIC_API_KEY not configured on server"})
            self.send({"type": "turn_done", "input_tokens": 0, "output_tokens": 0, "total_cost": 0, "context_pct": 0})
            return

        self.cancel_event.clear()
        self.conversation.append({"role": "user", "content": text})

        # Build effective system prompt with DAW context
        sys = SYSTEM_PROMPT
        if self.daw_context:
            sys += f"\n\nCurrent studio state: {json.dumps(self.daw_context)}"
        if self.mode == "plan":
            sys += "\n\nYou are in PLAN mode. Describe what you would do but DO NOT call mutation tools (set_bpm, set_chord, etc)."

        max_turns = 8
        total_in = total_out = 0
        for _turn in range(max_turns):
            if self.cancel_event.is_set():
                self.send({"type": "system", "text": "cancelled"})
                break

            try:
                with client.messages.stream(
                    model=self.model,
                    max_tokens=4096,
                    system=sys,
                    tools=TOOLS,
                    messages=self.conversation,
                ) as stream:
                    current_text = ""
                    tool_calls: List[Dict[str, Any]] = []
                    for event in stream:
                        if self.cancel_event.is_set():
                            break
                        et = getattr(event, "type", None)
                        if et == "text":
                            delta = event.text
                            current_text += delta
                            self.send({"type": "text_delta", "text": delta})
                        elif et == "content_block_stop":
                            cb = getattr(event, "content_block", None)
                            if cb and getattr(cb, "type", None) == "tool_use":
                                tool_calls.append({
                                    "id": cb.id,
                                    "name": cb.name,
                                    "input": cb.input,
                                })
                    final = stream.get_final_message()
                    total_in += final.usage.input_tokens
                    total_out += final.usage.output_tokens
            except Exception as e:
                traceback.print_exc()
                self.send({"type": "error", "text": f"agent error: {e}"})
                break

            # Append assistant turn
            assistant_content: List[Dict[str, Any]] = []
            if current_text:
                assistant_content.append({"type": "text", "text": current_text})
            for tc in tool_calls:
                assistant_content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]})
            self.conversation.append({"role": "assistant", "content": assistant_content})

            if not tool_calls:
                # No more tools to call — done
                break

            # Execute each tool, send updates, build user-role tool_result message
            tool_results: List[Dict[str, Any]] = []
            for tc in tool_calls:
                if self.cancel_event.is_set():
                    break
                self.send({"type": "tool_start", "id": tc["id"], "name": tc["name"]})

                # Permission check
                if not self.request_permission(tc["name"], tc["input"]):
                    self.send({"type": "tool_result", "id": tc["id"], "name": tc["name"], "ok": False, "text": "denied by user"})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": "denied by user",
                        "is_error": True,
                    })
                    continue

                # Execute
                result = execute_tool(tc["name"], tc["input"], self)
                if "client_action" in result:
                    self.send({"type": "client_action", "action": result["client_action"]})
                self.send({"type": "tool_result", "id": tc["id"], "name": tc["name"], "ok": result["ok"], "text": result["text"]})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result["text"],
                    "is_error": not result["ok"],
                })

            self.conversation.append({"role": "user", "content": tool_results})

        # Compute approximate cost (Sonnet 4.5 pricing)
        cost_in = total_in * 3.0 / 1_000_000
        cost_out = total_out * 15.0 / 1_000_000
        self.send({
            "type": "turn_done",
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_cost": round(cost_in + cost_out, 4),
            "context_pct": min(1.0, total_in / 200_000),
        })


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────

@sock.route("/ws")
def chat_ws(ws):
    session = ChatSession(ws)
    session.send({"type": "settings", "model": session.model, "mode": session.mode})
    session.send({"type": "status", "connected": True, "project": "studio", "summary": None})

    try:
        while True:
            raw = ws.receive()
            if raw is None:
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            t = msg.get("type")
            if t == "chat":
                threading.Thread(
                    target=session.handle_chat,
                    args=(msg.get("text", ""),),
                    daemon=True,
                ).start()
            elif t == "set_mode":
                session.mode = msg.get("mode", "default")
                session.send({"type": "mode_changed", "mode": session.mode})
            elif t == "set_model":
                session.model = msg.get("model", DEFAULT_MODEL)
                session.send({"type": "model_changed", "model": session.model})
            elif t == "clear_history":
                session.conversation = []
                session.send({"type": "system", "text": "history cleared"})
            elif t == "cancel":
                session.cancel_event.set()
            elif t == "permission_response":
                rid = msg.get("request_id")
                if rid in session.pending_permissions:
                    evt, holder = session.pending_permissions.pop(rid)
                    holder["approved"] = bool(msg.get("approved"))
                    evt.set()
            elif t == "question_response":
                rid = msg.get("request_id")
                if rid in session.pending_questions:
                    evt, holder = session.pending_questions.pop(rid)
                    holder["answer"] = msg.get("answer", "")
                    evt.set()
            elif t == "daw_context":
                session.daw_context = {
                    "bpm": msg.get("bpm"),
                    "key": msg.get("key"),
                    "trackCount": msg.get("trackCount"),
                }
            elif t == "compact":
                # Drop oldest half
                n = len(session.conversation) // 2
                session.conversation = session.conversation[n:]
                session.send({"type": "compact_done", "removed": n})
    except Exception as e:
        logger.warning("ws session ended: %s", e)


@app.route("/health")
def health():
    return {
        "status": "ok",
        "anthropic_key_configured": bool(ANTHROPIC_KEY),
        "model": DEFAULT_MODEL,
    }


if __name__ == "__main__":
    if not ANTHROPIC_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — chat will return errors until configured")
    app.run(host="0.0.0.0", port=8766, threaded=True)
