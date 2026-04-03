"""
WebSocket chat server for the doseedo.com/studio web DAW.
Adds a /ws endpoint to the existing chatbot_service.py FastAPI app,
implementing the same streaming agent protocol as the desktop chat_server.py.

Deploy alongside the existing chatbot_service.py on the production server.
Nginx config:  location /_chat/ws { proxy_pass http://127.0.0.1:8766/ws; ... }
"""

import asyncio
import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

try:
    import anthropic
except ImportError:
    anthropic = None

logger = logging.getLogger(__name__)

app = FastAPI(title="Dø Web Agent Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ──

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEFAULT_MODEL = os.getenv("DOO_MODEL", "claude-sonnet-4-20250514")
THINKING_BUDGET = int(os.getenv("DOO_THINKING_BUDGET", "10000"))
MAX_TOKENS = 16384

# ── Tool definitions for the web DAW ──
# These are the tools Claude can call to control the web studio session.
# The results are sent back to the frontend which dispatches them to the
# DAW reducer / AppContext.

WEB_DAW_TOOLS = [
    {
        "name": "get_session_info",
        "description": "Get current session info: BPM, key, time signature, track count, bus layout.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_tracks",
        "description": "List all tracks with name, type, volume, pan, mute, solo status.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_bpm",
        "description": "Set the session tempo.",
        "input_schema": {
            "type": "object",
            "properties": {"bpm": {"type": "number", "description": "Tempo in BPM (20-999)"}},
            "required": ["bpm"],
        },
    },
    {
        "name": "set_track_volume",
        "description": "Set volume for a track.",
        "input_schema": {
            "type": "object",
            "properties": {
                "track_index": {"type": "integer"},
                "volume": {"type": "number", "description": "Volume 0.0 to 1.0"},
            },
            "required": ["track_index", "volume"],
        },
    },
    {
        "name": "set_track_pan",
        "description": "Set pan for a track.",
        "input_schema": {
            "type": "object",
            "properties": {
                "track_index": {"type": "integer"},
                "pan": {"type": "number", "description": "Pan -1.0 (L) to 1.0 (R)"},
            },
            "required": ["track_index", "pan"],
        },
    },
    {
        "name": "set_track_mute",
        "description": "Mute or unmute a track.",
        "input_schema": {
            "type": "object",
            "properties": {
                "track_index": {"type": "integer"},
                "muted": {"type": "boolean"},
            },
            "required": ["track_index", "muted"],
        },
    },
    {
        "name": "set_track_solo",
        "description": "Solo or unsolo a track.",
        "input_schema": {
            "type": "object",
            "properties": {
                "track_index": {"type": "integer"},
                "soloed": {"type": "boolean"},
            },
            "required": ["track_index", "soloed"],
        },
    },
    {
        "name": "add_track",
        "description": "Add a new track to the session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": ["audio", "midi", "instrument", "bus"]},
                "bus": {"type": "string", "description": "Bus name to add to (optional)"},
            },
            "required": ["name", "type"],
        },
    },
    {
        "name": "remove_track",
        "description": "Remove a track. This is destructive and requires permission.",
        "input_schema": {
            "type": "object",
            "properties": {"track_index": {"type": "integer"}},
            "required": ["track_index"],
        },
    },
    {
        "name": "rename_track",
        "description": "Rename a track.",
        "input_schema": {
            "type": "object",
            "properties": {
                "track_index": {"type": "integer"},
                "name": {"type": "string"},
            },
            "required": ["track_index", "name"],
        },
    },
    {
        "name": "generate_audio",
        "description": "Generate audio using AI (text-to-music). Returns audio URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Description of the audio to generate"},
                "duration": {"type": "number", "description": "Duration in seconds (default 30)"},
                "style": {"type": "string", "description": "Musical style/genre"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "ask_user_question",
        "description": "Ask the user a clarifying question. Use when you need more information.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for information.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

DESTRUCTIVE_TOOLS = {"remove_track", "delete_midi_notes", "delete_region", "clear_session"}
READ_ONLY_TOOLS = {"get_session_info", "get_tracks", "get_markers", "get_midi_events", "web_search"}

# ── Pricing ──

PRICING = {
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.8, 4.0),
}


# ── Connection State ──

class ConnectionState:
    def __init__(self):
        self.messages = []  # conversation history
        self.model = DEFAULT_MODEL
        self.mode = "default"  # "default" | "auto" | "plan"
        self.input_tokens = 0
        self.output_tokens = 0
        self.daw_context = {}
        self.pending_permissions = {}  # request_id -> asyncio.Future
        self.pending_questions = {}    # request_id -> asyncio.Future
        self.todos = []


# ── WebSocket endpoint ──

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state = ConnectionState()
    logger.info("WebSocket connected")

    # Send initial settings
    await ws.send_json({
        "type": "settings",
        "model": state.model,
        "mode": state.mode,
    })

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type", "")

            if msg_type == "chat":
                text = data.get("text", "").strip()
                if text:
                    asyncio.create_task(_handle_turn(ws, state, text))

            elif msg_type == "set_mode":
                state.mode = data.get("mode", "default")
                await ws.send_json({"type": "mode_changed", "mode": state.mode})

            elif msg_type == "set_model":
                state.model = data.get("model", DEFAULT_MODEL)
                await ws.send_json({"type": "model_changed", "model": state.model})

            elif msg_type == "clear_history":
                state.messages = []
                state.input_tokens = 0
                state.output_tokens = 0
                state.todos = []
                await ws.send_json({"type": "system", "text": "Conversation cleared."})

            elif msg_type == "compact":
                # Simple compaction: keep last 10 messages
                if len(state.messages) > 20:
                    state.messages = state.messages[-10:]
                    await ws.send_json({"type": "compact_done"})
                else:
                    await ws.send_json({"type": "system", "text": "History is short, no compaction needed."})

            elif msg_type == "cancel":
                # Cancel is handled by the turn task checking a flag
                state._cancelled = True

            elif msg_type == "daw_context":
                state.daw_context = data

            elif msg_type == "permission_response":
                req_id = data.get("request_id")
                fut = state.pending_permissions.pop(req_id, None)
                if fut and not fut.done():
                    fut.set_result(data.get("approved", False))

            elif msg_type == "question_response":
                req_id = data.get("request_id")
                fut = state.pending_questions.pop(req_id, None)
                if fut and not fut.done():
                    fut.set_result(data.get("answer", ""))

            elif msg_type == "status_request":
                ctx = state.daw_context
                summary = f"BPM: {ctx.get('bpm', '?')} | Tracks: {ctx.get('trackCount', '?')}"
                await ws.send_json({"type": "status", "connected": True, "summary": summary})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# ── Agent turn ──

async def _handle_turn(ws: WebSocket, state: ConnectionState, user_text: str):
    """Run one agentic turn: send to Claude, stream response, execute tools, loop."""

    if not ANTHROPIC_API_KEY:
        await ws.send_json({"type": "error", "text": "ANTHROPIC_API_KEY not configured on server."})
        return

    if anthropic is None:
        await ws.send_json({"type": "error", "text": "anthropic package not installed on server."})
        return

    state._cancelled = False

    # Add user message to history
    state.messages.append({"role": "user", "content": user_text})

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build system prompt with DAW context
    from ChatWindow.systemPrompt import SYSTEM_PROMPT
    ctx = state.daw_context
    system_text = SYSTEM_PROMPT
    if ctx:
        system_text += f"\n\nCurrent session: BPM={ctx.get('bpm', 120)}, Key={ctx.get('key', 'C')}, Tracks={ctx.get('trackCount', 0)}"

    # Tool definitions
    tools = [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]} for t in WEB_DAW_TOOLS]

    turn_input_tokens = 0
    turn_output_tokens = 0

    try:
        while True:
            if state._cancelled:
                await ws.send_json({"type": "system", "text": "Cancelled."})
                break

            # Call Claude with streaming
            stream_kwargs = {
                "model": state.model,
                "max_tokens": MAX_TOKENS,
                "system": system_text,
                "messages": state.messages,
                "tools": tools,
            }

            # Add thinking if supported
            if THINKING_BUDGET > 0 and "opus" in state.model:
                stream_kwargs["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET}

            accumulated_text = ""
            tool_uses = []
            has_tool_use = False

            with client.messages.stream(**stream_kwargs) as stream:
                thinking_started = False

                for event in stream:
                    if state._cancelled:
                        break

                    if event.type == "content_block_start":
                        block = event.content_block
                        if hasattr(block, "type"):
                            if block.type == "thinking":
                                if not thinking_started:
                                    await ws.send_json({"type": "thinking_start"})
                                    thinking_started = True
                            elif block.type == "tool_use":
                                has_tool_use = True
                                tool_uses.append({
                                    "id": block.id,
                                    "name": block.name,
                                    "input_json": "",
                                })
                                await ws.send_json({
                                    "type": "tool_start",
                                    "id": block.id,
                                    "name": block.name,
                                })

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "type"):
                            if delta.type == "text_delta":
                                accumulated_text += delta.text
                                await ws.send_json({"type": "text_delta", "text": delta.text})
                            elif delta.type == "input_json_delta" and tool_uses:
                                tool_uses[-1]["input_json"] += delta.partial_json
                            elif delta.type == "thinking_delta":
                                pass  # Don't send thinking content to client

                    elif event.type == "content_block_stop":
                        if thinking_started:
                            await ws.send_json({"type": "thinking_done", "elapsed_ms": 0})
                            thinking_started = False

                    elif event.type == "message_delta":
                        if hasattr(event, "usage") and event.usage:
                            turn_output_tokens += getattr(event.usage, "output_tokens", 0)

                    elif event.type == "message_start":
                        if hasattr(event, "message") and hasattr(event.message, "usage"):
                            turn_input_tokens += getattr(event.message.usage, "input_tokens", 0)

                # Get final message
                final_message = stream.get_final_message()

            # Process tool calls
            if has_tool_use and tool_uses:
                # Add assistant message with tool_use blocks to history
                assistant_content = []
                if accumulated_text:
                    assistant_content.append({"type": "text", "text": accumulated_text})
                for tu in tool_uses:
                    try:
                        tool_input = json.loads(tu["input_json"]) if tu["input_json"] else {}
                    except json.JSONDecodeError:
                        tool_input = {}
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tu["id"],
                        "name": tu["name"],
                        "input": tool_input,
                    })
                state.messages.append({"role": "assistant", "content": assistant_content})

                # Execute tools
                tool_results = []
                for tu in tool_uses:
                    try:
                        tool_input = json.loads(tu["input_json"]) if tu["input_json"] else {}
                    except json.JSONDecodeError:
                        tool_input = {}

                    # Permission check for destructive tools
                    if tu["name"] in DESTRUCTIVE_TOOLS and state.mode == "default":
                        req_id = f"perm_{time.time()}"
                        fut = asyncio.get_event_loop().create_future()
                        state.pending_permissions[req_id] = fut
                        await ws.send_json({
                            "type": "permission_request",
                            "request_id": req_id,
                            "tool_name": tu["name"],
                            "tool_input": tool_input,
                            "description": f"Execute {tu['name'].replace('_', ' ')}",
                        })
                        try:
                            approved = await asyncio.wait_for(fut, timeout=60)
                        except asyncio.TimeoutError:
                            approved = False
                        if not approved:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tu["id"],
                                "content": "Permission denied by user.",
                                "is_error": True,
                            })
                            await ws.send_json({
                                "type": "tool_result",
                                "id": tu["id"],
                                "name": tu["name"],
                                "ok": False,
                                "text": "Permission denied",
                            })
                            continue

                    # Ask-user-question tool
                    if tu["name"] == "ask_user_question":
                        req_id = f"q_{time.time()}"
                        fut = asyncio.get_event_loop().create_future()
                        state.pending_questions[req_id] = fut
                        await ws.send_json({
                            "type": "question_request",
                            "request_id": req_id,
                            "question": tool_input.get("question", ""),
                        })
                        try:
                            answer = await asyncio.wait_for(fut, timeout=300)
                        except asyncio.TimeoutError:
                            answer = "(no response)"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tu["id"],
                            "content": answer,
                        })
                        await ws.send_json({
                            "type": "tool_result",
                            "id": tu["id"],
                            "name": tu["name"],
                            "ok": True,
                            "text": f"User answered: {answer}",
                        })
                        continue

                    # Execute the tool — send command to frontend for DAW tools
                    # The frontend dispatches these to the AppContext reducer
                    result = await _execute_web_tool(tu["name"], tool_input, state)
                    ok = not result.get("error")
                    result_text = json.dumps(result, default=str)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": result_text,
                        "is_error": not ok,
                    })
                    await ws.send_json({
                        "type": "tool_result",
                        "id": tu["id"],
                        "name": tu["name"],
                        "ok": ok,
                        "text": result_text[:200],
                    })

                # Add tool results to history
                state.messages.append({"role": "user", "content": tool_results})

                # Continue the agentic loop
                continue

            else:
                # No tool calls — turn is done
                if accumulated_text:
                    state.messages.append({"role": "assistant", "content": accumulated_text})
                break

    except Exception as e:
        logger.error(f"Turn error: {traceback.format_exc()}")
        await ws.send_json({"type": "error", "text": str(e)})

    # Calculate cost
    pricing = PRICING.get(state.model, (3.0, 15.0))
    cost = (turn_input_tokens * pricing[0] + turn_output_tokens * pricing[1]) / 1_000_000
    state.input_tokens += turn_input_tokens
    state.output_tokens += turn_output_tokens
    total_cost = (state.input_tokens * pricing[0] + state.output_tokens * pricing[1]) / 1_000_000

    # Estimate context usage
    total_tokens = state.input_tokens + state.output_tokens
    context_pct = min(total_tokens / 200_000, 1.0)

    await ws.send_json({
        "type": "turn_done",
        "input_tokens": turn_input_tokens,
        "output_tokens": turn_output_tokens,
        "total_cost": total_cost,
        "context_pct": context_pct,
    })


async def _execute_web_tool(name: str, args: dict, state: ConnectionState) -> dict:
    """
    Execute a web DAW tool. For session-state tools (get_session_info, get_tracks),
    return data from the state.daw_context. For mutation tools, return a command
    that the frontend will dispatch to its reducer.
    """
    ctx = state.daw_context

    if name == "get_session_info":
        return {
            "bpm": ctx.get("bpm", 120),
            "key": ctx.get("key", "C"),
            "trackCount": ctx.get("trackCount", 0),
        }

    if name == "get_tracks":
        # Return track list from DAW context
        buses = ctx.get("buses", [])
        tracks = []
        for bus in buses:
            for track in bus.get("tracks", []):
                tracks.append({
                    "name": track.get("name", "Untitled"),
                    "type": track.get("type", "audio"),
                    "bus": bus.get("name", ""),
                    "volume": track.get("volume", 0.8),
                    "pan": track.get("pan", 0),
                    "muted": track.get("muted", False),
                    "soloed": track.get("soloed", False),
                })
        return {"tracks": tracks}

    if name == "set_bpm":
        bpm = args.get("bpm", 120)
        ctx["bpm"] = bpm
        return {"ok": True, "bpm": bpm, "_dispatch": {"type": "SET_BPM", "payload": bpm}}

    if name in ("set_track_volume", "set_track_pan", "set_track_mute", "set_track_solo",
                "add_track", "remove_track", "rename_track"):
        # Return dispatch command for the frontend reducer
        return {"ok": True, "_dispatch": {"type": f"TOOL_{name.upper()}", "payload": args}}

    if name == "generate_audio":
        # Placeholder — the real implementation calls the generation API
        return {
            "ok": True,
            "message": f"Audio generation requested: {args.get('prompt', '')}",
            "status": "queued",
        }

    if name == "web_search":
        return {"ok": True, "message": "Web search not yet implemented in web studio."}

    return {"error": f"Unknown tool: {name}"}


# ── Health endpoint ──

@app.get("/")
async def health():
    return {"service": "Dø Web Agent", "status": "ok", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("DOO_WEB_PORT", "8766"))
    uvicorn.run(app, host="0.0.0.0", port=port)
