"""HTTP client for the Doseedo Modal vision endpoint (Moondream 2).

Modal deployment lives in `modal/modal_chatbot.py`. The vision app exposes
two routes on its own URL (separate from the vLLM text-completions URL):

    GET  /health
    POST /v1/vision/analyze   (Bearer-gated, body matches VisionAnalyzeBody)

Tasks supported by the analyze route:
    "query"   — free-text VQA. body: {image_base64, prompt}
    "detect"  — bounding boxes for a named object. body: {image_base64, object}
    "point"   — x/y points for a named object. body: {image_base64, object}

Auth: VLLM_API_KEY (same secret the chat side uses, see modal_chatbot.py
header comment lines 49 and 467).
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


DEFAULT_VISION_URL = os.environ.get(
    "DOSEEDO_VISION_URL",
    "https://arlo--doseedo-chatbot-qwenchatbot-vision.modal.run",
)


@dataclass
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class VisionClient:
    """Thin wrapper over POST /v1/vision/analyze.

    Single-image-per-request: the server route does not batch (see
    modal_chatbot.py:506). Callers wanting many frames per scene should issue
    parallel requests — Modal's @modal.concurrent(max_inputs=16) lets up to 16
    in-flight calls share the container.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_VISION_URL,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        key = api_key or os.environ.get("VLLM_API_KEY")
        if not key:
            raise RuntimeError(
                "VLLM_API_KEY missing — set env var or pass api_key=. "
                "This is the bearer enforced by modal_chatbot.py vision route."
            )
        self.api_key = key
        self.timeout = timeout
        self._session = session or requests.Session()

    # --- low-level ---------------------------------------------------------

    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/vision/analyze"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        r = self._session.post(url, json=body, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _encode(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("ascii")

    # --- public ------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        url = f"{self.base_url}/health"
        r = self._session.get(url, timeout=self.timeout)
        return {"status_code": r.status_code, **(r.json() if r.headers.get("content-type", "").startswith("application/json") else {})}

    def query(self, image_bytes: bytes, prompt: str) -> str:
        out = self._post({
            "image_base64": self._encode(image_bytes),
            "prompt": prompt,
            "task": "query",
        })
        return out.get("answer", "")

    def detect(self, image_bytes: bytes, obj: str) -> List[BBox]:
        out = self._post({
            "image_base64": self._encode(image_bytes),
            "object": obj,
            "task": "detect",
        })
        boxes: List[BBox] = []
        for raw in out.get("objects", []) or []:
            try:
                boxes.append(BBox(
                    x_min=float(raw["x_min"]),
                    y_min=float(raw["y_min"]),
                    x_max=float(raw["x_max"]),
                    y_max=float(raw["y_max"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return boxes

    # --- structured-JSON helper ------------------------------------------

    def query_json(
        self,
        image_bytes: bytes,
        prompt: str,
        retries: int = 1,
    ) -> Dict[str, Any]:
        """Ask Moondream for JSON, parse it leniently.

        Moondream's `query` is free-text — no guided-JSON support. We append a
        strict instruction, then strip code fences and isolate the first
        balanced `{...}` span. Returns {} on hard parse failure rather than
        raising; the caller decides how to fall back.
        """
        instruction = (
            "\n\nRespond with a single valid JSON object only. "
            "No prose, no markdown, no code fences."
        )
        last_text = ""
        for attempt in range(retries + 1):
            text = self.query(image_bytes, prompt + instruction)
            last_text = text
            parsed = _parse_json_lenient(text)
            if parsed is not None:
                return parsed
        # Hard fail: surface the raw text under a known key so callers can log.
        return {"_parse_failed": True, "_raw": last_text}


# ---------------------------------------------------------------------------
# JSON-from-prose parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)


def _parse_json_lenient(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidate = text.strip()

    # Strip ```json ... ``` fences if present.
    m = _FENCE_RE.search(candidate)
    if m:
        candidate = m.group(1).strip()

    # Direct parse.
    try:
        v = json.loads(candidate)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass

    # Isolate first balanced { ... } span.
    start = candidate.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                fragment = candidate[start:i + 1]
                try:
                    v = json.loads(fragment)
                    return v if isinstance(v, dict) else None
                except json.JSONDecodeError:
                    return None
    return None
