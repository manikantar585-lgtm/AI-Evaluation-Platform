"""Targets = the system under test.

  {"type": "openai", "base_url": "...", "model": "...", "api_key_env": "GROQ_API_KEY",
   "system_prompt": "...", "prompt_template": "Context:\\n{context}\\n\\nQuestion: {input}",
   "temperature": 0, "price_per_1m_input": 0.05, "price_per_1m_output": 0.08}

  {"type": "http", "url": "http://localhost:8000/api/ask", "method": "POST",
   "body": {"session_id": "eval-{id}", "question": "{input}"}, "response_path": "answer",
   "headers": {}}

  {"type": "replay", "predictions": {"case-id": "saved output", ...}}   # evaluate saved outputs
"""
import json
import os
import time
from typing import Any, cast

import httpx

from . import config
from .llm import chat


class TargetError(Exception):
    pass


def _fill(obj: Any, case: dict[str, Any]) -> Any:
    """Replace {input}, {context}, {id} placeholders anywhere inside a JSON-like body."""
    if isinstance(obj, str):
        out = obj
        for k in ("input", "context", "id"):
            out = out.replace("{" + k + "}", str(case.get(k, "")))
        return out
    if isinstance(obj, dict):
        return {k: _fill(v, case) for k, v in cast(dict[str, Any], obj).items()}
    if isinstance(obj, list):
        return [_fill(v, case) for v in cast(list[Any], obj)]
    return obj


def _dig(data: Any, path: str) -> Any:
    for part in [p for p in path.split(".") if p]:
        if isinstance(data, list):
            data = cast(list[Any], data)[int(part)]
        elif isinstance(data, dict):
            data = cast(dict[str, Any], data)[part]
        else:
            raise KeyError(path)
    return data


def call_target(cfg: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Returns {"output", "latency_ms", "prompt_tokens", "completion_tokens", "cost"}; raises TargetError."""
    t = cfg.get("type")
    if t == "replay":
        preds: dict[str, str] = cfg.get("predictions") or {}
        if case["id"] not in preds:
            raise TargetError("no saved prediction for this case")
        return {"output": preds[case["id"]], "latency_ms": None, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}

    if t == "openai":
        key = os.getenv(cfg.get("api_key_env", ""), "") or cfg.get("api_key", "")
        if not key:
            raise TargetError(f"API key missing: set {cfg.get('api_key_env', 'api_key')}")
        msgs: list[dict[str, str]] = []
        if cfg.get("system_prompt"):
            msgs.append({"role": "system", "content": cfg["system_prompt"]})
        tmpl: str = cfg.get("prompt_template") or ("Context:\n{context}\n\nQuestion: {input}" if case.get("context") else "{input}")
        msgs.append({"role": "user", "content": _fill(tmpl, case)})
        r = chat(msgs, cfg["model"], key, cfg.get("base_url", "https://api.openai.com/v1"),
                 max_tokens=cfg.get("max_tokens", 512), temperature=cfg.get("temperature", 0.0))
        if r is None:
            raise TargetError("LLM request failed")
        cost = (r["prompt_tokens"] * cfg.get("price_per_1m_input", 0) +
                r["completion_tokens"] * cfg.get("price_per_1m_output", 0)) / 1e6
        return {"output": r["text"], "latency_ms": r["latency_ms"], "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"], "cost": round(cost, 6)}

    if t == "http":
        body = _fill(cfg.get("body", {"input": "{input}"}), case)
        t0 = time.perf_counter()
        try:
            resp = httpx.request(cfg.get("method", "POST"), cfg["url"], json=body, headers=cfg.get("headers") or {},
                              timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json() if "json" in resp.headers.get("content-type", "") else resp.text
            out = _dig(data, cfg["response_path"]) if cfg.get("response_path") else data
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise TargetError(f"HTTP target failed: {e}")
        return {"output": out if isinstance(out, str) else json.dumps(out),
                "latency_ms": round((time.perf_counter() - t0) * 1000), "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}

    raise TargetError(f"unknown target type '{t}'")


def validate(cfg: dict[str, Any]) -> None:
    t = cfg.get("type")
    need = {"openai": ["model"], "http": ["url"], "replay": ["predictions"]}
    if t not in need:
        raise ValueError("type must be one of: openai, http, replay")
    missing = [k for k in need[t] if not cfg.get(k)]
    if missing:
        raise ValueError(f"{t} target needs: {', '.join(missing)}")
