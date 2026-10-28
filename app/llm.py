"""OpenAI-compatible chat + embeddings client with latency and token usage."""
import math
import time

import httpx

from . import config


def chat(messages, model, api_key, base_url, max_tokens=512, temperature=0.0, json_mode=False):
    """Returns {"text", "latency_ms", "prompt_tokens", "completion_tokens"} or None on failure."""
    if not api_key:
        return None
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=body,
                       headers={"Authorization": f"Bearer {api_key}"}, timeout=config.REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    usage = data.get("usage") or {}
    return {"text": data["choices"][0]["message"].get("content") or "",
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)}


def embed(texts: list[str]) -> list[list[float]] | None:
    if not config.EMBED_API_KEY:
        return None
    try:
        r = httpx.post(f"{config.EMBED_BASE_URL.rstrip('/')}/embeddings",
                       headers={"Authorization": f"Bearer {config.EMBED_API_KEY}"},
                       json={"model": config.EMBED_MODEL, "input": texts}, timeout=config.REQUEST_TIMEOUT)
        r.raise_for_status()
        vecs = [d["embedding"] for d in r.json()["data"]]
    except Exception:
        return None
    out = []
    for v in vecs:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out
