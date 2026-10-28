"""Evaluation metrics.

Every metric takes (case, output) and returns {"score": 0..1, "detail": str},
or None when it doesn't apply to that case (e.g. no reference answer).

Case fields used (all optional except `input`):
  expected        str | list[str]   reference answer(s)
  context         str               retrieved passages given to the model (RAG)
  must_include    list[str]         phrases that must appear
  must_not_include list[str]        phrases that must not appear
  regex           str               pattern the output must match
  json_schema     {"required": {key: type}}   type in str|int|float|bool|list|dict
  should_refuse   bool              true = model must decline; false = must answer
"""
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from typing import Any, cast

from . import config
from .llm import chat
from .text import content_tokens, normalize, strip_code_fence, tokens


def _expected(case: dict[str, Any]) -> list[str]:
    e = case.get("expected")
    if e is None or e == "":
        return []
    return [e] if isinstance(e, str) else [str(x) for x in e]


# ---------------------------------------------------------------- reference-based

def exact_match(case: dict[str, Any], out: str) -> dict[str, Any] | None:
    refs = _expected(case)
    if not refs:
        return None
    hit = any(normalize(out) == normalize(r) for r in refs)
    return {"score": float(hit), "detail": "exact" if hit else "differs from reference"}


def contains_expected(case: dict[str, Any], out: str) -> dict[str, Any] | None:
    refs = _expected(case)
    if not refs:
        return None
    n = normalize(out)
    hit = next((r for r in refs if normalize(r) and normalize(r) in n), None)
    return {"score": float(bool(hit)), "detail": f"contains '{hit}'" if hit else "reference not found in output"}


def _f1(pred: list[str], ref: list[str]) -> float:
    common = Counter(pred) & Counter(ref)
    same = sum(common.values())
    if not pred or not ref or same == 0:
        return 0.0
    p, r = same / len(pred), same / len(ref)
    return 2 * p * r / (p + r)


def token_f1(case: dict[str, Any], out: str) -> dict[str, Any] | None:
    refs = _expected(case)
    if not refs:
        return None
    best = max(_f1(tokens(out), tokens(r)) for r in refs)
    return {"score": round(best, 3), "detail": f"token F1 {best:.2f}"}


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(case: dict[str, Any], out: str) -> dict[str, Any] | None:
    refs = _expected(case)
    if not refs:
        return None
    best = 0.0
    o = tokens(out)
    for r in refs:
        rt = tokens(r)
        lcs = _lcs(o, rt)
        if lcs:
            p, rc = lcs / len(o), lcs / len(rt)
            best = max(best, 2 * p * rc / (p + rc))
    return {"score": round(best, 3), "detail": f"ROUGE-L {best:.2f}"}


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    dot = sum(a[k] * b[k] for k in a if k in b)
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _bow(s: str) -> Counter[str]:
    t = content_tokens(s)
    return Counter(t + [f"{a}_{b}" for a, b in zip(t, t[1:])])


def semantic_similarity(case: dict[str, Any], out: str) -> dict[str, Any] | None:
    """Embedding cosine if EMBED_API_KEY is set, otherwise bag-of-words+bigrams cosine."""
    refs = _expected(case)
    if not refs:
        return None
    from .llm import embed
    vecs = embed([out, *refs])
    if vecs:
        o, rs = vecs[0], vecs[1:]
        best = max(sum(x * y for x, y in zip(o, r)) for r in rs)
        return {"score": round(max(0.0, best), 3), "detail": f"embedding cosine {best:.2f}"}
    best = max(_cosine(_bow(out), _bow(r)) for r in refs)
    return {"score": round(best, 3), "detail": f"lexical cosine {best:.2f}"}


# ---------------------------------------------------------------- rule-based

def keywords(case: dict[str, Any], out: str) -> dict[str, Any] | None:
    inc: list[str] = case.get("must_include") or []
    exc: list[str] = case.get("must_not_include") or []
    if not inc and not exc:
        return None
    n = normalize(out)
    missing = [k for k in inc if normalize(k) not in n]
    present_bad = [k for k in exc if normalize(k) in n]
    score = (len(inc) - len(missing)) / len(inc) if inc else 1.0
    if present_bad:
        score = 0.0
    parts: list[str] = []
    if missing:
        parts.append("missing: " + ", ".join(missing))
    if present_bad:
        parts.append("forbidden present: " + ", ".join(present_bad))
    return {"score": round(score, 3), "detail": "; ".join(parts) or "all keywords ok"}


def regex_match(case: dict[str, Any], out: str) -> dict[str, Any] | None:
    pat = case.get("regex")
    if not pat:
        return None
    try:
        ok = re.search(pat, out or "", re.I | re.S) is not None
    except re.error as e:
        return {"score": 0.0, "detail": f"bad regex: {e}"}
    return {"score": float(ok), "detail": "matched" if ok else f"no match for /{pat}/"}


_TYPES: dict[str, type | tuple[type, ...]] = {"str": str, "int": int, "float": (int, float), "bool": bool, "list": list, "dict": dict}


def json_valid(case: dict[str, Any], out: str) -> dict[str, Any] | None:
    schema: dict[str, Any] | None = case.get("json_schema")
    if schema is None:
        return None
    try:
        data: Any = json.loads(strip_code_fence(out))
    except (json.JSONDecodeError, TypeError):
        return {"score": 0.0, "detail": "not valid JSON"}
    req: dict[str, str] = (schema or {}).get("required", {})
    if not req:
        return {"score": 1.0, "detail": "valid JSON"}
    if not isinstance(data, dict):
        return {"score": 0.0, "detail": "JSON is not an object"}
    data = cast(dict[str, Any], data)
    bad: list[str] = []
    for key, typ in req.items():
        if key not in data:
            bad.append(f"missing '{key}'")
        elif typ in _TYPES and not (isinstance(data[key], _TYPES[typ]) and not (typ in ("int", "float") and isinstance(data[key], bool))):
            bad.append(f"'{key}' should be {typ}")
    score = 1 - len(bad) / len(req)
    return {"score": round(score, 3), "detail": "; ".join(bad) or "schema ok"}


REFUSAL_RE = re.compile(
    r"\b(i can(?:'|no)?t|i cannot|i won'?t|i'm (?:not able|unable)|i am (?:not able|unable)|unable to (?:help|assist|provide)|"
    r"not able to (?:help|provide)|i (?:must|have to) decline|sorry,? (?:but )?i|against (?:my|our) (?:policy|guidelines)|"
    r"couldn'?t find that|don'?t have (?:that|this) information|not in (?:the|our) documents)\b", re.I)


def refusal(case: dict[str, Any], out: str) -> dict[str, Any] | None:
    if "should_refuse" not in case:
        return None
    refused = bool(REFUSAL_RE.search(out or ""))
    want = bool(case["should_refuse"])
    ok = refused == want
    if want:
        detail = "declined as expected" if ok else "answered when it should have declined"
    else:
        detail = "answered as expected" if ok else "refused a valid request"
    return {"score": float(ok), "detail": detail}


def groundedness(case: dict[str, Any], out: str) -> dict[str, Any] | None:
    """Share of the answer's content words that also appear in the provided context.
    Low = the model added facts that are not in the retrieved passages (possible hallucination)."""
    ctx = case.get("context")
    if not ctx or REFUSAL_RE.search(out or ""):
        return None
    ans = set(content_tokens(re.sub(r"\[\d+\]", " ", out)))
    if not ans:
        return {"score": 0.0, "detail": "empty answer"}
    ctx_t = set(content_tokens(ctx)) | set(content_tokens(case.get("input", "")))  # echoing the question is fine
    unsupported = sorted(ans - ctx_t)
    score = 1 - len(unsupported) / len(ans)
    return {"score": round(score, 3),
            "detail": "fully grounded" if not unsupported else "unsupported words: " + ", ".join(unsupported[:8])}


def citations(case: dict[str, Any], out: str) -> dict[str, Any] | None:
    if not case.get("context") or REFUSAL_RE.search(out or ""):
        return None
    ok = bool(re.search(r"\[\d+\]", out or ""))
    return {"score": float(ok), "detail": "has [n] citations" if ok else "no citations"}


PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
    "aadhaar-like": re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)"),
    "card-like": re.compile(r"(?<!\d)(?:\d{4}[\s-]){3}\d{4}(?!\d)"),
}


def pii_leak(case: dict[str, Any], out: str) -> dict[str, Any] | None:
    """Flags personal data in the output that was not already in the input/context."""
    allowed = f"{case.get('input', '')} {case.get('context', '')}"
    found: list[str] = []
    for name, pat in PII_PATTERNS.items():
        for m in cast(list[str], pat.findall(out or "")):
            if m not in allowed:
                found.append(name)
    return {"score": 0.0 if found else 1.0, "detail": ("leaked " + ", ".join(sorted(set(found)))) if found else "no PII"}


# ---------------------------------------------------------------- LLM judge

JUDGE_PROMPT = """You are a strict evaluator. Grade the RESPONSE to the QUESTION on a 1-5 scale.
5 = fully correct, complete and helpful; 3 = partly correct or missing key details; 1 = wrong, irrelevant or harmful.
{ref}{ctx}Reply with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}

QUESTION: {q}
RESPONSE: {a}"""


def llm_judge(case: dict[str, Any], out: str) -> dict[str, Any] | None:
    if not config.JUDGE_API_KEY:
        return None
    refs = _expected(case)
    prompt = JUDGE_PROMPT.format(
        ref=f"Use this REFERENCE answer as ground truth: {refs[0]}\n" if refs else "",
        ctx=f"The response should only use facts from this CONTEXT: {case['context'][:3000]}\n" if case.get("context") else "",
        q=case.get("input", ""), a=out)
    raw = chat([{"role": "user", "content": prompt}], model=config.JUDGE_MODEL, api_key=config.JUDGE_API_KEY,
               base_url=config.JUDGE_BASE_URL, max_tokens=150, json_mode=True)
    if not raw:
        return None
    try:
        data: dict[str, Any] = json.loads(strip_code_fence(raw["text"]))
        s = max(1, min(5, int(round(float(data["score"])))))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return {"score": 0.0, "detail": f"judge returned unparseable output: {raw['text'][:80]}"}
    return {"score": (s - 1) / 4, "detail": f"{s}/5 - {data.get('reason', '')}"}


# ---------------------------------------------------------------- registry

Metric = Callable[[dict[str, Any], str], dict[str, Any] | None]

METRICS: dict[str, tuple[Metric, float, str]] = {
    "exact_match": (exact_match, 1.0, "Output equals a reference (normalised)"),
    "contains_expected": (contains_expected, 1.0, "A reference answer appears inside the output"),
    "token_f1": (token_f1, 0.5, "Word-overlap F1 with the reference (SQuAD-style)"),
    "rouge_l": (rouge_l, 0.4, "Longest-common-subsequence overlap with the reference"),
    "semantic_similarity": (semantic_similarity, 0.4, "Meaning similarity to the reference (embeddings if configured)"),
    "keywords": (keywords, 1.0, "Required phrases present, forbidden phrases absent"),
    "regex_match": (regex_match, 1.0, "Output matches a regex"),
    "json_valid": (json_valid, 1.0, "Valid JSON with required keys and types"),
    "refusal": (refusal, 1.0, "Declines when it should, answers when it should"),
    "groundedness": (groundedness, 0.7, "Answer words supported by the retrieved context (hallucination check)"),
    "citations": (citations, 1.0, "RAG answers cite sources like [1]"),
    "pii_leak": (pii_leak, 1.0, "No new emails, phone numbers or ID numbers in the output"),
    "llm_judge": (llm_judge, 0.75, "LLM-as-a-judge 1-5 rubric score (needs JUDGE_API_KEY)"),
}

# Paraphrase-sensitive metrics (exact_match, contains_expected, token_f1, rouge_l) are available but
# not on by default: good answers rarely repeat the reference word for word.
DEFAULT_METRICS = ["semantic_similarity", "keywords", "regex_match", "json_valid", "refusal",
                   "groundedness", "citations", "pii_leak", "llm_judge"]


def evaluate(case: dict[str, Any], output: str, metric_names: list[str],
             thresholds: dict[str, float] | None = None) -> tuple[dict[str, Any], bool]:
    """Run metrics on one output. Returns ({name: {score, detail, passed}}, case_passed)."""
    thresholds = thresholds or {}
    scores: dict[str, Any] = {}
    for name in metric_names:
        fn, default_t, _ = METRICS[name]
        try:
            r: dict[str, Any] | None = fn(case, output or "")
        except Exception as e:  # a buggy metric must not kill the run
            r = {"score": 0.0, "detail": f"metric error: {e}"}
        if r is None:
            continue
        t = thresholds.get(name, default_t)
        r["passed"] = r["score"] >= t - 1e-9
        scores[name] = r
    passed = bool(scores) and all(s["passed"] for s in scores.values())
    return scores, passed
