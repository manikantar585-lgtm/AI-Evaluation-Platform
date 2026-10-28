"""Run a dataset against a target, score every case, summarise, and compare runs."""
import statistics
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from . import config
from .metrics import METRICS, evaluate
from .store import Row, Store
from .targets import TargetError, call_target


def run_case(target_cfg: Row, case: Row, metric_names: list[str], thresholds: dict[str, float] | None) -> Row:
    row: Row = {"case_id": case["id"], "input": case.get("input", ""), "expected": case.get("expected"),
           "tags": case.get("tags", []), "output": "", "scores": {}, "passed": False, "error": None,
           "latency_ms": None, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
    try:
        res = call_target(target_cfg, case)
    except TargetError as e:
        row["error"] = str(e)
        return row
    row.update({k: res[k] for k in ("latency_ms", "prompt_tokens", "completion_tokens", "cost")})
    row["output"] = res["output"]
    row["scores"], row["passed"] = evaluate(case, res["output"], metric_names, thresholds)
    return row


def summarize(rows: list[Row]) -> Row:
    n = len(rows)
    ok = [r for r in rows if not r["error"]]
    per_metric: dict[str, list[float]] = {}
    metric_pass: dict[str, list[bool]] = {}
    for r in ok:
        for name, s in r["scores"].items():
            per_metric.setdefault(name, []).append(s["score"])
            metric_pass.setdefault(name, []).append(s["passed"])
    by_tag: dict[str, list[bool]] = {}
    for r in rows:
        for t in r["tags"] or ["untagged"]:
            by_tag.setdefault(t, []).append(r["passed"])
    lat: list[int] = [r["latency_ms"] for r in ok if r["latency_ms"] is not None]
    return {
        "cases": n,
        "passed": sum(r["passed"] for r in rows),
        "pass_rate": round(sum(r["passed"] for r in rows) / n, 4) if n else 0.0,
        "errors": n - len(ok),
        "metrics": {m: {"mean": round(statistics.mean(v), 4), "pass_rate": round(sum(metric_pass[m]) / len(v), 4), "n": len(v)}
                    for m, v in per_metric.items()},
        "by_tag": {t: {"pass_rate": round(sum(v) / len(v), 4), "n": len(v)} for t, v in sorted(by_tag.items())},
        "latency_ms": {"p50": round(statistics.median(lat)), "p95": round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)]),
                       "mean": round(statistics.mean(lat))} if lat else None,
        "tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in ok),
        "cost": round(sum(r["cost"] or 0 for r in ok), 6),
    }


def execute_run(store: Store, run_id: int, progress_cb: Callable[[int, int], Any] | None = None) -> Row:
    run = store.get_run(run_id)
    if not run:
        raise KeyError("run not found")
    ds = store.get_dataset(run["dataset_id"])
    target = store.get_target(run["target_id"])
    if not ds or not target:
        raise KeyError("dataset or target not found")
    metric_names = [m for m in run["metrics"] if m in METRICS]
    store.update_run(run_id, status="running")
    rows: list[Row] = []
    try:
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
            futs = [pool.submit(run_case, target["config"], c, metric_names, run["thresholds"]) for c in ds["cases"]]
            for i, f in enumerate(as_completed(futs), 1):
                row = f.result()
                store.save_result(run_id, row)
                rows.append(row)
                store.update_run(run_id, progress=i)
                if progress_cb:
                    progress_cb(i, len(futs))
        order: dict[str, int] = {c["id"]: i for i, c in enumerate(ds["cases"])}
        rows.sort(key=lambda r: order.get(r["case_id"], 0))
        summary = summarize(rows)
        store.update_run(run_id, status="done", summary=summary)
        return summary
    except Exception as e:
        store.update_run(run_id, status="failed", error=str(e))
        raise


def compare(store: Store, run_a: int, run_b: int) -> Row:
    """B relative to A: per-metric deltas and per-case fixed / regressed lists."""
    a, b = store.get_run(run_a), store.get_run(run_b)
    if not a or not b:
        raise KeyError("run not found")
    if a["dataset_id"] != b["dataset_id"]:
        raise ValueError("runs must use the same dataset")
    ra = {r["case_id"]: r for r in store.results(run_a)}
    rb = {r["case_id"]: r for r in store.results(run_b)}
    fixed: list[str] = []
    regressed: list[str] = []
    cases: list[Row] = []
    for cid in ra.keys() & rb.keys():
        x, y = ra[cid], rb[cid]
        if not x["passed"] and y["passed"]:
            fixed.append(cid)
        elif x["passed"] and not y["passed"]:
            regressed.append(cid)
        changed = {m: round(y["scores"].get(m, {}).get("score", 0) - x["scores"].get(m, {}).get("score", 0), 3)
                   for m in set(x["scores"] or {}) | set(y["scores"] or {})}
        cases.append({"case_id": cid, "input": x["input"], "a_output": x["output"], "b_output": y["output"],
                      "a_passed": x["passed"], "b_passed": y["passed"], "deltas": changed})
    order = list(ra.keys())
    cases.sort(key=lambda c: (c["a_passed"] == c["b_passed"], order.index(c["case_id"])))
    sa: Row = a["summary"] or {}
    sb: Row = b["summary"] or {}
    metrics: dict[str, Row] = {}
    for m in set(sa.get("metrics", {})) | set(sb.get("metrics", {})):
        va, vb = sa.get("metrics", {}).get(m, {}).get("mean"), sb.get("metrics", {}).get(m, {}).get("mean")
        metrics[m] = {"a": va, "b": vb, "delta": round(vb - va, 4) if va is not None and vb is not None else None}
    return {"a": {"id": a["id"], "name": a["name"], "pass_rate": sa.get("pass_rate")},
            "b": {"id": b["id"], "name": b["name"], "pass_rate": sb.get("pass_rate")},
            "pass_rate_delta": round((sb.get("pass_rate") or 0) - (sa.get("pass_rate") or 0), 4),
            "metrics": dict(sorted(metrics.items())), "fixed": fixed, "regressed": regressed, "cases": cases}
