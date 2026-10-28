"""Run evals from the terminal or in CI.

  python -m app.cli run --dataset examples/datasets/coaching-rag-qa.jsonl \\
      --target examples/targets/groq.json --fail-under 0.8

  # evaluate saved outputs (no API calls):
  python -m app.cli run --dataset examples/datasets/safety.jsonl \\
      --predictions examples/predictions/safety-v1.json

Exit code 1 when the pass rate is below --fail-under, so a CI job can block a bad prompt/model change.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .datasets import parse_dataset
from .metrics import DEFAULT_METRICS, METRICS
from .runner import execute_run
from .store import Store


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalkit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate a dataset against a target")
    r.add_argument("--dataset", required=True, help=".jsonl/.json/.csv")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--target", help="target config JSON file")
    g.add_argument("--predictions", help='JSON file {"case_id": "output"} to evaluate saved outputs')
    r.add_argument("--metrics", help="comma-separated (default: %s)" % ",".join(DEFAULT_METRICS))
    r.add_argument("--fail-under", type=float, default=None, help="minimum pass rate 0-1")
    r.add_argument("--db", default=":memory:", help="SQLite file to keep results (default: in-memory)")
    r.add_argument("--json", action="store_true", help="print the summary as JSON")
    sub.add_parser("metrics", help="list available metrics")
    args = ap.parse_args(argv)

    if args.cmd == "metrics":
        for n, (_, t, d) in METRICS.items():
            print(f"{n:<20} threshold {t:<5} {d}")
        return 0

    ds_path = Path(args.dataset)
    cases = parse_dataset(ds_path.name, ds_path.read_bytes())
    if args.target:
        cfg: dict[str, Any] = json.loads(Path(args.target).read_text(encoding="utf-8"))
        tname = Path(args.target).stem
    else:
        cfg = {"type": "replay", "predictions": json.loads(Path(args.predictions).read_text(encoding="utf-8"))}
        tname = Path(args.predictions).stem
    metrics: list[str] = [m.strip() for m in args.metrics.split(",")] if args.metrics else DEFAULT_METRICS
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        ap.error(f"unknown metrics: {', '.join(unknown)}")

    store = Store(args.db)
    ds = store.find_dataset(ds_path.stem) or store.add_dataset(ds_path.stem, cases)
    t = store.find_target(tname) or store.add_target(tname, cfg)
    run_id = store.create_run(f"{ds_path.stem} × {tname}", ds["id"], t["id"], metrics, {}, len(cases))

    def progress(i: int, n: int) -> None:
        print(f"\r  {i}/{n} cases", end="", file=sys.stderr, flush=True)

    summary = execute_run(store, run_id, progress)
    print(file=sys.stderr)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"\n{ds_path.stem} × {tname}: {summary['passed']}/{summary['cases']} passed "
              f"({summary['pass_rate']:.0%}), errors: {summary['errors']}")
        for m, v in summary["metrics"].items():
            print(f"  {m:<20} mean {v['mean']:.2f}   pass {v['pass_rate']:.0%}   (n={v['n']})")
        failed = [x for x in store.results(run_id) if not x["passed"]][:10]
        if failed:
            print("\nFailing cases:")
            for x in failed:
                why = x["error"] or "; ".join(f"{k}: {v['detail']}" for k, v in x["scores"].items() if not v["passed"])
                print(f"  - {x['case_id']}: {why[:140]}")
    if args.fail_under is not None and summary["pass_rate"] < args.fail_under:
        print(f"\nFAIL: pass rate {summary['pass_rate']:.0%} is below {args.fail_under:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
