"""FastAPI app for the evaluation platform."""
import csv
import io
import json
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .datasets import parse_dataset, seed_examples
from .metrics import DEFAULT_METRICS, METRICS
from .runner import compare, execute_run
from .store import Store
from .targets import validate

STATIC = Path(__file__).resolve().parent.parent / "static"


class TargetIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    config: dict


class RunIn(BaseModel):
    dataset_id: int
    target_id: int
    name: str | None = None
    metrics: list[str] | None = None
    thresholds: dict[str, float] = {}
    wait: bool = False  # run synchronously (handy for CI / tests)


def start_run(store: Store, dataset_id: int, target_id: int, name: str | None = None,
              metrics: list[str] | None = None, thresholds: dict | None = None, wait: bool = True) -> int:
    ds, t = store.get_dataset(dataset_id), store.get_target(target_id)
    if not ds or not t:
        raise KeyError("dataset or target not found")
    metrics = metrics or DEFAULT_METRICS
    bad = [m for m in metrics if m not in METRICS]
    if bad:
        raise ValueError(f"unknown metrics: {', '.join(bad)}")
    run_id = store.create_run(name or f"{ds['name']} × {t['name']}", dataset_id, target_id, metrics,
                              thresholds or {}, ds["n_cases"])
    if wait:
        execute_run(store, run_id)
    else:
        threading.Thread(target=lambda: _safe(execute_run, store, run_id), daemon=True).start()
    return run_id


def _safe(fn, *a):
    try:
        fn(*a)
    except Exception:
        pass  # error is stored on the run


def create_app(db_path: str | None = None, seed: bool = True) -> FastAPI:
    app = FastAPI(title="AI Evaluation Platform", version="1.0.0")
    store = Store(db_path)
    app.state.store = store
    if seed:
        for ds_id, t_id, name in seed_examples(store):
            start_run(store, ds_id, t_id, name)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/api/metrics")
    def metrics():
        return [{"name": n, "threshold": t, "description": d, "default": n in DEFAULT_METRICS}
                for n, (_, t, d) in METRICS.items()]

    # datasets
    @app.get("/api/datasets")
    def datasets():
        return store.list_datasets()

    @app.get("/api/datasets/{ds_id}")
    def dataset(ds_id: int):
        d = store.get_dataset(ds_id)
        if not d:
            raise HTTPException(404, "Dataset not found")
        return d

    @app.post("/api/datasets", status_code=201)
    async def upload_dataset(file: UploadFile = File(...), name: str = Form(""), description: str = Form("")):
        raw = await file.read()
        try:
            cases = parse_dataset(file.filename or "data.jsonl", raw)
            return store.add_dataset(name or Path(file.filename or "dataset").stem, cases, description)
        except (ValueError, json.JSONDecodeError) as e:
            raise HTTPException(400, f"Invalid dataset: {e}")
        except Exception as e:
            if "UNIQUE" in str(e):
                raise HTTPException(409, "A dataset with that name exists")
            raise

    @app.delete("/api/datasets/{ds_id}")
    def delete_dataset(ds_id: int):
        try:
            if not store.delete_dataset(ds_id):
                raise HTTPException(404, "Dataset not found")
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    # targets
    @app.get("/api/targets")
    def targets():
        return store.list_targets()

    @app.post("/api/targets", status_code=201)
    def add_target(body: TargetIn):
        try:
            validate(body.config)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if store.find_target(body.name):
            raise HTTPException(409, "A target with that name exists")
        t = store.add_target(body.name, body.config)
        return store.get_target(t["id"], redact=True)

    @app.delete("/api/targets/{t_id}")
    def delete_target(t_id: int):
        try:
            if not store.delete_target(t_id):
                raise HTTPException(404, "Target not found")
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    # runs
    @app.get("/api/runs")
    def runs():
        return store.list_runs()

    @app.post("/api/runs", status_code=201)
    def new_run(body: RunIn):
        try:
            run_id = start_run(store, body.dataset_id, body.target_id, body.name, body.metrics, body.thresholds, body.wait)
        except KeyError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return store.get_run(run_id)

    @app.get("/api/runs/{run_id}")
    def run(run_id: int):
        r = store.get_run(run_id)
        if not r:
            raise HTTPException(404, "Run not found")
        r["results"] = store.results(run_id)
        return r

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: int):
        if not store.delete_run(run_id):
            raise HTTPException(404, "Run not found")
        return {"ok": True}

    @app.get("/api/runs/{run_id}/export.csv")
    def export(run_id: int):
        rows = store.results(run_id)
        if not rows:
            raise HTTPException(404, "No results")
        metric_names = sorted({m for r in rows for m in (r["scores"] or {})})
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["case_id", "input", "output", "passed", "error", "latency_ms", *metric_names])
        for r in rows:
            w.writerow([r["case_id"], r["input"], r["output"], r["passed"], r["error"] or "", r["latency_ms"] or "",
                        *[(r["scores"] or {}).get(m, {}).get("score", "") for m in metric_names]])
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": f"attachment; filename=run-{run_id}.csv"})

    @app.get("/api/compare")
    def compare_runs(a: int, b: int):
        try:
            return compare(store, a, b)
        except KeyError:
            raise HTTPException(404, "Run not found")
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app


app = create_app()
