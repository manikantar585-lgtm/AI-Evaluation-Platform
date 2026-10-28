"""SQLite storage: datasets, targets, runs and per-case results."""
import json
import sqlite3
import threading
from collections.abc import Sequence
from typing import Any

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, description TEXT DEFAULT '',
    cases TEXT NOT NULL, created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, config TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    target_id INTEGER NOT NULL REFERENCES targets(id), metrics TEXT NOT NULL, thresholds TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'queued', progress INTEGER DEFAULT 0, total INTEGER DEFAULT 0,
    summary TEXT, error TEXT, created_at TEXT DEFAULT (datetime('now')), finished_at TEXT);
CREATE TABLE IF NOT EXISTS results (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE, case_id TEXT NOT NULL,
    input TEXT, output TEXT, expected TEXT, tags TEXT, scores TEXT, passed INTEGER, error TEXT,
    latency_ms INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER, cost REAL,
    PRIMARY KEY (run_id, case_id));
"""

Row = dict[str, Any]


def _j(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def _id(cur: sqlite3.Cursor) -> int:
    assert cur.lastrowid is not None
    return cur.lastrowid


class Store:
    def __init__(self, path: str | None = None) -> None:
        self.conn = sqlite3.connect(path or config.DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.lock = threading.RLock()

    def _one(self, q: str, args: Sequence[Any] = ()) -> Row | None:
        with self.lock:
            r = self.conn.execute(q, args).fetchone()
        return dict(r) if r else None

    def _all(self, q: str, args: Sequence[Any] = ()) -> list[Row]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(q, args)]

    def _exec(self, q: str, args: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self.lock, self.conn:
            return self.conn.execute(q, args)

    # datasets
    def add_dataset(self, name: str, cases: list[Row], description: str = "") -> Row:
        cur = self._exec("INSERT INTO datasets (name, description, cases) VALUES (?,?,?)", (name, description, _j(cases)))
        d = self.get_dataset(_id(cur))
        assert d is not None
        return d

    def get_dataset(self, ds_id: int, with_cases: bool = True) -> Row | None:
        d = self._one("SELECT * FROM datasets WHERE id=?", (ds_id,))
        if d:
            d["cases"] = json.loads(d["cases"])
            d["n_cases"] = len(d["cases"])
            if not with_cases:
                d.pop("cases")
        return d

    def find_dataset(self, name: str) -> Row | None:
        d = self._one("SELECT id FROM datasets WHERE name=?", (name,))
        return self.get_dataset(d["id"]) if d else None

    def list_datasets(self) -> list[Row]:
        rows = (self.get_dataset(r["id"], with_cases=False) for r in self._all("SELECT id FROM datasets ORDER BY id"))
        return [d for d in rows if d]

    def delete_dataset(self, ds_id: int) -> bool:
        if self._one("SELECT 1 FROM runs WHERE dataset_id=?", (ds_id,)):
            raise ValueError("dataset has runs; delete them first")
        return self._exec("DELETE FROM datasets WHERE id=?", (ds_id,)).rowcount > 0

    # targets
    def add_target(self, name: str, cfg: Row) -> Row:
        cur = self._exec("INSERT INTO targets (name, config) VALUES (?,?)", (name, _j(cfg)))
        t = self.get_target(_id(cur))
        assert t is not None
        return t

    def get_target(self, t_id: int, redact: bool = False) -> Row | None:
        t = self._one("SELECT * FROM targets WHERE id=?", (t_id,))
        if t:
            t["config"] = json.loads(t["config"])
            if redact:
                t["config"].pop("api_key", None)
                if t["config"].get("type") == "replay":
                    t["config"]["predictions"] = f"{len(t['config'].get('predictions') or {})} saved outputs"
        return t

    def find_target(self, name: str) -> Row | None:
        t = self._one("SELECT id FROM targets WHERE name=?", (name,))
        return self.get_target(t["id"]) if t else None

    def list_targets(self) -> list[Row]:
        rows = (self.get_target(r["id"], redact=True) for r in self._all("SELECT id FROM targets ORDER BY id"))
        return [t for t in rows if t]

    def delete_target(self, t_id: int) -> bool:
        if self._one("SELECT 1 FROM runs WHERE target_id=?", (t_id,)):
            raise ValueError("target has runs; delete them first")
        return self._exec("DELETE FROM targets WHERE id=?", (t_id,)).rowcount > 0

    # runs
    def create_run(self, name: str, dataset_id: int, target_id: int, metrics: list[str],
                   thresholds: dict[str, float], total: int) -> int:
        cur = self._exec("INSERT INTO runs (name, dataset_id, target_id, metrics, thresholds, total) VALUES (?,?,?,?,?,?)",
                         (name, dataset_id, target_id, _j(metrics), _j(thresholds), total))
        return _id(cur)

    def update_run(self, run_id: int, **fields: Any) -> None:
        if "summary" in fields:
            fields["summary"] = _j(fields["summary"])
        sets = ", ".join(f"{k}=?" for k in fields)
        if fields.get("status") in ("done", "failed"):
            sets += ", finished_at=datetime('now')"
        self._exec(f"UPDATE runs SET {sets} WHERE id=?", (*fields.values(), run_id))

    def get_run(self, run_id: int) -> Row | None:
        r = self._one("SELECT r.*, d.name AS dataset, t.name AS target FROM runs r JOIN datasets d ON d.id=r.dataset_id "
                      "JOIN targets t ON t.id=r.target_id WHERE r.id=?", (run_id,))
        if r:
            for k in ("metrics", "thresholds", "summary"):
                r[k] = json.loads(r[k]) if r[k] else None
        return r

    def list_runs(self) -> list[Row]:
        rows = (self.get_run(x["id"]) for x in self._all("SELECT id FROM runs ORDER BY id DESC"))
        return [r for r in rows if r]

    def delete_run(self, run_id: int) -> bool:
        return self._exec("DELETE FROM runs WHERE id=?", (run_id,)).rowcount > 0

    def save_result(self, run_id: int, row: Row) -> None:
        self._exec(
            "INSERT OR REPLACE INTO results (run_id, case_id, input, output, expected, tags, scores, passed, error, "
            "latency_ms, prompt_tokens, completion_tokens, cost) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, row["case_id"], row["input"], row["output"], _j(row["expected"]), _j(row["tags"]), _j(row["scores"]),
             int(row["passed"]), row["error"], row["latency_ms"], row["prompt_tokens"], row["completion_tokens"], row["cost"]))

    def results(self, run_id: int) -> list[Row]:
        rows = self._all("SELECT * FROM results WHERE run_id=?", (run_id,))
        run = self._one("SELECT dataset_id FROM runs WHERE id=?", (run_id,))
        ds = self.get_dataset(run["dataset_id"]) if run else None
        order: dict[str, int] = {c["id"]: i for i, c in enumerate(ds["cases"])} if ds else {}
        rows.sort(key=lambda r: order.get(r["case_id"], len(order)))
        for r in rows:
            for k in ("expected", "tags", "scores"):
                r[k] = json.loads(r[k]) if r[k] else None
            r["passed"] = bool(r["passed"])
        return rows
