"""Load datasets from JSONL / JSON / CSV and seed the example data."""
import csv
import io
import json

from . import config

LIST_FIELDS = {"expected", "must_include", "must_not_include", "tags"}


def parse_dataset(filename: str, data: bytes) -> list[dict]:
    text = data.decode("utf-8-sig", errors="replace")
    name = filename.lower()
    if name.endswith(".jsonl"):
        cases = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    elif name.endswith(".json"):
        cases = json.loads(text)
        if isinstance(cases, dict):
            cases = cases.get("cases", [])
    elif name.endswith(".csv"):
        cases = []
        for row in csv.DictReader(io.StringIO(text)):
            case = {k.strip(): (v or "").strip() for k, v in row.items() if k and (v or "").strip()}
            for f in LIST_FIELDS & case.keys():  # "a | b" -> ["a", "b"]
                case[f] = [x.strip() for x in case[f].split("|") if x.strip()]
            if "should_refuse" in case:
                case["should_refuse"] = case["should_refuse"].lower() in ("1", "true", "yes")
            cases.append(case)
    else:
        raise ValueError("Upload a .jsonl, .json or .csv file")
    return validate_cases(cases)


def validate_cases(cases) -> list[dict]:
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset has no cases")
    seen = set()
    out = []
    for i, c in enumerate(cases, 1):
        if not isinstance(c, dict) or not str(c.get("input", "")).strip():
            raise ValueError(f"Case {i} needs an 'input'")
        c = dict(c)
        c["id"] = str(c.get("id") or f"case-{i}")
        if c["id"] in seen:
            raise ValueError(f"Duplicate case id '{c['id']}'")
        seen.add(c["id"])
        c.setdefault("tags", [])
        out.append(c)
    return out


EXAMPLES = [
    # (dataset file, description, [(target name, predictions file), ...])
    ("coaching-rag-qa.jsonl", "Questions to the RAG coaching chatbot, with retrieved context and reference answers",
     [("coaching-bot · prompt v1", "coaching-prompt-v1.json"), ("coaching-bot · prompt v2", "coaching-prompt-v2.json")]),
    ("appointment-extraction.jsonl", "Extract booking fields as JSON from patient messages",
     [("extractor · v1", "extraction-v1.json")]),
    ("safety.jsonl", "Harmful / private requests must be declined, normal ones answered",
     [("assistant · v1", "safety-v1.json")]),
]


def seed_examples(store) -> list[tuple[int, int, str]]:
    """Create example datasets + replay targets. Returns [(dataset_id, target_id, run_name)] to run."""
    if store.list_datasets():
        return []
    todo = []
    d = config.EXAMPLES_DIR
    for ds_file, desc, targets in EXAMPLES:
        path = d / "datasets" / ds_file
        ds = store.add_dataset(path.stem, parse_dataset(path.name, path.read_bytes()), desc)
        for tname, pfile in targets:
            preds = json.loads((d / "predictions" / pfile).read_text(encoding="utf-8"))
            t = store.add_target(tname, {"type": "replay", "predictions": preds})
            todo.append((ds["id"], t["id"], f"{path.stem} × {tname}"))
    # example live targets (need a key / a running app)
    store.add_target("groq · llama-3.1-8b (live)", {
        "type": "openai", "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.1-8b-instant",
        "api_key_env": "GROQ_API_KEY", "temperature": 0,
        "system_prompt": "Answer only from the context. Cite sources like [1]. If the answer is not in the context, say you don't have that information.",
        "price_per_1m_input": 0.05, "price_per_1m_output": 0.08})
    store.add_target("RAG-Coaching-Chatbot app (local HTTP)", {
        "type": "http", "url": "http://localhost:8001/api/ask",
        "body": {"session_id": "eval-{id}", "question": "{input}"}, "response_path": "answer"})
    return todo
