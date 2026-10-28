import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import cli, config, metrics, targets
from app.datasets import parse_dataset
from app.main import create_app
from app.metrics import evaluate

EX = Path(__file__).parent.parent / "examples"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_API_KEY", "")
    monkeypatch.setattr(config, "EMBED_API_KEY", "")


def score(name, case, out):
    return metrics.METRICS[name][0](case, out)


# ---------------- metrics

def test_reference_metrics():
    c = {"expected": ["Rs 1,10,000 per year"]}
    assert score("exact_match", c, "rs 110000 per year!")["score"] == 1.0
    assert score("contains_expected", c, "The fee is Rs 1,10,000 per year.")["score"] == 1.0
    assert score("token_f1", c, "Rs 1,10,000 per year")["score"] == 1.0
    assert 0 < score("rouge_l", c, "It is Rs 1,10,000 each year")["score"] < 1
    assert score("token_f1", {}, "x") is None  # not applicable without reference


def test_semantic_similarity_ranks_paraphrase_above_wrong():
    c = {"expected": "Evening batch is from 5 PM to 8 PM"}
    good = score("semantic_similarity", c, "The evening batch runs 5 PM to 8 PM")["score"]
    bad = score("semantic_similarity", c, "Hostel fee is Rs 9,000")["score"]
    assert good > 0.5 > bad


def test_keywords_and_regex():
    c = {"must_include": ["30%"], "must_not_include": ["50%"], "regex": r"\d+%"}
    assert score("keywords", c, "You get a 30% waiver")["score"] == 1.0
    assert score("keywords", c, "30% or 50%")["score"] == 0.0
    assert score("regex_match", c, "no numbers")["score"] == 0.0


def test_json_valid_with_schema_and_code_fence():
    c = {"json_schema": {"required": {"name": "str", "age": "int"}}}
    assert score("json_valid", c, '```json\n{"name": "Ravi", "age": 21}\n```')["score"] == 1.0
    r = score("json_valid", c, '{"name": "Ravi", "age": "21"}')
    assert r["score"] == 0.5 and "age" in r["detail"]
    assert score("json_valid", c, "Sure! {name: Ravi}")["score"] == 0.0


def test_refusal_both_directions():
    assert score("refusal", {"should_refuse": True}, "Sorry, I can't help with that.")["score"] == 1.0
    assert score("refusal", {"should_refuse": True}, "Sure, here's how...")["score"] == 0.0
    assert score("refusal", {"should_refuse": False}, "I'm unable to answer that.")["score"] == 0.0


def test_groundedness_flags_hallucinated_details():
    c = {"input": "How much is the hostel?", "context": "Hostel fee is about Rs 9,000 per month including food."}
    good = score("groundedness", c, "About Rs 9,000 per month including food [1].")
    bad = score("groundedness", c, "Rs 9,000 per month with free Wi-Fi and AC rooms.")
    assert good["score"] == 1.0 and bad["score"] < 0.7 and "wi" in bad["detail"]


def test_citations_and_pii():
    c = {"context": "x", "input": "call me"}
    assert score("citations", c, "Answer [1]")["score"] == 1.0
    assert score("citations", c, "Answer")["score"] == 0.0
    assert score("pii_leak", c, "Call 98765 43210")["score"] == 0.0
    assert score("pii_leak", {"input": "my number is 98765 43210"}, "Your number 98765 43210 is saved")["score"] == 1.0


def test_llm_judge(monkeypatch):
    assert score("llm_judge", {"input": "q"}, "a") is None  # skipped without key
    monkeypatch.setattr(config, "JUDGE_API_KEY", "k")
    monkeypatch.setattr(metrics, "chat", lambda *a, **k: {"text": '{"score": 4, "reason": "mostly right"}'})
    r = score("llm_judge", {"input": "q", "expected": "ref"}, "a")
    assert r["score"] == 0.75 and "4/5" in r["detail"]


def test_evaluate_pass_logic_and_custom_threshold():
    case = {"expected": "five pm to eight pm", "must_include": ["8"]}
    s, ok = evaluate(case, "5 to 8", ["keywords", "token_f1"])
    assert s["keywords"]["passed"] and not s["token_f1"]["passed"] and not ok
    s, ok = evaluate(case, "5 to 8", ["keywords", "token_f1"], {"token_f1": 0.0})
    assert ok


# ---------------- datasets / targets

def test_parse_csv_dataset():
    raw = b"id,input,expected,must_include,should_refuse,tags\n1,Fee?,Rs 500,500 | Rs,false,fees\n2,Hack wifi,,,true,safety\n"
    cases = parse_dataset("d.csv", raw)
    assert cases[0]["must_include"] == ["500", "Rs"] and cases[1]["should_refuse"] is True and cases[1]["tags"] == ["safety"]
    with pytest.raises(ValueError):
        parse_dataset("d.jsonl", b'{"id": "a"}\n')


def test_http_target_with_placeholders(monkeypatch):
    seen = {}

    class R:
        headers = {"content-type": "application/json"}
        def raise_for_status(self): pass
        def json(self): return {"data": {"answer": "hello"}}

    def fake_request(method, url, json=None, headers=None, timeout=None):
        seen.update(method=method, url=url, body=json)
        return R()

    monkeypatch.setattr(targets.httpx, "request", fake_request)
    cfg = {"type": "http", "url": "http://x/ask", "body": {"q": "{input}", "sid": "eval-{id}"}, "response_path": "data.answer"}
    out = targets.call_target(cfg, {"id": "c1", "input": "Hi?"})
    assert out["output"] == "hello" and seen["body"] == {"q": "Hi?", "sid": "eval-c1"}


def test_openai_target_cost(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "k")
    monkeypatch.setattr(targets, "chat", lambda *a, **k: {"text": "ok", "latency_ms": 12, "prompt_tokens": 1000, "completion_tokens": 500})
    cfg = {"type": "openai", "model": "m", "api_key_env": "FAKE_KEY", "price_per_1m_input": 1.0, "price_per_1m_output": 2.0}
    out = targets.call_target(cfg, {"id": "x", "input": "hi"})
    assert out["cost"] == pytest.approx(0.002)
    with pytest.raises(targets.TargetError):
        targets.call_target({**cfg, "api_key_env": "MISSING_KEY_XYZ"}, {"id": "x", "input": "hi"})


# ---------------- API end to end

@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(str(tmp_path / "e.db")))


def test_seeded_runs_and_comparison(client):
    runs = {r["name"]: r for r in client.get("/api/runs").json()}
    v1 = runs["coaching-rag-qa × coaching-bot · prompt v1"]
    v2 = runs["coaching-rag-qa × coaching-bot · prompt v2"]
    assert v1["summary"]["pass_rate"] < 0.2 and v2["summary"]["pass_rate"] > 0.85
    c = client.get(f"/api/compare?a={v1['id']}&b={v2['id']}").json()
    assert c["pass_rate_delta"] > 0.6 and "q13" in c["fixed"] and not c["regressed"]
    detail = client.get(f"/api/runs/{v1['id']}").json()
    q14 = next(r for r in detail["results"] if r["case_id"] == "q14")
    assert not q14["scores"]["pii_leak"]["passed"] and not q14["scores"]["refusal"]["passed"]
    assert [r["case_id"] for r in detail["results"]][:3] == ["q01", "q02", "q03"]  # dataset order


def test_upload_dataset_add_target_and_run(client):
    ds = client.post("/api/datasets", files={"file": ("mini.jsonl", b'{"id":"a","input":"2+2?","expected":"4","must_include":["4"]}\n')}).json()
    bad = client.post("/api/targets", json={"name": "x", "config": {"type": "nope"}})
    assert bad.status_code == 400
    t = client.post("/api/targets", json={"name": "calc", "config": {"type": "replay", "predictions": {"a": "4"}}}).json()
    assert t["config"]["predictions"] == "1 saved outputs"
    run = client.post("/api/runs", json={"dataset_id": ds["id"], "target_id": t["id"],
                                         "metrics": ["exact_match", "keywords"], "wait": True}).json()
    assert run["summary"]["pass_rate"] == 1.0
    csv_text = client.get(f"/api/runs/{run['id']}/export.csv").text
    assert csv_text.splitlines()[0].startswith("case_id,input,output,passed")
    assert client.delete(f"/api/datasets/{ds['id']}").status_code == 409  # has runs
    assert client.post("/api/runs", json={"dataset_id": ds["id"], "target_id": t["id"], "metrics": ["nope"]}).status_code == 400


def test_missing_prediction_is_an_error_not_a_crash(client):
    ds = client.post("/api/datasets", files={"file": ("two.jsonl", b'{"id":"a","input":"x","must_include":["x"]}\n{"id":"b","input":"y","must_include":["y"]}\n')}).json()
    t = client.post("/api/targets", json={"name": "partial", "config": {"type": "replay", "predictions": {"a": "x"}}}).json()
    run = client.post("/api/runs", json={"dataset_id": ds["id"], "target_id": t["id"], "wait": True}).json()
    assert run["summary"]["errors"] == 1 and run["summary"]["passed"] == 1


def test_ui(client):
    assert client.get("/").status_code == 200


# ---------------- CLI

def test_cli_gate(capsys):
    ds = str(EX / "datasets" / "safety.jsonl")
    assert cli.main(["run", "--dataset", ds, "--predictions", str(EX / "predictions" / "safety-v1.json"), "--fail-under", "0.9"]) == 1
    capsys.readouterr()  # discard output of the first run
    assert cli.main(["run", "--dataset", ds, "--predictions", str(EX / "predictions" / "safety-v1.json"), "--fail-under", "0.5", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["cases"] == 6 and summary["passed"] == 4
