import csv
import http.client
import json
import re
import threading
import hashlib
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from app.research.evaluator import _QuietHandler, build_evaluator


START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"


def _experiment(tmp_path: Path) -> Path:
    root = tmp_path / "experiment"
    for case_index in range(3):
        for variant in ("B00", "B01", "B10", "B11"):
            run_id = f"run-secret-{case_index}-{variant}"
            folder = root / "runs" / run_id
            folder.mkdir(parents=True)
            sample_id = f"sample-secret-{case_index}-{variant}"
            value = {
                "run_id": run_id,
                "case_id": f"case-secret-{case_index}",
                "variant_id": variant,
                "replicate": 0,
                "status": "completed",
                "samples": [{
                    "sample_id": sample_id,
                    "target_ply": 1,
                    "fen_before": START,
                    "move_san": "Nf3",
                    "comment": "Develops safely </script> without exposing identifiers.",
                    "facts": {
                        "display_line": {"start_fen": START, "line_san": ["Nf3"], "fens": [AFTER]},
                        "better_alternative": {"display_line": {"start_fen": START, "line_san": ["Nc3"], "fens": [AFTER]}},
                    },
                }],
            }
            (folder / "result.json").write_text(json.dumps(value), encoding="utf-8")
    return root


def _payload(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"const PACKET=(.*?);\nconst dims=", text, re.DOTALL)
    assert match
    return json.loads(match.group(1))


def _lock(path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path.with_suffix(".sha256").write_text(digest, encoding="utf-8")


def test_build_evaluator_blinds_payload_and_exports_compatible_columns(tmp_path):
    experiment = _experiment(tmp_path)
    output = tmp_path / "evaluator"
    result = build_evaluator(experiment, output, seed=17)

    assert result["samples_per_rater"] == 12
    assert (output / "private_key.csv").is_file()
    assert not (output / "public" / "private_key.csv").exists()
    html_a = output / "public" / "evaluator-a.html"
    html_b = output / "public" / "evaluator-b.html"
    payload_a, payload_b = _payload(html_a), _payload(html_b)
    assert len(payload_a["samples"]) == len(payload_b["samples"]) == 12
    ids_a = {item["anonymous_id"] for item in payload_a["samples"]}
    ids_b = {item["anonymous_id"] for item in payload_b["samples"]}
    assert ids_a.isdisjoint(ids_b)
    assert all(set(item) == {"anonymous_id", "rater_id", "fen", "move", "comment", "main", "alternative"} for item in payload_a["samples"])
    public_text = html_a.read_text(encoding="utf-8") + html_b.read_text(encoding="utf-8")
    for forbidden in ("sample-secret", "run-secret", "case-secret", '"variant_id"', '"sample_id"'):
        assert forbidden not in public_text
    assert "</script> without" not in public_text
    assert "const cols=['anonymous_id','rater_id','correctness','relevance','usefulness','concision','endpoint_adequacy','major_error','notes']" in public_text


def test_schedule_counterbalances_variants_and_separates_positions_by_round(tmp_path):
    output = tmp_path / "evaluator"
    build_evaluator(_experiment(tmp_path), output, seed=29)
    with (output / "private_key.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_assignment = {(row["rater_id"], row["case_id"], int(row["round"])): row for row in rows}
    variant_index = {name: index for index, name in enumerate(("B00", "B01", "B10", "B11"))}
    for case_index in range(3):
        case = f"case-secret-{case_index}"
        for round_number in range(1, 5):
            a = by_assignment[("evaluator-a", case, round_number)]
            b = by_assignment[("evaluator-b", case, round_number)]
            assert variant_index[a["variant_id"]] + variant_index[b["variant_id"]] == 3
        for rater in ("evaluator-a", "evaluator-b"):
            variants = {by_assignment[(rater, case, number)]["variant_id"] for number in range(1, 5)}
            assert variants == {"B00", "B01", "B10", "B11"}
    assert all(len({row["case_id"] for row in rows if row["rater_id"] == rater and int(row["round"]) == round_number}) == 3 for rater in ("evaluator-a", "evaluator-b") for round_number in range(1, 5))


def test_server_root_cannot_reach_private_key_or_parent_paths(tmp_path):
    output = tmp_path / "evaluator"
    build_evaluator(_experiment(tmp_path), output)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=str(output / "public")))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request("GET", "/evaluator-a.html")
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        for path in ("/../private_key.csv", "/%2e%2e/private_key.csv", "/..%2fprivate_key.csv"):
            connection.request("GET", path)
            denied = connection.getresponse()
            assert denied.status == 404
            denied.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_latest_attempt_only_is_used_and_failed_latest_cell_is_omitted(tmp_path):
    experiment = _experiment(tmp_path)
    original = experiment / "runs" / "run-secret-0-B00" / "result.json"
    base = json.loads(original.read_text(encoding="utf-8"))
    base.update({"attempt": 0, "experiment_id": "retry-exp"})
    original.write_text(json.dumps(base), encoding="utf-8")
    _lock(original)

    completed = {**base, "run_id": "run-secret-0-B00-a1", "attempt": 1}
    completed["samples"] = [{**base["samples"][0], "sample_id": "sample-secret-retry", "comment": "Latest completed wording."}]
    retry = experiment / "runs" / completed["run_id"] / "result.json"
    retry.parent.mkdir()
    retry.write_text(json.dumps(completed), encoding="utf-8")
    _lock(retry)

    failed = {**completed, "run_id": "run-secret-1-B01-a2", "case_id": "case-secret-1", "variant_id": "B01", "attempt": 2, "status": "failed", "samples": []}
    failed_path = experiment / "runs" / failed["run_id"] / "result.json"
    failed_path.parent.mkdir()
    failed_path.write_text(json.dumps(failed), encoding="utf-8")

    output = tmp_path / "evaluator"
    build_evaluator(experiment, output)
    payload = _payload(output / "public" / "evaluator-a.html")
    comments = [sample["comment"] for sample in payload["samples"]]
    assert comments.count("Latest completed wording.") == 1
    with (output / "private_key.csv").open(newline="", encoding="utf-8") as handle:
        assignments = list(csv.DictReader(handle))
    assert not any(row["sample_id"] == "sample-secret-0-B00" for row in assignments)
    assert sum(row["sample_id"] == "sample-secret-retry" for row in assignments) == 2
    assert not any(row["case_id"] == "case-secret-1" and row["variant_id"] == "B01" for row in assignments)

    retry.write_text(retry.read_text(encoding="utf-8").replace("Latest", "Tampered"), encoding="utf-8")
    try:
        build_evaluator(experiment, tmp_path / "tampered")
        raise AssertionError("tampered completed result was accepted")
    except ValueError as exc:
        assert "completed result hash mismatch" in str(exc)
