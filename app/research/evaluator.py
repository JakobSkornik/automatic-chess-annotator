"""Build and serve anonymous, offline browser packets for human evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable


_TEMPLATE = Path(__file__).with_name("evaluator.html")
_PAYLOAD_MARKER = "__EVALUATOR_PAYLOAD__"
_PRIVATE_FIELDS = ("anonymous_id", "sample_id", "run_id", "case_id", "variant_id", "replicate", "rater_id", "round", "order")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _runner_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _verify_optional_lock(path: Path, value: Any, label: str) -> None:
    lock = path.with_suffix(".sha256")
    if lock.is_file() and lock.read_text(encoding="utf-8").strip() != _runner_digest(value):
        raise ValueError(f"{label} hash mismatch: {path}")


def _opaque_id(seed: int, rater_id: str, sample_id: str) -> str:
    # Stable IDs make a frozen packet reproducible. The private inputs cannot be
    # recovered from this one-way digest and are never copied into public files.
    material = f"evaluator-v1\0{seed}\0{rater_id}\0{sample_id}".encode("utf-8")
    return "E" + hashlib.sha256(material).hexdigest()[:23]


def _line(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    start = str(value.get("start_fen") or "")
    sans = [str(item) for item in value.get("line_san", [])]
    fens = [str(item) for item in value.get("fens", [])]
    if not start or len(sans) != len(fens):
        return None
    return {"start_fen": start, "line_san": sans, "fens": fens}


def _public_sample(sample: dict[str, Any], anonymous_id: str, rater_id: str) -> dict[str, Any]:
    facts = sample.get("facts") if isinstance(sample.get("facts"), dict) else {}
    alternative = facts.get("better_alternative") if isinstance(facts.get("better_alternative"), dict) else {}
    return {
        "anonymous_id": anonymous_id,
        "rater_id": rater_id,
        "fen": str(sample.get("fen_before") or ""),
        "move": str(sample.get("move_san") or ""),
        "comment": str(sample.get("comment") or ""),
        "main": _line(facts.get("display_line")),
        "alternative": _line(alternative.get("display_line")),
    }


def _collect(experiment_dir: Path) -> dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]]:
    manifest_path = experiment_dir / "manifest.json"
    manifest = _load(manifest_path) if manifest_path.is_file() else None
    if manifest is not None:
        _verify_optional_lock(manifest_path, manifest, "manifest")
    latest: dict[tuple[str, str, int], tuple[int, Path, dict[str, Any]]] = {}
    for path in sorted((experiment_dir / "runs").glob("*/result.json")):
        result = _load(path)
        identity = (
            str(result.get("case_id") or ""),
            str(result.get("variant_id") or ""),
            int(result.get("replicate", 0)),
        )
        attempt = int(result.get("attempt", 0))
        current = latest.get(identity)
        if current is None or attempt > current[0]:
            latest[identity] = (attempt, path, result)

    groups: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for _, path, result in sorted(latest.values(), key=lambda item: str(item[1])):
        if result.get("status") != "completed":
            continue
        _verify_optional_lock(path, result, "completed result")
        if manifest is not None:
            expected_experiment = manifest.get("experiment_id")
            if expected_experiment and result.get("experiment_id") not in (None, expected_experiment):
                raise ValueError(f"result experiment_id does not match manifest: {path}")
        for sample in result.get("samples", []) or []:
            if not isinstance(sample, dict) or not sample.get("sample_id"):
                continue
            key = (str(result.get("case_id") or ""), int(sample.get("target_ply", 0)))
            groups.setdefault(key, []).append((result, sample))
    if not groups:
        raise ValueError("experiment contains no completed samples")
    for key, cells in groups.items():
        identities = [(str(r.get("variant_id")), int(r.get("replicate", 0))) for r, _ in cells]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate variant/replicate samples for position {key}")
        cells.sort(key=lambda pair: (str(pair[0].get("variant_id")), int(pair[0].get("replicate", 0))))
    return groups


def _schedule(groups: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]], rater_index: int, seed: int) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    """Interleave positions in rounds; reverse each position's cells for rater B."""
    keys = sorted(groups)
    max_rounds = max(len(cells) for cells in groups.values())
    scheduled: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for round_index in range(max_rounds):
        round_keys = [key for key in keys if round_index < len(groups[key])]
        random.Random(seed + 1009 * rater_index + 7919 * round_index).shuffle(round_keys)
        for position_index, key in enumerate(round_keys):
            cells = groups[key]
            # Rotate by position so no variant systematically occupies an early
            # round, then reverse for the second evaluator (Williams-style pair).
            offset = keys.index(key) % len(cells)
            index = (round_index + offset) % len(cells)
            if rater_index % 2:
                index = len(cells) - 1 - index
            result, sample = cells[index]
            scheduled.append((round_index + 1, result, sample))
    return scheduled


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_evaluator(
    experiment_dir: Path,
    output_dir: Path,
    rater_ids: tuple[str, ...] = ("evaluator-a", "evaluator-b"),
    seed: int = 20260905,
) -> dict[str, Any]:
    """Create one standalone blinded HTML packet per evaluator.

    ``output_dir/public`` is safe to serve or copy to raters. The assignment key
    remains one level above it and must stay with the researcher.
    """
    experiment_dir = Path(experiment_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if len(rater_ids) < 2 or len(set(rater_ids)) != len(rater_ids):
        raise ValueError("at least two distinct rater_ids are required")
    if any(not value or any(char in value for char in "/\\") for value in rater_ids):
        raise ValueError("rater_ids must be non-empty file-safe labels")
    template = _TEMPLATE.read_text(encoding="utf-8")
    if template.count(_PAYLOAD_MARKER) != 1:
        raise ValueError("evaluator template must contain exactly one payload marker")
    groups = _collect(experiment_dir)
    public = output_dir / "public"
    public.mkdir(parents=True, exist_ok=True)
    assignments: list[dict[str, Any]] = []
    packet_summaries: list[dict[str, Any]] = []

    for rater_index, rater_id in enumerate(rater_ids):
        schedule = _schedule(groups, rater_index, seed)
        samples: list[dict[str, Any]] = []
        for order, (round_number, result, sample) in enumerate(schedule, start=1):
            sample_id = str(sample["sample_id"])
            anonymous_id = _opaque_id(seed, rater_id, sample_id)
            samples.append(_public_sample(sample, anonymous_id, rater_id))
            assignments.append({
                "anonymous_id": anonymous_id,
                "sample_id": sample_id,
                "run_id": str(result.get("run_id") or ""),
                "case_id": str(result.get("case_id") or ""),
                "variant_id": str(result.get("variant_id") or ""),
                "replicate": int(result.get("replicate", 0)),
                "rater_id": rater_id,
                "round": round_number,
                "order": order,
            })
        packet_id = "P" + hashlib.sha256(json.dumps([seed, rater_id, samples], sort_keys=True).encode()).hexdigest()[:15]
        payload = {"packet_id": packet_id, "rater_id": rater_id, "samples": samples}
        filename = f"{rater_id}.html"
        (public / filename).write_text(template.replace(_PAYLOAD_MARKER, _json_for_script(payload)), encoding="utf-8")
        packet_summaries.append({"rater_id": rater_id, "packet_id": packet_id, "file": filename, "samples": len(samples)})

    _write_csv(output_dir / "private_key.csv", assignments, _PRIVATE_FIELDS)
    public_manifest = {
        "format": "chess-comment-evaluator-v1",
        "seed": seed,
        "packets": packet_summaries,
        "rating_columns": ["correctness", "relevance", "usefulness", "concision", "endpoint_adequacy", "major_error", "notes"],
    }
    (public / "manifest.json").write_text(json.dumps(public_manifest, indent=2), encoding="utf-8")
    return {
        "public_dir": str(public),
        "private_key_path": str(output_dir / "private_key.csv"),
        "packets": packet_summaries,
        "samples_per_rater": len(assignments) // len(rater_ids),
        "seed": seed,
    }


class _QuietHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        candidate = Path(super().translate_path(path)).resolve()
        root = Path(self.directory or ".").resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return str(root / ".not-found")
        return str(candidate)

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover - cosmetic
        return


def serve_evaluator(output_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve only ``output_dir/public``; private keys and experiments are unreachable."""
    public = (Path(output_dir).resolve() / "public")
    if not public.is_dir():
        raise FileNotFoundError(f"public evaluator directory does not exist: {public}")
    handler = partial(_QuietHandler, directory=str(public))
    with ThreadingHTTPServer((host, port), handler) as server:
        server.serve_forever()
