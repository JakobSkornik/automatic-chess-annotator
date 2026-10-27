import io
import asyncio
import json
import zipfile

import pytest

from app.core.output_paths import dated_game_path
from app.routes.jobs import export_all_games, get_job_game


def _game(job_id: str) -> dict:
    return {
        "metadata": {"id": job_id, "white": "A", "black": "B", "result": "*"},
        "moves": [],
        "analysis_info": {"engine": "test", "depth": 1, "multipv": 1, "timestamp": 0},
    }


def test_game_endpoint_and_zip_use_dated_output(tmp_path, monkeypatch):
    monkeypatch.setenv("ACA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.chdir(tmp_path)
    job_id = "route-output-test"
    path = dated_game_path(job_id, "2026-10-24")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_game(job_id)), encoding="utf-8")

    assert asyncio.run(get_job_game(job_id))["metadata"]["id"] == job_id
    response = asyncio.run(export_all_games())
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        assert archive.namelist() == [f"{job_id}.json"]


def test_game_endpoint_rejects_traversal():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        asyncio.run(get_job_game("../escape"))
    assert exc.value.status_code == 400
