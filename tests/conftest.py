"""Regression tests describe the v18 behaviour: pin the pipeline profile so a shell with
ACA_PROFILE=v19 (or a per-flag override) does not change what they assert. The switch
tests in test_v19_flags.py set the switches they need themselves."""
import pytest


@pytest.fixture(autouse=True)
def _v18_pipeline_profile(monkeypatch):
    monkeypatch.setenv("ACA_PROFILE", "v18")
    from app.core.commentary import pipeline_flags
    for name in pipeline_flags.FLAGS:
        monkeypatch.delenv(f"ACA_V19_{name.upper()}", raising=False)
