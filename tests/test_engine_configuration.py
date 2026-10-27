"""Mock-only tests for environment-driven UCI engine configuration."""

import os
from unittest.mock import patch

import pytest

from app.core.engine.engine_connector import EngineConnector


class _Engine:
    def __init__(self, options):
        self.options = {name: object() for name in options}
        self.configured = []

    def configure(self, options):
        self.configured.append(options)


def _connector(engine, env):
    with patch.dict(os.environ, env, clear=True), patch(
        "app.core.engine.engine_connector.chess.engine.SimpleEngine.popen_uci",
        return_value=engine,
    ):
        connector = EngineConnector("mock-engine")
    return connector


def test_default_config_preserves_existing_values_when_supported():
    engine = _Engine(["Hash", "Threads", "Use NNUE"])
    connector = _connector(engine, {})
    assert engine.configured == [{"Hash": 32, "Threads": 4, "Use NNUE": False}]
    assert connector.hash_size == 32
    assert connector.threads == 4
    assert connector.use_nnue is False


def test_unsupported_nnue_is_not_sent_or_reported_as_disabled():
    engine = _Engine(["Hash", "Threads"])
    connector = _connector(engine, {})
    assert engine.configured == [{"Hash": 32, "Threads": 4}]
    assert connector.use_nnue is None


def test_positive_environment_values_are_applied():
    engine = _Engine(["Hash", "Threads", "Use NNUE"])
    connector = _connector(
        engine,
        {
            "STOCKFISH_THREADS": "6",
            "STOCKFISH_HASH_MB": "128",
            "STOCKFISH_USE_NNUE": "yes",
        },
    )
    assert engine.configured == [{"Hash": 128, "Threads": 6, "Use NNUE": True}]
    assert connector.threads == 6
    assert connector.hash_size == 128
    assert connector.use_nnue is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("STOCKFISH_THREADS", "0"),
        ("STOCKFISH_THREADS", "lots"),
        ("STOCKFISH_HASH_MB", "-1"),
        ("STOCKFISH_HASH_MB", "bad"),
        ("STOCKFISH_USE_NNUE", "sometimes"),
    ],
)
def test_malformed_environment_values_fail_clearly(name, value):
    with patch.dict(os.environ, {name: value}, clear=True), pytest.raises(
        ValueError, match=name
    ):
        EngineConnector("mock-engine")
