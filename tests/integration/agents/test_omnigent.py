"""The atomic Omnigent backend, driven against a stand-in executable."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from hmz.coganchor.agents import (
    DRIVEN,
    Failed,
    OmnigentAgent,
    OmnigentAgentConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

_MODEL = "databricks-glm-5-3"
_CONFIG = OmnigentAgentConfig(model=_MODEL, effort="auto")

_FAKE = """
import json, os, pathlib, sys

with pathlib.Path(os.environ["OMNIGENT_TEST_LOG"]).open("a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")

prompt = sys.argv[sys.argv.index("-p") + 1]
if prompt == "malformed":
    print("not json")
else:
    print("starting ephemeral execution", file=sys.stderr)
    print(json.dumps({
        "type": "result",
        "session_id": "run_123",
        "text": f"finished {prompt}",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 150,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 0,
            "total_cost_usd": 0.42,
            "models": {
                "databricks-glm-5-3": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 150,
                    "cache_read_input_tokens": 30,
                    "cache_creation_input_tokens": 0,
                    "total_cost_usd": 0.42,
                }
            },
        },
    }))
"""


@pytest.fixture
def omnigent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Put a receipt-writing Omnigent stand-in first on PATH."""
    binary = tmp_path / "omnigent"
    log = tmp_path / "calls.jsonl"
    binary.write_text(f"#!{sys.executable}\n{_FAKE}")
    binary.chmod(0o755)
    monkeypatch.setenv("OMNIGENT_TEST_LOG", str(log))
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return binary, log


def test_omnigent_is_registered_as_one_atomic_backend() -> None:
    assert DRIVEN["omnigent"] == (OmnigentAgent, OmnigentAgentConfig)
    assert OmnigentAgent.pursues
    assert OmnigentAgent.rungs == ("bypass",)


def test_one_execution_returns_text_usage_and_trace_id(
    omnigent: tuple[Path, Path], tmp_path: Path
) -> None:
    _, log = omnigent
    agent = OmnigentAgent(_CONFIG)
    session = agent.new(tmp_path)

    events = list(session.stream("optimize kernel 001"))

    assert [(event.kind, event.text) for event in events] == [
        ("reasoning", "starting ephemeral execution"),
        ("result", "finished optimize kernel 001"),
    ]
    result = events[-1]
    assert result.tokens == {_MODEL: 150}
    assert dict(result.spent) == {
        "input": 100.0,
        "output": 20.0,
        "cache_read": 30.0,
        "cache_write": 0.0,
    }
    assert session.id == "run_123"
    assert json.loads(log.read_text().strip()) == [
        "run",
        "/opt/agent",
        "--no-session",
        "--json",
        "--model",
        _MODEL,
        "-p",
        "optimize kernel 001",
    ]


def test_a_humanize_session_cannot_start_a_second_execution(
    omnigent: tuple[Path, Path], tmp_path: Path
) -> None:
    session = OmnigentAgent(_CONFIG).new(tmp_path)
    assert session("first") == "finished first"

    with pytest.raises(RuntimeError, match="cannot resume"):
        session("second")


def test_pursue_is_the_one_supervisory_execution(
    omnigent: tuple[Path, Path], tmp_path: Path
) -> None:
    session = OmnigentAgent(_CONFIG).new(tmp_path)

    assert session.pursue("finish the kernel") == "finished finish the kernel"


def test_malformed_receipt_is_a_failed_turn(
    omnigent: tuple[Path, Path], tmp_path: Path
) -> None:
    session = OmnigentAgent(_CONFIG).new(tmp_path)

    with pytest.raises(subprocess.CalledProcessError) as failed:
        session("malformed")

    assert isinstance(failed.value, Failed)
    assert "wrote no JSON result" in str(failed.value)
