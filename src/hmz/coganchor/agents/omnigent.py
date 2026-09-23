"""Omnigent, driven as one ephemeral headless execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

from .base import AgentBase, CommandSessionBase
from .config import AgentConfig
from .event import Event, Failed, Usage

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator

__all__ = ["OmnigentAgent", "OmnigentAgentConfig", "OmnigentSession"]

_COMMAND = "omnigent"
_KINDS = {
    "input": "input_tokens",
    "output": "output_tokens",
    "cache_read": "cache_read_input_tokens",
    "cache_write": "cache_creation_input_tokens",
}


@dataclass(frozen=True, kw_only=True)
class OmnigentAgentConfig(AgentConfig):
    """The model to run and the agent bundle Omnigent loads."""

    agent: str = "/opt/agent"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.agent.strip():
            raise ValueError("agent must name an Omnigent agent bundle")


def _receipt(transcript: str) -> dict[str, object]:
    """Read and validate the one JSON result an ephemeral run writes."""
    parsed = False
    for line in transcript.splitlines():
        try:
            raw: object = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        parsed = True
        if isinstance(raw, dict):
            receipt = cast("dict[str, object]", raw)
            if receipt.get("type") == "result":
                return receipt
    because = (
        "omnigent wrote no result receipt"
        if parsed
        else "omnigent wrote no JSON result"
    )
    raise Failed(1, [_COMMAND], transcript, because)


def _counted(raw: object, field: str) -> int | None:
    """Read one optional non-negative integer token count."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise Failed(1, [_COMMAND], "", f"omnigent wrote an invalid {field}")
    return raw


class OmnigentSession(CommandSessionBase):
    """One Omnigent execution, with no conversation to resume afterwards."""

    protocol: ClassVar[bool] = True

    def _turn(self, prompt: str) -> tuple[list[str], str | None]:
        """Build one isolated, machine-readable Omnigent invocation."""
        if self._id is not None:
            raise RuntimeError(
                "an Omnigent execution cannot resume; open a new Humanize session"
            )
        config = cast("OmnigentAgentConfig", self._agent.config)
        return (
            [
                _COMMAND,
                "run",
                config.agent,
                "--no-session",
                "--json",
                "--model",
                config.model,
                "-p",
                prompt,
            ],
            None,
        )

    def _reads(self, line: str, *, error: bool) -> Iterator[Event]:
        """Keep stdout for the receipt and expose stderr only as progress."""
        if error and line.strip():
            yield Event(kind="reasoning", text=line.rstrip("\n"))

    def _result(self, transcript: str) -> Event:
        """Turn Omnigent's receipt into Humanize's result and token accounting."""
        receipt = _receipt(transcript)
        text = receipt.get("text")
        usage = receipt.get("usage")
        if not isinstance(text, str) or not isinstance(usage, dict):
            raise Failed(
                1, [_COMMAND], transcript, "omnigent wrote an invalid result receipt"
            )
        counted_usage = cast("dict[str, object]", usage)

        spent: dict[str, float] = {}
        for kind, field in _KINDS.items():
            if (count := _counted(counted_usage.get(field), field)) is not None:
                spent[kind] = float(count)

        tokens: dict[str, int] = {}
        models = counted_usage.get("models")
        if models is not None and not isinstance(models, dict):
            raise Failed(
                1, [_COMMAND], transcript, "omnigent wrote invalid model usage"
            )
        for model, raw in cast("dict[str, object]", models or {}).items():
            if not isinstance(raw, dict):
                raise Failed(
                    1, [_COMMAND], transcript, "omnigent wrote invalid model usage"
                )
            model_usage = cast("dict[str, object]", raw)
            total = 0
            for field in _KINDS.values():
                count = _counted(model_usage.get(field), field)
                if count is not None:
                    total += count
            if total:
                tokens[str(model)] = total

        return Event(kind="result", text=text, tokens=tokens, spent=Usage(spent))

    def _read_session_id(self, transcript: str) -> str:
        """Read the ephemeral execution id for tracing, not for resumption."""
        named = _receipt(transcript).get("session_id")
        if not isinstance(named, str) or not named:
            raise Failed(1, [_COMMAND], transcript, "omnigent named no execution")
        return named

    def _pursue(self, objective: str) -> str:
        """Give the objective to Omnigent's own supervisory execution."""
        return self(objective)


class OmnigentAgent(AgentBase):
    """An atomic Omnigent agent execution."""

    pursues: ClassVar[bool] = True
    rungs: ClassVar[tuple[str, ...]] = ("bypass",)
    counts: ClassVar[frozenset[str]] = frozenset(_KINDS)

    def new(self, cwd: str | os.PathLike[str] | None = None) -> OmnigentSession:
        """Open one execution rooted in the given workspace."""
        return OmnigentSession(self, cwd)
