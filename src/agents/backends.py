"""
Pluggable LLM completion backends — the seam that makes the agent stack
debuggable, testable, and fully reproducible offline.

The patcher (``src.agents.patcher``) no longer calls ``litellm`` directly.
Instead it asks a :class:`CompletionBackend` for a :class:`CompletionResult`.
Swapping the backend lets the exact same agent code run in four modes:

    LiveBackend      → real Azure/litellm call (needs AZURE_* keys)
    MockBackend      → scripted, deterministic output (unit tests, no network)
    ReplayBackend    → read a recorded cassette (offline regression tests)
    RecordBackend    → call an inner backend once, cache to a cassette

A process-wide default backend can be installed with :func:`use_backend`
so that *existing* code paths (e.g. ``batch_inference`` → ``patch_one``)
transparently record or replay without any signature changes::

    from src.agents.backends import Cassette, RecordBackend, LiveBackend, use_backend

    cass = Cassette("tests/fixtures/agent/cassettes/smoke.jsonl")
    with use_backend(RecordBackend(LiveBackend(), cass)):
        run_batch_inference(...)      # records every call
    cass.save()

Design goals: zero new runtime dependencies, litellm imported lazily (so the
harness is importable and usable with no network / no litellm installed), and
byte-for-byte behavioural parity with the previous inline litellm call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── normalized response ──────────────────────────────────────────────


@dataclass
class CompletionResult:
    """Provider-agnostic view of a single chat completion.

    This is what every backend returns; the patcher copies these fields
    verbatim into its :class:`~src.agents.patcher.InvocationRecord`.
    """

    content: str
    finish_reason: str = "stop"
    response_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached: bool = False  # True when served from a cassette
    tool_calls: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "content": self.content,
            "finish_reason": self.finish_reason,
            "response_id": self.response_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d

    @classmethod
    def from_dict(cls, d: dict, *, cached: bool = False) -> "CompletionResult":
        return cls(
            content=d.get("content", ""),
            finish_reason=d.get("finish_reason", "stop"),
            response_id=d.get("response_id", ""),
            prompt_tokens=int(d.get("prompt_tokens", 0) or 0),
            completion_tokens=int(d.get("completion_tokens", 0) or 0),
            total_tokens=int(d.get("total_tokens", 0) or 0),
            cached=cached,
            tool_calls=d.get("tool_calls", []),
        )


@runtime_checkable
class CompletionBackend(Protocol):
    """Anything that can turn a chat request into a :class:`CompletionResult`."""

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        api_version: str = "",
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> CompletionResult: ...


# ── deterministic request keying (for cassettes) ─────────────────────


def request_fingerprint(
    *, model: str, messages: list[dict], temperature: float, max_tokens: int
) -> str:
    """Stable sha256 over the semantically-relevant request fields.

    ``api_version`` and auth are intentionally excluded — they do not change
    the model's output and would make cassettes brittle across environments.
    """
    canonical = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": round(float(temperature), 1),
            "max_tokens": int(max_tokens),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── live backend (litellm / Azure) ───────────────────────────────────


class LiveBackend:
    """Calls the real provider via litellm. Imported lazily.

    Replicates the exact behaviour of the previous inline ``litellm.completion``
    call in ``patcher.invoke`` (same kwargs, same error propagation).
    """

    def __init__(
        self,
        *,
        api_key_env: str = "AZURE_API_KEY",
        api_base_env: str = "AZURE_API_BASEURL",
    ):
        self._api_key_env = api_key_env
        self._api_base_env = api_base_env

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        api_version: str = "",
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> CompletionResult:
        import litellm  # lazy: keeps the harness importable & offline-safe

        kwargs: dict = dict(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
            api_key=os.getenv(self._api_key_env),
            api_base=os.getenv(self._api_base_env),
            api_version=api_version,
        )
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        response = litellm.completion(**kwargs)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)

        # Extract tool calls if present
        tc_list: list[dict] = []
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tc_list.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        return CompletionResult(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "",
            response_id=getattr(response, "id", "") or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            tool_calls=tc_list,
        )


# ── mock backend (scripted, deterministic, offline) ──────────────────


class MockBackend:
    """Deterministic backend for unit tests and offline debugging.

    Provide a ``responder`` callable ``(messages) -> str`` for full control,
    or use :meth:`from_patch` to emit a well-formed CoT/patch block that the
    real parser accepts.
    """

    def __init__(
        self,
        responder: Callable[[list[dict]], str] | str,
        *,
        finish_reason: str = "stop",
    ):
        if isinstance(responder, str):
            fixed = responder
            responder = lambda _messages: fixed  # noqa: E731
        self._responder = responder
        self._finish_reason = finish_reason
        self.calls: list[list[dict]] = []  # captured requests, for assertions

    @classmethod
    def from_patch(cls, cot: str, patch: str, **kw) -> "MockBackend":
        """Build a mock that returns the exact marker format the parser expects."""
        text = (
            f"[CoT START]\n{cot}\n[CoT END]\n\n"
            f"[Patched Code START]\n{patch}\n[Patched Code END]"
        )
        return cls(text, **kw)

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        api_version: str = "",
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> CompletionResult:
        self.calls.append(messages)
        content = self._responder(messages)
        # Rough but deterministic token estimate (~4 chars/token).
        prompt_tokens = sum(len(m.get("content", "")) for m in messages) // 4
        completion_tokens = len(content) // 4
        return CompletionResult(
            content=content,
            finish_reason=self._finish_reason,
            response_id="mock-0000",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


# ── cassette (record / replay store) ─────────────────────────────────


class CassetteMiss(KeyError):
    """Raised when a replay cassette has no entry for a request."""


class Cassette:
    """An append-only JSONL store of ``request_fingerprint → response``.

    Each line: ``{"key", "request": {...preview...}, "response": {...}, "recorded_at"}``.
    Requests are keyed by :func:`request_fingerprint`; a truncated preview of the
    request is stored purely for human inspection / diffing.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: dict[str, dict] = {}
        if self.path.exists():
            self.load()

    def load(self) -> "Cassette":
        self._entries.clear()
        with open(self.path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._entries[rec["key"]] = rec
                except (json.JSONDecodeError, KeyError):
                    logger.warning(
                        "Cassette %s: skipping bad line %d", self.path, lineno
                    )
        return self

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            for rec in self._entries.values():
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return self.path

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def get(self, key: str) -> CompletionResult:
        rec = self._entries.get(key)
        if rec is None:
            raise CassetteMiss(key)
        return CompletionResult.from_dict(rec["response"], cached=True)

    def put(self, key: str, request: dict, result: CompletionResult) -> None:
        preview_messages = [
            {"role": m.get("role", ""), "content": (m.get("content", "") or "")[:280]}
            for m in request.get("messages", [])
        ]
        self._entries[key] = {
            "key": key,
            "request": {
                "model": request.get("model", ""),
                "temperature": request.get("temperature"),
                "max_tokens": request.get("max_tokens"),
                "messages": preview_messages,
            },
            "response": result.to_dict(),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


class ReplayBackend:
    """Serves responses purely from a cassette — never touches the network.

    A miss raises :class:`CassetteMiss` (``strict=True``, the default) so that
    offline tests fail loudly when a fixture is stale.
    """

    def __init__(self, cassette: Cassette, *, strict: bool = True):
        self.cassette = cassette
        self.strict = strict

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        api_version: str = "",
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> CompletionResult:
        key = request_fingerprint(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return self.cassette.get(key)
        except CassetteMiss:
            if self.strict:
                raise
            return CompletionResult(content="", finish_reason="cassette_miss")


class RecordBackend:
    """Replay-if-present, otherwise call ``inner`` and record the result.

    This is the "record new interactions, replay existing ones" mode. Wrap a
    :class:`LiveBackend` to build fixtures, or a :class:`MockBackend` to exercise
    the record→replay round trip with no network at all.
    """

    def __init__(
        self, inner: CompletionBackend, cassette: Cassette, *, overwrite: bool = False
    ):
        self.inner = inner
        self.cassette = cassette
        self.overwrite = overwrite

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        api_version: str = "",
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> CompletionResult:
        key = request_fingerprint(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not self.overwrite and key in self.cassette:
            return self.cassette.get(key)
        result = self.inner.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_version=api_version,
            tools=tools,
            tool_choice=tool_choice,
        )
        self.cassette.put(
            key,
            {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": messages,
            },
            result,
        )
        return result


# ── process-wide default backend (thread-safe, scoped override) ──────

_DEFAULT_LOCK = threading.RLock()
_DEFAULT_BACKEND: CompletionBackend | None = None


def get_default_backend() -> CompletionBackend:
    """Return the installed default backend, or a fresh :class:`LiveBackend`."""
    with _DEFAULT_LOCK:
        return _DEFAULT_BACKEND if _DEFAULT_BACKEND is not None else LiveBackend()


def set_default_backend(backend: CompletionBackend | None) -> None:
    """Install (or clear, with ``None``) the process-wide default backend."""
    global _DEFAULT_BACKEND
    with _DEFAULT_LOCK:
        _DEFAULT_BACKEND = backend


@contextmanager
def use_backend(backend: CompletionBackend) -> Iterator[CompletionBackend]:
    """Temporarily install ``backend`` as the default for the enclosed block.

    Patchers constructed inside the block (including deep inside
    ``batch_inference``) pick it up automatically.
    """
    global _DEFAULT_BACKEND
    with _DEFAULT_LOCK:
        previous = _DEFAULT_BACKEND
        _DEFAULT_BACKEND = backend
    try:
        yield backend
    finally:
        with _DEFAULT_LOCK:
            _DEFAULT_BACKEND = previous


__all__ = [
    "CompletionResult",
    "CompletionBackend",
    "LiveBackend",
    "MockBackend",
    "Cassette",
    "CassetteMiss",
    "ReplayBackend",
    "RecordBackend",
    "request_fingerprint",
    "get_default_backend",
    "set_default_backend",
    "use_backend",
]
