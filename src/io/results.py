"""JSONL result I/O — reading completed queries, appending results, background writer."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from queue import Queue


def load_completed(jsonl_path: Path) -> set[tuple[str, str]]:
    """Load already-completed (cve_id, variant) keys from a JSONL file."""
    done: set[tuple[str, str]] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec["query_cve"], rec.get("query_variant", ""))
                done.add(key)
            except (json.JSONDecodeError, KeyError):
                print(f"  WARNING: skipping malformed line {lineno} in {jsonl_path}")
    return done


def append_jsonl(path: Path, records: list[dict]) -> None:
    """Append records to a JSONL file (atomic per batch, fsync'd)."""
    with open(path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


class BackgroundWriter:
    """Writes batches to disk in a background thread."""

    def __init__(self, jsonl_path: Path):
        self._path = jsonl_path
        self._queue: Queue[list[dict] | None] = Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._error: Exception | None = None

    def _worker(self):
        while True:
            batch = self._queue.get()
            if batch is None:
                break
            try:
                append_jsonl(self._path, batch)
            except Exception as e:
                self._error = e
            self._queue.task_done()

    def write(self, batch: list[dict]):
        """Enqueue a batch for writing.  Raises if a previous write failed."""
        if self._error:
            raise self._error
        self._queue.put(batch)

    def flush(self):
        """Wait for all pending writes to complete."""
        self._queue.join()
        if self._error:
            raise self._error

    def close(self):
        self._queue.put(None)
        self._thread.join(timeout=5)
