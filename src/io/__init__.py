"""I/O utilities — result writing, reading, and background flushing."""

from src.io.results import (
    BackgroundWriter,
    append_jsonl,
    find_cve_dir,
    load_completed,
    load_db_cache,
    load_ground_truth,
    read_code_file,
    save_json,
)

__all__ = [
    "BackgroundWriter",
    "append_jsonl",
    "find_cve_dir",
    "load_completed",
    "load_db_cache",
    "load_ground_truth",
    "read_code_file",
    "save_json",
]
