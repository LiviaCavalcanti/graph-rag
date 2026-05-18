from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from experiments.dashboard_scripts._theme import score_color as _score_color_raw
from src.evaluate.preprocessing import extract_function_body
from src.data.autopatch import AutoPatchDataset


def score_color(value) -> str:
    """Wrapper that handles None values gracefully."""
    if value is None:
        return "#888"
    return _score_color_raw(value)


# ── helpers ──────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


find_cve_dir = AutoPatchDataset.find_cve_dir


def load_input_code(cve_id: str, base_dir: Path) -> str | None:
    """Load the original vulnerable code for a CVE."""
    cve_dir = find_cve_dir(cve_id, base_dir)
    if cve_dir is None:
        return None
    original = cve_dir / "original_code.txt"
    if original.exists():
        return original.read_text(errors="replace").strip()
    return None


def load_ground_truth(cve_id: str, variant: str, base_dir: Path) -> str | None:
    """Load the ground-truth fixed code for a CVE+variant."""
    cve_dir = find_cve_dir(cve_id, base_dir)
    if cve_dir is None:
        return None
    gt_file = cve_dir / "out_v2" / "code" / f"{variant}_fixed.c"
    if gt_file.exists():
        raw = gt_file.read_text(errors="replace")
        return extract_function_body(raw).strip()
    return None


def fmt(val, ndigits: int = 4) -> str:
    if val is None:
        return "-"
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, float):
        return f"{val:.{ndigits}f}"
    return str(val)


def summarize_metric(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }
