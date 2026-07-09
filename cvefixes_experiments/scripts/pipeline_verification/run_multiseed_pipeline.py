#!/usr/bin/env python3
"""
Multi-seed pipeline for the Checkpoint-8 sampling-driven experiments.

════════════════════════════════════════════════════════════════════════
WHY
════════════════════════════════════════════════════════════════════════
Several Checkpoint-8 slides report metrics from experiments that randomly
sample / split the CVEfixes data. A single run therefore only shows *one draw*
of a random process, so numbers wobble slide-to-slide. This orchestrator makes
that variation explicit and reproducible:

  * It runs a fixed SET of experiments over a fixed LIST of seeds.
  * Within one seed, every experiment is driven by the SAME seed, so the
    sampling / split is consistent across experiments for that run.
  * Across seeds it aggregates each metric into ``mean ± std`` (sample std,
    ddof=1), quantifying the sampling variation instead of hiding it.

The experiments covered (the non-commented, sampling-driven slides):

  1. ``same_split``          — method-level vs file-level input on the same
                               paired CVEs, reshaped to the shared ``--sample-mode``
                               CWE mix (``exp_method_vs_file_level``).
  2. ``index_distribution``  — proportional vs balanced index on a shared
                               query set (``exp_cvefixes_index_distribution``).
  3. ``threshold_gating``    — similarity-threshold gating derived from the SAME
                               shared split and the SAME default index as (2).

The two file-level experiments (2 + 3) are served from ONE shared run per seed:
the leakage-safe split fixes the query set, and both the proportional/balanced
indices AND the threshold study are built from that single index pool. The
``--sample-mode`` parameter (default ``balanced``) selects which index the
threshold study reuses, so every file-level experiment shares an identical
query set and an identical default index instead of drifting apart.

Deterministic / anchored slides (baseline "Exact Sample", AutoPatch) and the
Joern-heavy pipeline-verification baseline are intentionally excluded.

════════════════════════════════════════════════════════════════════════
OUTPUTS  (under ``--output-dir``)
════════════════════════════════════════════════════════════════════════
  seed_<n>/                     per-seed experiment run dirs + ``metrics.json``
  multiseed_summary.json        per-seed values + mean/std for every metric
  latex_snippets.tex            ready-to-paste mean±std tabular bodies

════════════════════════════════════════════════════════════════════════
USAGE
════════════════════════════════════════════════════════════════════════
    uv run python -m cvefixes_experiments.scripts.pipeline_verification.run_multiseed_pipeline \
        --config config.yaml \
        --seeds 42 43 44 45 46 \
        --sample-mode balanced --sample-total 140

    # only re-aggregate (reuse already-computed per-seed metrics.json):
    uv run python -m ...run_multiseed_pipeline --aggregate-only

    # splice the mean±std tables back into the slide deck:
    uv run python -m ...run_multiseed_pipeline --update-tex meetings/checkpoint8/checkpoint8.tex
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import traceback
from pathlib import Path
from typing import Any, Callable

# ── Experiment entry points ──────────────────────────────────────────
from cvefixes_experiments.scripts.pipeline_verification.exp_method_vs_file_level import (
    run_comparison as run_method_vs_file,
)
from cvefixes_experiments.scripts.pipeline_verification.exp_cvefixes_index_distribution import (
    run_study as run_index_distribution,
)
from src.metrics.threshold_analysis import build_analysis


# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_OUTPUT_DIR = Path("cvefixes_experiments/output/multiseed_pipeline")
DEFAULT_GRAPHML_ROOT = "graphml_fixedcves_file"
DEFAULT_DB_PATH = "data/cvefixes/CVEfixes.db"
DEFAULT_KS = [1, 5, 10]

# Default index shaping shared by the file-level experiments. ``balanced`` (a
# ~uniform CWE mix) is the default single index reused by the threshold study;
# ``sample_total`` is the shared index size for BOTH index-distribution variants
# and that default index, so every file-level experiment uses the same index.
DEFAULT_SAMPLE_MODE = "balanced"
DEFAULT_SAMPLE_TOTAL = 140

# Metric name in a retrieval cell is ``cve_retrieval`` (older runs: ``self_retrieval``).
RETRIEVAL_METRICS = ["hit@1", "hit@5", "hit@10", "mrr"]
ALL_EXPERIMENTS = ["same_split", "index_distribution", "threshold_gating"]

# Embedder display names (match the slide labels).
EMB_LABEL = {
    "gin": "GIN",
    "wl": "WL",
    "combined": "Combined",
    "codebert_seq": "CodeBERT (seq)",
    "codebert_pattern": "CodeBERT (pat)",
}

# Per-experiment embedder order (mirrors the slides).
SAME_SPLIT_EMBEDDERS = ["gin", "combined", "codebert_seq", "codebert_pattern"]
INDEX_DIST_EMBEDDERS = ["codebert_seq", "codebert_pattern", "combined", "gin"]
THRESHOLD_EMBEDDERS = ["codebert_seq", "codebert_pattern", "combined", "gin"]


# ── Cell helpers ─────────────────────────────────────────────────────


def _cve_block(cell: dict) -> dict:
    """Return the same-CVE retrieval block, tolerating the legacy key."""
    return cell.get("cve_retrieval") or cell.get("self_retrieval") or {}


def _retrieval_row(cell: dict) -> dict[str, float]:
    """Extract hit@k / mrr / cwe_recall from one retrieval cell."""
    block = _cve_block(cell)
    row = {m: float(block.get(m, 0.0) or 0.0) for m in RETRIEVAL_METRICS}
    row["cwe_recall"] = float((cell.get("cwe_recall") or {}).get("macro_avg", 0.0) or 0.0)
    return row


# ── Per-experiment metric extraction ─────────────────────────────────


def _extract_same_split(comparison: dict) -> dict[str, Any]:
    """method_cells / file_cells → {embedder: {function|file: row}}."""
    out: dict[str, Any] = {}
    for level, key in (("function", "method_cells"), ("file", "file_cells")):
        for cell in comparison.get(key, []):
            emb = cell["embedder"]
            out.setdefault(emb, {})[level] = _retrieval_row(cell)
    di = comparison.get("dataset_info", {})
    out["_meta"] = {
        "n_paired": di.get("n_paired_instances"),
        "n_index": di.get("n_index"),
        "n_query": di.get("n_query"),
    }
    return out


def _extract_index_distribution(comparison: dict) -> dict[str, Any]:
    """proportional_cells / balanced_cells → {embedder: {proportional|balanced: row}}."""
    out: dict[str, Any] = {}
    for variant, key in (("proportional", "proportional_cells"), ("balanced", "balanced_cells")):
        for cell in comparison.get(key, []):
            emb = cell["embedder"]
            out.setdefault(emb, {})[variant] = _retrieval_row(cell)
    di = comparison.get("dataset_info", {})
    out["_meta"] = {
        "n_query": di.get("n_query_shared"),
        "n_index": di.get("n_index_each"),
    }
    return out


def _sweep_row_at(sweep: list[dict], threshold: float | None) -> dict:
    """Return the sweep row nearest ``threshold`` (or the t=0 row if None)."""
    if not sweep:
        return {}
    if threshold is None:
        target = 0.0
    else:
        target = float(threshold)
    return min(sweep, key=lambda r: abs(float(r.get("threshold", 0.0)) - target))


def _extract_threshold(analysis: dict) -> dict[str, Any]:
    """threshold_analysis cells → {embedder: {base, tstar, coverage, sel_acc, tp, fp, fn, tn}}."""
    out: dict[str, Any] = {}
    n_query = None
    for cell in analysis.get("cells", []):
        emb = cell["embedder"]
        sweep = (cell.get("sweep") or {}).get("cwe") or []
        base_row = _sweep_row_at(sweep, 0.0)
        tstar = (cell.get("markers") or {}).get("f1max_cwe")
        star_row = _sweep_row_at(sweep, tstar)
        n_query = cell.get("n_queries_cwe", n_query)
        out[emb] = {
            "base": float(base_row.get("selective_accuracy", 0.0) or 0.0),
            "tstar": float(tstar) if tstar is not None else None,
            "coverage": float(star_row.get("coverage", 0.0) or 0.0),
            "sel_acc": float(star_row.get("selective_accuracy", 0.0) or 0.0),
            "tp": float(star_row.get("tp", 0.0) or 0.0),
            "fp": float(star_row.get("fp", 0.0) or 0.0),
            "fn": float(star_row.get("fn", 0.0) or 0.0),
            "tn": float(star_row.get("tn", 0.0) or 0.0),
        }
    out["_meta"] = {"n_query": n_query}
    return out


# ── Per-seed run ─────────────────────────────────────────────────────


def _run_same_split(
    cfg_path: str,
    seed_dir: Path,
    seed: int,
    ks: list[int],
    *,
    sample_mode: str,
    sample_total: int,
) -> dict:
    comparison = run_method_vs_file(
        config_path=cfg_path,
        output_dir=seed_dir / "method_vs_file",
        embedders=SAME_SPLIT_EMBEDDERS,
        ks=ks,
        seed=seed,
        reuse_only=True,  # cache-only → Joern-free & deterministic paired set
        sample_mode=sample_mode,  # reshape paired set to the shared CWE mix
        sample_total=sample_total,
    )
    return _extract_same_split(comparison)


def _run_file_level_unit(
    cfg_path: str,
    seed_dir: Path,
    seed: int,
    ks: list[int],
    graphml_root: str,
    db_path: str,
    *,
    sample_mode: str,
    sample_total: int,
    want_index: bool,
    want_threshold: bool,
) -> dict[str, dict]:
    """Serve BOTH file-level experiments from a SINGLE shared run.

    The index-distribution study loads the file-level pairs, performs the
    leakage-safe split ONCE (fixed shared query set) and builds two equal-size
    (``sample_total``) indices — proportional and balanced — from the same
    index pool. Because its ``{mode}_cells`` already record every query's
    retrieved items (with similarity ``score`` + ``cwe_id``), the
    threshold-gating study is derived from the chosen default index
    (``sample_mode``) with NO extra retrieval. Consequently both file-level
    slides share the SAME query set and the SAME default index.
    """
    comparison = run_index_distribution(
        config_path=cfg_path,
        output_dir=str(seed_dir / "index_distribution"),
        graphml_root=graphml_root,
        db_path=db_path,
        embedders=INDEX_DIST_EMBEDDERS,
        ks=ks,
        index_total=sample_total,
        seed=seed,
    )

    out: dict[str, dict] = {}
    if want_index:
        out["index_distribution"] = _extract_index_distribution(comparison)
    if want_threshold:
        cells = comparison.get(f"{sample_mode}_cells") or []
        if not cells:
            raise ValueError(
                f"index-distribution comparison has no '{sample_mode}_cells' "
                f"to derive the threshold study from"
            )
        analysis = build_analysis({"cells": cells, "run_id": f"threshold_{sample_mode}"})
        thr = _extract_threshold(analysis)
        di = comparison.get("dataset_info", {})
        thr["_meta"] = {
            "n_query": di.get("n_query_shared"),
            "n_index": di.get("n_index_each"),
            "sample_mode": sample_mode,
        }
        out["threshold_gating"] = thr
    return out


def run_one_seed(
    seed: int,
    *,
    cfg_path: str,
    output_dir: Path,
    experiments: list[str],
    ks: list[int],
    graphml_root: str,
    db_path: str,
    sample_mode: str,
    sample_total: int,
    force: bool,
) -> dict:
    """Run the selected experiments for a single seed; cache the extracted metrics.

    ``index_distribution`` and ``threshold_gating`` are computed together from
    one shared file-level run so they use an identical query set and default
    index; ``same_split`` is an independent paired (method-vs-file) comparison.
    """
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = seed_dir / "metrics.json"

    metrics: dict[str, Any] = {"seed": seed}
    if metrics_path.exists() and not force:
        metrics = json.loads(metrics_path.read_text())

    def _todo(name: str) -> bool:
        if name not in experiments:
            return False
        cached = metrics.get(name)
        if force or not cached or "_error" in cached:
            return True
        print(f"  [seed {seed}] {name}: cached, skipping")
        return False

    changed = False

    # 1) method-vs-file (paired; own cached CPG set — independent data path).
    if _todo("same_split"):
        print(f"\n{'#' * 70}\n# seed={seed}  experiment=same_split "
              f"(paired method-vs-file; sample={sample_mode}@{sample_total})\n{'#' * 70}")
        try:
            metrics["same_split"] = _run_same_split(
                cfg_path, seed_dir, seed, ks,
                sample_mode=sample_mode, sample_total=sample_total,
            )
        except Exception as exc:  # pragma: no cover - surfaced to the user
            print(f"  [seed {seed}] same_split FAILED: {exc}")
            traceback.print_exc()
            metrics["same_split"] = {"_error": str(exc)}
        changed = True

    # 2) file-level unit: index_distribution + threshold_gating from ONE run,
    #    so both share the same query set and the same default index.
    want_index = _todo("index_distribution")
    want_threshold = _todo("threshold_gating")
    if want_index or want_threshold:
        print(
            f"\n{'#' * 70}\n# seed={seed}  experiment=file_level  "
            f"(index_distribution + threshold_gating; "
            f"default index={sample_mode}@{sample_total}, shared query set)\n{'#' * 70}"
        )
        try:
            metrics.update(
                _run_file_level_unit(
                    cfg_path, seed_dir, seed, ks, graphml_root, db_path,
                    sample_mode=sample_mode, sample_total=sample_total,
                    want_index=want_index, want_threshold=want_threshold,
                )
            )
        except Exception as exc:  # pragma: no cover - surfaced to the user
            print(f"  [seed {seed}] file-level unit FAILED: {exc}")
            traceback.print_exc()
            if want_index:
                metrics["index_distribution"] = {"_error": str(exc)}
            if want_threshold:
                metrics["threshold_gating"] = {"_error": str(exc)}
        changed = True

    if changed:
        metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\n[seed {seed}] metrics → {metrics_path}")
    return metrics


# ── Aggregation ──────────────────────────────────────────────────────


def _stat(values: list[float]) -> dict[str, Any]:
    """mean / sample-std (ddof=1) / n over the non-None values."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0, "values": []}
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(vals), "values": vals}


def _agg_nested(per_seed: list[dict], sub_keys: list[str], metrics: list[str]) -> dict:
    """Aggregate {emb: {sub: {metric: value}}} dicts across seeds.

    Returns {emb: {sub: {metric: stat}}}.
    """
    embs: list[str] = []
    for snap in per_seed:
        for emb in snap:
            if emb != "_meta" and emb not in embs:
                embs.append(emb)
    agg: dict[str, Any] = {}
    for emb in embs:
        agg[emb] = {}
        for sub in sub_keys:
            agg[emb][sub] = {}
            for metric in metrics:
                series = []
                for snap in per_seed:
                    row = (snap.get(emb) or {}).get(sub) or {}
                    series.append(row.get(metric))
                agg[emb][sub][metric] = _stat(series)
    return agg


def _agg_threshold(per_seed: list[dict]) -> dict:
    """Aggregate threshold {emb: {base, tstar, coverage, ...}} across seeds."""
    fields = ["base", "tstar", "coverage", "sel_acc", "tp", "fp", "fn", "tn"]
    embs: list[str] = []
    for snap in per_seed:
        for emb in snap:
            if emb != "_meta" and emb not in embs:
                embs.append(emb)
    agg: dict[str, Any] = {}
    for emb in embs:
        agg[emb] = {}
        for field in fields:
            series = [(snap.get(emb) or {}).get(field) for snap in per_seed]
            agg[emb][field] = _stat(series)
    return agg


def aggregate(all_metrics: list[dict], experiments: list[str]) -> dict:
    """Build the mean±std summary from every seed's extracted metrics."""
    summary: dict[str, Any] = {}
    metrics = RETRIEVAL_METRICS + ["cwe_recall"]

    if "same_split" in experiments:
        snaps = [m["same_split"] for m in all_metrics if _ok(m, "same_split")]
        summary["same_split"] = {
            "levels": ["function", "file"],
            "metrics": metrics,
            "agg": _agg_nested(snaps, ["function", "file"], metrics),
            "meta": [s.get("_meta") for s in snaps],
        }
    if "index_distribution" in experiments:
        snaps = [m["index_distribution"] for m in all_metrics if _ok(m, "index_distribution")]
        summary["index_distribution"] = {
            "variants": ["proportional", "balanced"],
            "metrics": metrics,
            "agg": _agg_nested(snaps, ["proportional", "balanced"], metrics),
            "meta": [s.get("_meta") for s in snaps],
        }
    if "threshold_gating" in experiments:
        snaps = [m["threshold_gating"] for m in all_metrics if _ok(m, "threshold_gating")]
        summary["threshold_gating"] = {
            "fields": ["base", "tstar", "coverage", "sel_acc", "tp", "fp", "fn", "tn"],
            "agg": _agg_threshold(snaps),
            "meta": [s.get("_meta") for s in snaps],
        }
    return summary


def _ok(metrics: dict, name: str) -> bool:
    block = metrics.get(name)
    return bool(block) and "_error" not in block


# ── Console rendering ────────────────────────────────────────────────


def _fmt(stat: dict, dec: int = 3) -> str:
    if not stat or stat.get("mean") is None:
        return "   --   "
    return f"{stat['mean']:.{dec}f}±{stat['std']:.{dec}f}"


def print_console(summary: dict, seeds: list[int]) -> None:
    print("\n" + "=" * 78)
    print(f"MULTI-SEED SUMMARY  (mean ± sample-std over {len(seeds)} seeds: {seeds})")
    print("=" * 78)

    if "same_split" in summary:
        print("\n── same_split  (function → file input) ──")
        _print_nested(summary["same_split"], SAME_SPLIT_EMBEDDERS)
    if "index_distribution" in summary:
        print("\n── index_distribution  (proportional vs balanced index) ──")
        _print_nested(summary["index_distribution"], INDEX_DIST_EMBEDDERS)
    if "threshold_gating" in summary:
        print("\n── threshold_gating  (F1-optimal CWE gate) ──")
        _print_threshold(summary["threshold_gating"], THRESHOLD_EMBEDDERS)


def _print_nested(block: dict, emb_order: list[str]) -> None:
    subs = block.get("levels") or block.get("variants")
    metrics = block["metrics"]
    agg = block["agg"]
    header = f"  {'embedder':<16} {'sub':<12} " + " ".join(f"{m:>15}" for m in metrics)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for emb in emb_order:
        if emb not in agg:
            continue
        for sub in subs:
            cells = " ".join(f"{_fmt(agg[emb][sub][m]):>15}" for m in metrics)
            print(f"  {EMB_LABEL.get(emb, emb):<16} {sub:<12} {cells}")


def _print_threshold(block: dict, emb_order: list[str]) -> None:
    agg = block["agg"]
    cols = ["base", "tstar", "coverage", "sel_acc", "tp", "fp", "fn", "tn"]
    header = f"  {'embedder':<16} " + " ".join(f"{c:>13}" for c in cols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for emb in emb_order:
        if emb not in agg:
            continue
        cells = []
        for c in cols:
            dec = 2 if c == "tstar" else (1 if c in ("tp", "fp", "fn", "tn") else 3)
            cells.append(f"{_fmt(agg[emb][c], dec):>13}")
        print(f"  {EMB_LABEL.get(emb, emb):<16} " + " ".join(cells))


# ── LaTeX rendering ──────────────────────────────────────────────────


def _tex_ms(stat: dict, dec: int = 3, sdec: int = 3) -> str:
    """A ``mean\\,±std`` table cell (text mode; ± is inline math)."""
    if not stat or stat.get("mean") is None:
        return "---"
    return f"{stat['mean']:.{dec}f}\\,{{\\tiny$\\pm${stat['std']:.{sdec}f}}}"


def _tex_same_split(block: dict) -> str:
    agg = block["agg"]
    metrics = ["hit@1", "hit@5", "hit@10", "mrr", "cwe_recall"]
    lines: list[str] = []
    embs = [e for e in SAME_SPLIT_EMBEDDERS if e in agg]
    for i, emb in enumerate(embs):
        for level, label in (("function", "function"), ("file", "\\textbf{file}")):
            cells = " & ".join(_tex_ms(agg[emb][level][m]) for m in metrics)
            row = f"      {EMB_LABEL.get(emb, emb)} & {label} & {cells} \\\\"
            if level == "file":
                row = "      \\rowcolor{hit!8}\n" + row
            lines.append(row)
        if i != len(embs) - 1:
            lines.append("      \\midrule")
    return "\n".join(lines)


def _tex_index_distribution(block: dict) -> str:
    agg = block["agg"]
    metrics = ["hit@1", "hit@5", "hit@10", "mrr", "cwe_recall"]
    lines: list[str] = []
    embs = [e for e in INDEX_DIST_EMBEDDERS if e in agg]
    for i, emb in enumerate(embs):
        for j, variant in enumerate(("proportional", "balanced")):
            cells = " & ".join(_tex_ms(agg[emb][variant][m]) for m in metrics)
            name = EMB_LABEL.get(emb, emb) if j == 0 else ""
            lines.append(f"      {name} & {variant} & {cells} \\\\")
        if i != len(embs) - 1:
            lines.append("      \\midrule")
    return "\n".join(lines)


def _tex_threshold(block: dict) -> str:
    agg = block["agg"]
    lines: list[str] = []
    embs = [e for e in THRESHOLD_EMBEDDERS if e in agg]
    for emb in embs:
        a = agg[emb]
        base = _tex_ms(a["base"])
        tstar = _tex_ms(a["tstar"], dec=2, sdec=2)
        cov = _tex_ms(a["coverage"])
        sel = _tex_ms(a["sel_acc"])
        ok = _tex_ms(a["tp"], dec=1, sdec=1)
        wr = _tex_ms(a["fp"], dec=1, sdec=1)
        fn = _tex_ms(a["fn"], dec=1, sdec=1)
        tn = _tex_ms(a["tn"], dec=1, sdec=1)
        lines.append(
            f"      {EMB_LABEL.get(emb, emb)} & {base} & {tstar} & {cov} & {sel} "
            f"& {ok}\\,/\\,{wr} & {fn}\\,/\\,{tn} \\\\"
        )
    return "\n".join(lines)


TEX_BUILDERS: dict[str, Callable[[dict], str]] = {
    "same_split": _tex_same_split,
    "index_distribution": _tex_index_distribution,
    "threshold_gating": _tex_threshold,
}

# Marker keys used in checkpoint8.tex for --update-tex.
TEX_MARKERS = {
    "same_split": "mvf_same_split",
    "index_distribution": "index_distribution",
    "threshold_gating": "threshold_gating",
}


def write_latex_snippets(summary: dict, path: Path, seeds: list[int]) -> None:
    parts = [
        f"% Auto-generated mean±std tabular bodies (seeds={seeds}).",
        "% Paste a body between the \\midrule after the header and \\bottomrule.",
        "",
    ]
    for name in ALL_EXPERIMENTS:
        if name not in summary:
            continue
        body = TEX_BUILDERS[name](summary[name])
        marker = TEX_MARKERS[name]
        parts += [
            f"% >>> AUTOGEN:{marker}",
            body,
            f"% <<< AUTOGEN:{marker}",
            "",
        ]
    path.write_text("\n".join(parts))
    print(f"LaTeX snippets → {path}")


def update_tex(tex_path: Path, summary: dict) -> None:
    """Replace content between ``% >>> AUTOGEN:<key>`` / ``% <<< AUTOGEN:<key>``."""
    text = tex_path.read_text()
    updated = 0
    missing = []
    for name in ALL_EXPERIMENTS:
        if name not in summary:
            continue
        marker = TEX_MARKERS[name]
        body = TEX_BUILDERS[name](summary[name])
        pattern = re.compile(
            r"(% >>> AUTOGEN:" + re.escape(marker) + r"\n).*?(\n\s*% <<< AUTOGEN:" + re.escape(marker) + r")",
            re.DOTALL,
        )
        new_text, n = pattern.subn(lambda m: m.group(1) + body + m.group(2), text)
        if n:
            text = new_text
            updated += 1
        else:
            missing.append(marker)
    if missing:
        print(f"  WARNING: no AUTOGEN markers found for: {missing}. "
              f"Add '% >>> AUTOGEN:<key>' / '% <<< AUTOGEN:<key>' around the table body.")
    tex_path.write_text(text)
    print(f"Updated {updated} table(s) in {tex_path}")


# ── Orchestration ────────────────────────────────────────────────────


def run_pipeline(
    *,
    config_path: str,
    seeds: list[int],
    output_dir: Path,
    experiments: list[str],
    ks: list[int],
    graphml_root: str,
    db_path: str,
    sample_mode: str,
    sample_total: int,
    force: bool,
    aggregate_only: bool,
    update_tex_path: Path | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    for seed in seeds:
        if aggregate_only:
            mp = output_dir / f"seed_{seed}" / "metrics.json"
            if not mp.exists():
                print(f"  [seed {seed}] no metrics.json (aggregate-only), skipping")
                continue
            all_metrics.append(json.loads(mp.read_text()))
            continue
        all_metrics.append(
            run_one_seed(
                seed,
                cfg_path=config_path,
                output_dir=output_dir,
                experiments=experiments,
                ks=ks,
                graphml_root=graphml_root,
                db_path=db_path,
                sample_mode=sample_mode,
                sample_total=sample_total,
                force=force,
            )
        )

    summary = aggregate(all_metrics, experiments)
    payload = {
        "seeds": seeds,
        "experiments": experiments,
        "config": {
            "config_path": config_path,
            "graphml_root": graphml_root,
            "ks": ks,
            "sample_mode": sample_mode,
            "sample_total": sample_total,
        },
        "std": "sample (ddof=1)",
        "per_seed": all_metrics,
        "summary": summary,
    }
    summary_path = output_dir / "multiseed_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSummary → {summary_path}")

    write_latex_snippets(summary, output_dir / "latex_snippets.tex", seeds)
    print_console(summary, seeds)

    if update_tex_path is not None:
        update_tex(update_tex_path, summary)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--experiments", nargs="+", default=ALL_EXPERIMENTS, choices=ALL_EXPERIMENTS,
        help="Subset of experiments to run/aggregate.",
    )
    parser.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS)
    parser.add_argument("--graphml-root", default=DEFAULT_GRAPHML_ROOT)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--sample-mode", choices=["proportional", "balanced"], default=DEFAULT_SAMPLE_MODE,
        help=(
            "Default index CWE distribution reused by the non-distribution "
            "file-level experiments (threshold gating). The index-distribution "
            "study always builds BOTH proportional and balanced; this only "
            "selects which one the other experiments share (default: balanced)."
        ),
    )
    parser.add_argument(
        "--sample-total", type=int, default=DEFAULT_SAMPLE_TOTAL,
        help=(
            "Shared index size for the file-level experiments (both "
            "index-distribution variants and the default index)."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if metrics.json exists.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip running; just aggregate cached per-seed metrics.json.")
    parser.add_argument("--update-tex", default=None,
                        help="Path to a .tex file to splice mean±std tables into (AUTOGEN markers).")
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        seeds=args.seeds,
        output_dir=Path(args.output_dir),
        experiments=args.experiments,
        ks=args.ks,
        graphml_root=args.graphml_root,
        db_path=args.db_path,
        sample_mode=args.sample_mode,
        sample_total=args.sample_total,
        force=args.force,
        aggregate_only=args.aggregate_only,
        update_tex_path=Path(args.update_tex) if args.update_tex else None,
    )


if __name__ == "__main__":
    main()
