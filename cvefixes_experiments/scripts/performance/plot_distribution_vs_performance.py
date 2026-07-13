"""
Does the per-CWE class distribution of the index explain the per-CWE retrieval
performance?

For a ``results.json`` from the index-distribution study, this script draws, for
each CWE class:

* a bar  = number of indexed examples of that class (the "class support"),
* one line per model = that model's performance on the class (default: CWE
  recall), with the points connected by linear interpolation.

Classes are sorted by support (descending), so if a model's performance line
trends with the bars, then class frequency correlates with performance. To make
that quantitative, the Spearman (rank) and Pearson correlations between support
and performance are computed per model; the Spearman coefficient is shown in the
legend. Each model gets its own colour.

Usage (always via uv, from the repo root):

    uv run python cvefixes_experiments/scripts/plot_distribution_vs_performance.py
    uv run python cvefixes_experiments/scripts/plot_distribution_vs_performance.py \
        --results cvefixes_experiments/output/index_distribution_study/balanced/results.json \
        --metric mrr
    uv run python cvefixes_experiments/scripts/plot_distribution_vs_performance.py \
        --results cvefixes_experiments/output/index_distribution_study/proportional/results.json \
                  cvefixes_experiments/output/index_distribution_study/balanced/results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = (
    REPO_ROOT
    / "cvefixes_experiments/output/index_distribution_study/proportional/results.json"
)

# metric key -> (display label, y-axis upper bound)
METRICS = {
    "recall": ("CWE recall", 1.05),
    "mrr": ("MRR", 1.05),
    "hit": ("Hit rate (hit@k)", 1.05),
}


# ── data loading ─────────────────────────────────────────────────────────────
def load_cells(results_path: Path):
    data = json.loads(Path(results_path).read_text())
    cells = data.get("cells", [])
    if not cells:
        raise ValueError(f"No 'cells' array found in {results_path}")
    return data, cells


def class_support(cells) -> dict[str, int]:
    """# indexed examples per CWE (identical across models -> take/validate)."""
    support: dict[str, int] = {}
    for cell in cells:
        per_cwe = cell.get("cwe_recall", {}).get("per_cwe", {})
        for cwe, stats in per_cwe.items():
            s = int(stats.get("support", 0))
            if cwe in support and support[cwe] != s:
                print(f"[warn] inconsistent support for {cwe}: {support[cwe]} vs {s}")
            support[cwe] = max(support.get(cwe, 0), s)
    return support


def per_model_performance(cells, metric: str) -> dict[str, dict[str, float]]:
    """model -> {cwe -> performance} for the requested metric."""
    perf: dict[str, dict[str, float]] = {}
    for cell in cells:
        model = cell["embedder"]
        if metric == "recall":
            per_cwe = cell.get("cwe_recall", {}).get("per_cwe", {})
            perf[model] = {
                c: float(s.get("recall", np.nan)) for c, s in per_cwe.items()
            }
        else:  # aggregate per-query retrieval metric by CWE
            field = "mrr" if metric == "mrr" else "hit"
            groups: dict[str, list[float]] = {}
            for q in cell.get("cve_retrieval", {}).get("raw_queries", []):
                cwe = q.get("query_cwe")
                if cwe is None:
                    continue
                groups.setdefault(cwe, []).append(float(q.get(field, 0.0)))
            perf[model] = {
                c: (float(np.mean(v)) if v else np.nan) for c, v in groups.items()
            }
    return perf


# ── correlations (numpy-only, no scipy dependency) ───────────────────────────
def _rankdata(a):
    """Average-tie ranks (like scipy.stats.rankdata)."""
    a = np.asarray(a, float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=int)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    return _pearson(_rankdata(x[m]), _rankdata(y[m]))


# ── plotting ─────────────────────────────────────────────────────────────────
def make_figure(results_path: Path, metric: str):
    _, cells = load_cells(results_path)
    support = class_support(cells)
    perf = per_model_performance(cells, metric)
    label, y_top = METRICS[metric]

    classes = sorted(support, key=lambda c: (-support[c], c))  # by count desc
    counts = np.array([support[c] for c in classes], float)
    x = np.arange(len(classes))
    models = sorted(perf)
    cmap = plt.get_cmap("tab10")

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # bars: class distribution
    bars = ax1.bar(
        x,
        counts,
        width=0.6,
        color="0.82",
        edgecolor="0.55",
        label="Indexed examples (class support)",
        zorder=1,
    )
    for xi, c in zip(x, counts):
        ax1.text(
            xi,
            c + counts.max() * 0.01,
            str(int(c)),
            ha="center",
            va="bottom",
            fontsize=9,
            color="0.35",
        )
    ax1.set_ylabel("# indexed examples per CWE (class support)")
    ax1.set_ylim(0, counts.max() * 1.18)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}\n(n={int(support[c])})" for c in classes])
    ax1.set_xlabel("CWE class (sorted by support, descending)")

    # lines: per-model performance on a second axis
    ax2 = ax1.twinx()
    line_handles = []
    summary = []
    for i, model in enumerate(models):
        y = np.array([perf[model].get(c, np.nan) for c in classes], float)
        rho = _spearman(counts, y)
        r = _pearson(counts, y)
        summary.append((model, rho, r))
        (h,) = ax2.plot(
            x,
            y,
            marker="o",
            markersize=6,
            linewidth=2.2,
            color=cmap(i % 10),
            label=f"{model}  (\u03c1={rho:+.2f})",
            zorder=3,
        )
        line_handles.append(h)
    ax2.set_ylabel(label)
    ax2.set_ylim(0, y_top)

    # combined legend (bars + model lines)
    handles = [bars, *line_handles]
    ax2.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper right",
        framealpha=0.92,
        fontsize=9,
    )

    dist = cells[0].get("index_distribution") or results_path.parent.name
    fig.suptitle(
        f"Class distribution vs. {label} per CWE   \u2014   index = {dist}",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_title(
        "\u03c1 = Spearman rank correlation between class support and "
        f"{label} (per model)",
        fontsize=10,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, summary


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--results",
        nargs="+",
        type=Path,
        default=[DEFAULT_RESULTS],
        help="Path(s) to results.json (default: the proportional study).",
    )
    ap.add_argument(
        "--metric",
        choices=list(METRICS),
        default="recall",
        help="Per-class performance metric (default: recall).",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: next to each results.json).",
    )
    args = ap.parse_args()

    for results_path in args.results:
        results_path = results_path.resolve()
        fig, summary = make_figure(results_path, args.metric)
        out_dir = args.outdir or results_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"distribution_vs_{args.metric}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"\n{results_path.parent.name}/{results_path.name}  [metric={args.metric}]")
        print(f"  saved: {out_path}")
        print(f"  {'model':<18}{'Spearman rho':>14}{'Pearson r':>12}")
        for model, rho, r in summary:
            print(f"  {model:<18}{rho:>14.3f}{r:>12.3f}")


if __name__ == "__main__":
    main()
