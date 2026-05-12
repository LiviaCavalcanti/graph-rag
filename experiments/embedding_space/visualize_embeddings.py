"""Visualize embedding spaces: raw (high-dim) vs PCA-reduced.

Generates:
  1. PNG: t-SNE 2D comparison (raw vs PCA)
  2. HTML: Interactive 3D scatter (UMAP or PCA-3d) via Plotly
  3. PNG: UMAP 2D comparison (raw vs PCA)
  4. PNG: PCA variance explained (scree curve)

Usage:
    python -m experiments.visualize_embeddings [--output-dir plots/]
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from src.data.autopatch import load_pairs
from src.data.split import build_split
from src.embeddings import REGISTRY
from src.io.read_write import load_config


def get_raw_embeddings(embedder, graphs: list) -> np.ndarray:
    """Get raw embeddings before PCA/L2 (for combined: concatenated sub-embedders)."""
    if hasattr(embedder, "_raw_one"):
        # CombinedEmbedder — get pre-PCA concatenation
        results = []
        for i, G in enumerate(graphs):
            try:
                results.append(embedder._raw_one(G))
            except Exception as e:
                print(f"    [WARN] graph {i} failed: {e}")
                # Use zeros matching expected dim (netlsd=128 + wl=128 + gin=128 = 384)
                if results:
                    results.append(np.zeros_like(results[0]))
                else:
                    results.append(np.zeros(384, dtype=np.float32))
        return np.stack(results).astype(np.float32)
    elif hasattr(embedder, "_build_raw"):
        # CodeBERTPatternEmbedder — get true pre-PCA concatenation (802d)
        raw, _ = embedder._build_raw(graphs)
        return raw.astype(np.float32)
    else:
        # Other embedders — disable L2 temporarily to get raw
        orig_l2 = embedder.l2_normalize
        embedder.l2_normalize = False
        raw = embedder.embed_many(graphs)
        embedder.l2_normalize = orig_l2
        return raw


def get_pca_l2_embeddings(embedder, graphs: list) -> np.ndarray:
    """Get final embeddings (with PCA + L2 as configured)."""
    return embedder.embed_many(graphs)


def tsne_project(X: np.ndarray, perplexity: float = 30, seed: int = 42) -> np.ndarray:
    """Project high-dim embeddings to 2D via t-SNE."""
    n_samples = X.shape[0]
    perp = min(perplexity, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca")
    return tsne.fit_transform(X)


def umap_project(X: np.ndarray, n_components: int = 2, seed: int = 42) -> np.ndarray:
    """Project embeddings via UMAP (preserves global structure better than t-SNE)."""
    reducer = umap.UMAP(n_components=n_components, random_state=seed, metric="cosine")
    return reducer.fit_transform(X)


def plot_pca_scree(raw_embs: np.ndarray, emb_name: str, out_path: Path):
    """Plot PCA cumulative explained variance (scree curve)."""
    pca_full = PCA(random_state=42)
    pca_full.fit(raw_embs)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_dims = len(cumvar)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: cumulative variance
    ax1.plot(range(1, n_dims + 1), cumvar, "b-", linewidth=1.5)
    ax1.axhline(0.95, color="red", linestyle="--", alpha=0.7, label="95%")
    ax1.axhline(0.99, color="orange", linestyle="--", alpha=0.7, label="99%")
    ax1.axvline(128, color="green", linestyle=":", alpha=0.7, label="dim=128 (used)")
    # Mark where 95% and 99% are reached
    n95 = np.searchsorted(cumvar, 0.95) + 1
    n99 = np.searchsorted(cumvar, 0.99) + 1
    ax1.scatter([n95], [0.95], color="red", zorder=5, s=40)
    ax1.scatter([n99], [0.99], color="orange", zorder=5, s=40)
    ax1.set_xlabel("Number of PCA components")
    ax1.set_ylabel("Cumulative explained variance")
    ax1.set_title(f"{emb_name}: PCA Scree Curve")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, n_dims)
    ax1.set_ylim(0, 1.02)
    ax1.annotate(f"95% at d={n95}", (n95, 0.95), xytext=(n95 + 10, 0.90),
                 fontsize=8, arrowprops=dict(arrowstyle="->", color="red"), color="red")
    ax1.annotate(f"99% at d={n99}", (n99, 0.99), xytext=(n99 + 10, 0.93),
                 fontsize=8, arrowprops=dict(arrowstyle="->", color="orange"), color="orange")

    # Right: individual variance per component (bar chart, first 50)
    show_n = min(50, n_dims)
    ax2.bar(range(1, show_n + 1), pca_full.explained_variance_ratio_[:show_n],
            color="steelblue", alpha=0.8)
    ax2.set_xlabel("PCA component")
    ax2.set_ylabel("Individual explained variance")
    ax2.set_title(f"{emb_name}: Per-Component Variance (top {show_n})")
    ax2.set_xlim(0, show_n + 1)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_3d_interactive(
    coords_3d: np.ndarray,
    labels: list[str],
    split_markers: list[str],
    cve_ids: list[str],
    title: str,
    out_path: Path,
    unique_labels: list[str],
):
    """Generate interactive 3D scatter plot as HTML using Plotly."""
    import plotly.express as px

    short_labels = [l.split("(")[0].strip()[:30] for l in labels]
    hover_text = [
        f"{cve_ids[i]}<br>{short_labels[i]}<br>({split_markers[i]})"
        for i in range(len(labels))
    ]

    fig = go.Figure()

    # Plot each CWE class as a separate trace for legend
    for cwe in unique_labels:
        mask = np.array([l == cwe for l in labels])
        short_cwe = cwe.split("(")[0].strip()[:25]
        fig.add_trace(go.Scatter3d(
            x=coords_3d[mask, 0],
            y=coords_3d[mask, 1],
            z=coords_3d[mask, 2],
            mode="markers",
            name=short_cwe,
            marker=dict(size=4, opacity=0.7),
            text=[hover_text[i] for i in range(len(labels)) if mask[i]],
            hoverinfo="text",
        ))

    # Highlight query points
    query_mask = np.array([s == "query" for s in split_markers])
    fig.add_trace(go.Scatter3d(
        x=coords_3d[query_mask, 0],
        y=coords_3d[query_mask, 1],
        z=coords_3d[query_mask, 2],
        mode="markers",
        name="query (outline)",
        marker=dict(size=7, opacity=0.5, color="rgba(0,0,0,0)",
                    line=dict(width=2, color="black")),
        text=[hover_text[i] for i in range(len(labels)) if query_mask[i]],
        hoverinfo="text",
    ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3"),
        width=1000,
        height=700,
        legend=dict(font=dict(size=9)),
    )
    fig.write_html(str(out_path))



def plot_2d_comparison(
    left_2d: np.ndarray,
    right_2d: np.ndarray,
    cwe_labels: list[str],
    split_markers: list[str],
    unique_cwes: list[str],
    short_cwes: list[str],
    cmap,
    left_title: str,
    right_title: str,
    suptitle: str,
    out_path: Path,
):
    """Create a side-by-side 2D scatter comparison plot with query annotations."""
    query_mask = np.array([s == "query" for s in split_markers])
    query_cwes = [cwe_labels[i] for i in range(len(cwe_labels)) if split_markers[i] == "query"]
    short_query_cwes = [c.split("(")[0].strip()[:20] for c in query_cwes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    plot_space(ax1, left_2d, cwe_labels, left_title, unique_cwes, cmap)
    plot_space(ax2, right_2d, cwe_labels, right_title, unique_cwes, cmap)

    for ax, coords in [(ax1, left_2d), (ax2, right_2d)]:
        ax.scatter(
            coords[query_mask, 0],
            coords[query_mask, 1],
            facecolors="none",
            edgecolors="black",
            s=80,
            linewidths=1.0,
            label="query",
        )
        query_coords = coords[query_mask]
        for j, (x, y) in enumerate(query_coords):
            ax.annotate(
                short_query_cwes[j],
                (x, y),
                fontsize=4,
                alpha=0.8,
                ha="left",
                va="bottom",
                xytext=(3, 3),
                textcoords="offset points",
            )

    handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(i),
            markersize=8,
            label=short_cwes[i],
        )
        for i in range(len(unique_cwes))
    ]
    handles.append(
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=8,
            label="query sample",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(6, len(unique_cwes) + 1),
        fontsize=7,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_space(
    ax: plt.Axes,
    coords_2d: np.ndarray,
    labels: list[str],
    title: str,
    unique_labels: list[str],
    cmap,
):
    """Scatter plot colored by CWE class."""
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    colors = [label_to_idx[l] for l in labels]

    scatter = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=colors,
        cmap=cmap,
        alpha=0.7,
        s=30,
        edgecolors="white",
        linewidths=0.3,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    return scatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=str, default="plots/embedding_spaces", help="Output dir"
    )
    parser.add_argument(
        "--embedders",
        nargs="+",
        default=["combined", "gin", "wl", "codebert_seq", "codebert_pattern"],
        help="Embedders to visualize",
    )
    parser.add_argument("--perplexity", type=float, default=30, help="t-SNE perplexity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    cfg = load_config("config.yaml")
    pairs = load_pairs(cfg)
    index_pairs, query_pairs, split_info = build_split(pairs, cfg)
    all_pairs = index_pairs + query_pairs
    graphs = [p.G_vuln for p in all_pairs]
    cwe_labels = [p.cwe_id for p in all_pairs]
    split_markers = ["index"] * len(index_pairs) + ["query"] * len(query_pairs)

    unique_cwes = sorted(set(cwe_labels))
    # Shorten CWE names for display
    short_cwes = [c.split("(")[0].strip()[:25] for c in unique_cwes]
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_cwes))

    print(f"  {len(all_pairs)} samples, {len(unique_cwes)} CWE classes")

    emb_cfg = cfg["embeddings"]
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for emb_name in args.embedders:
        print(f"\nProcessing: {emb_name}")
        embedder = REGISTRY[emb_name](emb_cfg)

        # Check for cached embeddings
        raw_cache = cache_dir / f"{emb_name}_raw.npy"
        pca_cache = cache_dir / f"{emb_name}_pca.npy"

        if raw_cache.exists() and pca_cache.exists():
            print(f"  Loading cached embeddings from {cache_dir}")
            raw_embs = np.load(raw_cache)
            pca_embs = np.load(pca_cache)
        else:
            # Get raw embeddings
            print(f"  Computing raw embeddings...")
            raw_embs = get_raw_embeddings(embedder, graphs)
            print(f"  Raw dim: {raw_embs.shape[1]}")

            # Get PCA+L2 embeddings
            print(f"  Computing PCA+L2 embeddings...")
            pca_embs = get_pca_l2_embeddings(embedder, graphs)
            print(f"  PCA dim: {pca_embs.shape[1]}")

            # Save to cache
            np.save(raw_cache, raw_embs)
            np.save(pca_cache, pca_embs)
            print(f"  Cached to {cache_dir}")

        raw_dim = raw_embs.shape[1]
        pca_dim = pca_embs.shape[1]

        # t-SNE on raw space (apply L2 first for fair comparison)
        print(f"  t-SNE on raw space ({raw_dim}d)...")
        raw_normed = normalize(raw_embs, norm="l2")
        raw_2d = tsne_project(raw_normed, perplexity=args.perplexity, seed=args.seed)

        # t-SNE on PCA+L2 space
        print(f"  t-SNE on PCA+L2 space ({pca_dim}d)...")
        pca_2d = tsne_project(pca_embs, perplexity=args.perplexity, seed=args.seed)

        plot_2d_comparison(
            raw_2d, pca_2d,
            cwe_labels, split_markers, unique_cwes, short_cwes, cmap,
            left_title=f"{emb_name} — Raw ({raw_dim}d, L2-normed)",
            right_title=f"{emb_name} — PCA→{pca_dim}d + L2",
            suptitle=f"Embedding Space: {emb_name} (n={len(all_pairs)}, seed={args.seed})",
            out_path=out_dir / f"{emb_name}_raw_vs_pca.png",
        )

        # ── UMAP 2D comparison ────────────────────────────────────────
        print(f"  UMAP on raw space ({raw_dim}d)...")
        raw_umap_2d = umap_project(raw_normed, n_components=2, seed=args.seed)
        print(f"  UMAP on PCA+L2 space ({pca_dim}d)...")
        pca_umap_2d = umap_project(pca_embs, n_components=2, seed=args.seed)

        plot_2d_comparison(
            raw_umap_2d, pca_umap_2d,
            cwe_labels, split_markers, unique_cwes, short_cwes, cmap,
            left_title=f"{emb_name} — UMAP of Raw ({raw_dim}d)",
            right_title=f"{emb_name} — UMAP of PCA→{pca_dim}d + L2",
            suptitle=f"UMAP: {emb_name} (n={len(all_pairs)}, cosine metric)",
            out_path=out_dir / f"{emb_name}_umap.png",
        )

        # ── 3D interactive (UMAP 3D on PCA space) ─────────────────────
        print(f"  UMAP 3D on PCA+L2 space...")
        pca_umap_3d = umap_project(pca_embs, n_components=3, seed=args.seed)
        cve_ids = [p.cve_id for p in all_pairs]
        html_path = out_dir / f"{emb_name}_3d_interactive.html"
        plot_3d_interactive(
            pca_umap_3d, cwe_labels, split_markers, cve_ids,
            f"{emb_name} — 3D UMAP (PCA→{pca_dim}d + L2)",
            html_path, unique_cwes,
        )
        print(f"  Saved: {html_path}")

        # ── PCA variance explained (scree curve) ──────────────────────
        print(f"  PCA scree curve...")
        scree_path = out_dir / f"{emb_name}_pca_scree.png"
        plot_pca_scree(raw_embs, emb_name, scree_path)
        print(f"  Saved: {scree_path}")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
