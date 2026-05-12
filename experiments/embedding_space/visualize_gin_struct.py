"""Generate 3D interactive HTML plot of trained GIN-struct embedding space."""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import umap
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from torch_geometric.data import Batch

from src.data.autopatch import load_pairs
from src.data.split import build_split
from src.embeddings.gin_struct_model import GINStructModel
from src.embeddings.wl import nx_to_pyg
from src.io.read_write import load_config
from src.training.struct_trainer import StructTripletTrainer

from experiments.embedding_space.visualize_embeddings import (
    plot_2d_comparison,
    tsne_project,
    umap_project,
)


def embed_graphs(model, graphs):
    """Embed graphs using trained model (train mode for BN consistency)."""
    model.train()
    embeddings = []
    with torch.no_grad():
        for G in graphs:
            data = nx_to_pyg(G)
            if data is None or data.x.shape[0] < 2:
                embeddings.append(np.zeros(model.out_dim, dtype=np.float32))
                continue
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
            batch = Batch.from_data_list([data])
            emb = model(batch).cpu().numpy()[0]
            embeddings.append(emb)
    return np.stack(embeddings)


def main():
    out_dir = Path("experiments/output/checkpoint5")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    cfg = load_config("config.yaml")
    pairs = load_pairs(cfg)
    index_pairs, query_pairs, _ = build_split(pairs, cfg)
    all_pairs = index_pairs + query_pairs
    graphs = [p.G_vuln for p in all_pairs]
    cwe_labels = [p.cwe_id for p in all_pairs]
    cve_ids = [p.cve_id for p in all_pairs]
    split_markers = ["index"] * len(index_pairs) + ["query"] * len(query_pairs)

    print(f"  {len(all_pairs)} samples, {len(set(cwe_labels))} CWE classes")

    # Load trained model
    ckpt_path = Path("experiments/output/checkpoint5/gin_struct_training/gin_struct_checkpoint.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    model = StructTripletTrainer.load_checkpoint(ckpt_path)

    # Compute embeddings
    print("Computing embeddings...")
    embs = embed_graphs(model, graphs)
    print(f"  Shape: {embs.shape}")

    # UMAP 3D projection
    print("UMAP 3D projection...")
    reducer = umap.UMAP(n_components=3, random_state=42, metric="cosine")
    coords_3d = reducer.fit_transform(embs)

    # Build 3D interactive plot
    unique_cwes = sorted(set(cwe_labels))
    short_labels = [l.split("(")[0].strip()[:30] for l in cwe_labels]
    hover_text = [
        f"{cve_ids[i]}<br>{short_labels[i]}<br>({split_markers[i]})"
        for i in range(len(all_pairs))
    ]

    fig = go.Figure()

    for cwe in unique_cwes:
        mask = np.array([l == cwe for l in cwe_labels])
        short_cwe = cwe.split("(")[0].strip()[:25]
        fig.add_trace(go.Scatter3d(
            x=coords_3d[mask, 0],
            y=coords_3d[mask, 1],
            z=coords_3d[mask, 2],
            mode="markers",
            name=short_cwe,
            marker=dict(size=4, opacity=0.7),
            text=[hover_text[i] for i in range(len(cwe_labels)) if mask[i]],
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
        text=[hover_text[i] for i in range(len(cwe_labels)) if query_mask[i]],
        hoverinfo="text",
    ))

    fig.update_layout(
        title="GIN-Struct (trained) — 3D UMAP Embedding Space",
        scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
        width=1000,
        height=700,
        legend=dict(font=dict(size=9)),
    )

    out_path = out_dir / "gin_struct_trained_3d_interactive.html"
    fig.write_html(str(out_path))
    print(f"\nSaved: {out_path}")

    # ── 2D plots (UMAP + t-SNE) ───────────────────────────────────
    short_cwes = [c.split("(")[0].strip()[:25] for c in unique_cwes]
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_cwes))
    embs_normed = normalize(embs, norm="l2")
    n_samples = len(all_pairs)

    # UMAP 2D
    print("UMAP 2D projection...")
    umap_2d = umap_project(embs_normed, n_components=2, seed=42)

    # t-SNE 2D
    print("t-SNE 2D projection...")
    tsne_2d = tsne_project(embs_normed, perplexity=30, seed=42)

    plot_2d_comparison(
        tsne_2d, umap_2d,
        cwe_labels, split_markers, unique_cwes, short_cwes, cmap,
        left_title=f"GIN-Struct — t-SNE ({embs.shape[1]}d, L2-normed)",
        right_title=f"GIN-Struct — UMAP ({embs.shape[1]}d, L2-normed)",
        suptitle=f"GIN-Struct Trained Embedding Space (n={n_samples})",
        out_path=out_dir / "gin_struct_trained_tsne_vs_umap.png",
    )

    # ── Raw vs PCA comparison ─────────────────────────────────────
    pca_dim = 64
    print(f"PCA {embs.shape[1]}d → {pca_dim}d...")
    pca = PCA(n_components=pca_dim, random_state=42)
    pca_embs = normalize(pca.fit_transform(embs), norm="l2")

    print("t-SNE on raw space...")
    raw_tsne_2d = tsne_project(embs_normed, perplexity=30, seed=42)
    print(f"t-SNE on PCA+L2 space ({pca_dim}d)...")
    pca_tsne_2d = tsne_project(pca_embs, perplexity=30, seed=42)

    plot_2d_comparison(
        raw_tsne_2d, pca_tsne_2d,
        cwe_labels, split_markers, unique_cwes, short_cwes, cmap,
        left_title=f"GIN-Struct — Raw ({embs.shape[1]}d, L2-normed)",
        right_title=f"GIN-Struct — PCA→{pca_dim}d + L2",
        suptitle=f"GIN-Struct Raw vs PCA (n={n_samples})",
        out_path=out_dir / "gin_struct_raw_vs_pca.png",
    )


if __name__ == "__main__":
    main()
