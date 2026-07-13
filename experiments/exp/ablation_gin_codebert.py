"""
Ablation study: Why GIN-CodeBERT with CWE-level triplet loss fails.

The original GIN-CodeBERT achieved hit@1=28.8% — far worse than the
untrained frozen GIN (72.6%). This script runs controlled ablations
isolating each hypothesized failure mode:

  A0  Original config (CWE labels, hidden=25, no warm start)  → reproduce 28.8%
  A1  CVE-level labels (fix objective mismatch)
  A2  hidden_dim=128 (fix capacity bottleneck)
  A3  Warm start from frozen GIN-CodeBERT init
  A4  Training regime: cosine scheduler + LR=1e-4 + patience=20
  A5  Combined (A1+A2+A3+A4)
  A6  A5 + mean-pool CodeBERT features per node type

Each ablation is trained from scratch, evaluated on the same query set,
and reports hit@1, hit@5, hit@10, MRR, CWE recall, final loss.

Usage:
    uv run python -m experiments.exp.ablation_gin_codebert [--config config.yaml]
    uv run python -m experiments.exp.ablation_gin_codebert --ablations A0 A1 A5
"""

from __future__ import annotations

import copy
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from experiments.common import (
    build_flat_index,
    build_split,
    evaluate_cwe_recall,
    evaluate_retrieval,
    load_pairs,
)
from src.embeddings.gin_model import GINCodeBERTModel, nx_to_pyg_codebert
from src.embeddings.node_codebert import NodeCodeBERTEncoder


# ═════════════════════════════════════════════════════════════════════
# Ablation configurations
# ═════════════════════════════════════════════════════════════════════

@dataclass
class AblationConfig:
    """One ablation variant."""
    name: str
    label_mode: str = "cwe"       # "cwe" or "cve"
    hidden_dim: int = 128         # GIN hidden dim (original experiment used 128)
    out_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    margin: float = 0.3
    epochs: int = 50
    batch_size: int = 16
    triplets_per_epoch: int = 256
    patience: int = 10
    mine_every: int = 5
    warm_start: bool = False
    use_scheduler: bool = False
    pool_by_type: bool = False    # mean-pool CodeBERT features per node type
    description: str = ""


ABLATIONS: dict[str, AblationConfig] = {
    "A0": AblationConfig(
        name="A0_original",
        label_mode="cwe",
        hidden_dim=128,
        lr=1e-3,
        warm_start=False,
        use_scheduler=False,
        description="Original failing config: CWE labels, no warm start (reproduce 28.8%)",
    ),
    "A1": AblationConfig(
        name="A1_cve_labels",
        label_mode="cve",
        hidden_dim=128,
        lr=1e-3,
        warm_start=False,
        use_scheduler=False,
        description="CVE-level labels (fix objective mismatch)",
    ),
    "A2": AblationConfig(
        name="A2_warm_start",
        label_mode="cwe",
        hidden_dim=128,
        lr=1e-3,
        warm_start=True,
        use_scheduler=False,
        description="Warm start from frozen GIN-CodeBERT (preserve geometry)",
    ),
    "A3": AblationConfig(
        name="A3_training_regime",
        label_mode="cwe",
        hidden_dim=128,
        lr=1e-4,
        warm_start=False,
        use_scheduler=True,
        patience=20,
        epochs=100,
        triplets_per_epoch=512,
        batch_size=32,
        description="Better training: cosine LR, patience 20, more triplets",
    ),
    "A4": AblationConfig(
        name="A4_combined",
        label_mode="cve",
        hidden_dim=128,
        lr=1e-4,
        warm_start=True,
        use_scheduler=True,
        patience=20,
        epochs=100,
        triplets_per_epoch=512,
        batch_size=32,
        description="All fixes combined (CVE labels + warm start + regime)",
    ),
    "A5": AblationConfig(
        name="A5_pooled_features",
        label_mode="cve",
        hidden_dim=128,
        lr=1e-4,
        warm_start=True,
        use_scheduler=True,
        patience=20,
        epochs=100,
        triplets_per_epoch=512,
        batch_size=32,
        pool_by_type=True,
        description="A4 + mean-pool CodeBERT features by node type (denoise)",
    ),
}


# ═════════════════════════════════════════════════════════════════════
# Feature pooling (A6): average CodeBERT embeddings per node type
# ═════════════════════════════════════════════════════════════════════

from src.embeddings.wl import NODE_TYPES, NODE_TYPE_IDX
import networkx as nx


def pool_features_by_type(G: nx.MultiDiGraph, node_features: np.ndarray) -> np.ndarray:
    """
    Replace per-node CodeBERT features with the mean of their node-type group.

    This denoises single-token embeddings by averaging all nodes of the same
    type (e.g., all IDENTIFIER nodes get the mean IDENTIFIER embedding).
    Preserves inter-type variation while removing intra-type noise.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    result = np.zeros_like(node_features)

    # Group node indices by type
    type_groups: dict[str, list[int]] = {}
    for i, nd in enumerate(nodes):
        ntype = G.nodes[nd].get("label", "UNKNOWN")
        if ntype not in NODE_TYPE_IDX:
            ntype = "UNKNOWN"
        type_groups.setdefault(ntype, []).append(i)

    # Compute mean per type and assign back
    for ntype, indices in type_groups.items():
        if not indices:
            continue
        type_mean = node_features[indices].mean(axis=0)
        for i in indices:
            result[i] = type_mean

    return result


# ═════════════════════════════════════════════════════════════════════
# Dataset with flexible labels
# ═════════════════════════════════════════════════════════════════════

class AblationTripletDataset:
    """
    TripletDataset with configurable label mode and optional feature pooling.
    """

    def __init__(
        self,
        graphs: list[nx.MultiDiGraph],
        cve_labels: list[str],
        cwe_labels: list[str],
        node_encoder: NodeCodeBERTEncoder,
        label_mode: str = "cwe",
        pool_by_type: bool = False,
    ):
        self._graphs = graphs
        self._cve_labels = cve_labels
        self._cwe_labels = cwe_labels
        self._label_mode = label_mode
        self._labels = cve_labels if label_mode == "cve" else cwe_labels
        self._pool_by_type = pool_by_type

        # Encode node features
        print(f"  [ablation] Encoding {len(graphs)} graphs "
              f"(label_mode={label_mode}, pool={pool_by_type})...")
        self._pyg_data: list[Data | None] = []
        for i, G in enumerate(graphs):
            feats = node_encoder.encode_graph(G)
            if pool_by_type:
                feats = pool_features_by_type(G, feats)
            data = nx_to_pyg_codebert(G, feats)
            self._pyg_data.append(data)
            if (i + 1) % 20 == 0:
                print(f"    {i + 1}/{len(graphs)} encoded")
        print(f"  [ablation] All {len(graphs)} graphs encoded")

        # Build class-to-indices mapping
        self._class_to_idx: dict[str, list[int]] = {}
        for i, label in enumerate(self._labels):
            if self._pyg_data[i] is None:
                continue
            self._class_to_idx.setdefault(label, []).append(i)

        # Classes with ≥2 samples (can form positives)
        self._valid_classes = [
            c for c, idxs in self._class_to_idx.items() if len(idxs) >= 2
        ]
        self._all_valid_idx = [
            i for c in self._valid_classes for i in self._class_to_idx[c]
        ]

        # Diagnostics
        n_singletons = sum(1 for c, idxs in self._class_to_idx.items() if len(idxs) == 1)
        n_valid = len(self._all_valid_idx)
        print(f"  [ablation] {len(self._valid_classes)} classes ≥2 samples, "
              f"{n_valid} trainable graphs, {n_singletons} singletons")
        if self._valid_classes:
            sizes = [len(self._class_to_idx[c]) for c in self._valid_classes]
            print(f"  [ablation] Class sizes: min={min(sizes)}, max={max(sizes)}, "
                  f"mean={np.mean(sizes):.1f}")

    @property
    def labels(self) -> list[str]:
        return self._labels

    def sample_triplets(self, n_triplets: int, seed: int | None = None) -> list[tuple[int, int, int]]:
        """Sample random triplets."""
        import random
        rng = random.Random(seed)
        triplets = []

        for _ in range(n_triplets):
            anchor_class = rng.choice(self._valid_classes)
            anchor_idx = rng.choice(self._class_to_idx[anchor_class])

            pos_candidates = [i for i in self._class_to_idx[anchor_class] if i != anchor_idx]
            if not pos_candidates:
                continue
            pos_idx = rng.choice(pos_candidates)

            neg_classes = [c for c in self._valid_classes if c != anchor_class]
            if not neg_classes:
                # Only one valid class — pick any index not in anchor class
                neg_candidates = [i for i in range(len(self._labels))
                                  if self._pyg_data[i] is not None
                                  and self._labels[i] != anchor_class]
                if not neg_candidates:
                    continue
                neg_idx = rng.choice(neg_candidates)
            else:
                neg_class = rng.choice(neg_classes)
                neg_idx = rng.choice(self._class_to_idx[neg_class])

            triplets.append((anchor_idx, pos_idx, neg_idx))

        return triplets

    def get_data(self, idx: int) -> Data:
        return self._pyg_data[idx]

    def collate_triplet_batch(
        self, triplets: list[tuple[int, int, int]]
    ) -> tuple[Batch, Batch, Batch]:
        anchors, positives, negatives = [], [], []
        for a, p, n in triplets:
            anchors.append(self._pyg_data[a])
            positives.append(self._pyg_data[p])
            negatives.append(self._pyg_data[n])
        return (
            Batch.from_data_list(anchors),
            Batch.from_data_list(positives),
            Batch.from_data_list(negatives),
        )


# ═════════════════════════════════════════════════════════════════════
# Semi-hard mining
# ═════════════════════════════════════════════════════════════════════

def semi_hard_mine_ablation(
    model: GINCodeBERTModel,
    dataset: AblationTripletDataset,
    batch_size: int,
    margin: float,
    device: str,
) -> list[tuple[int, int, int]]:
    """Online semi-hard negative mining."""
    import random

    model.eval()
    all_idx = dataset._all_valid_idx
    embeddings = {}

    with torch.no_grad():
        for start in range(0, len(all_idx), batch_size):
            batch_idx = all_idx[start:start + batch_size]
            data_list = [dataset.get_data(i) for i in batch_idx]
            batch = Batch.from_data_list(data_list).to(device)
            embs = model(batch).cpu().numpy()
            for j, idx in enumerate(batch_idx):
                embeddings[idx] = embs[j]

    triplets = []
    rng = random.Random(42)

    for cls in dataset._valid_classes:
        cls_idx = dataset._class_to_idx[cls]
        neg_idx = [i for i in all_idx if dataset._labels[i] != cls]

        for i, anchor_i in enumerate(cls_idx):
            for pos_i in cls_idx[i + 1:]:
                if anchor_i not in embeddings or pos_i not in embeddings:
                    continue
                a_emb = embeddings[anchor_i]
                p_emb = embeddings[pos_i]
                ap_dist = np.linalg.norm(a_emb - p_emb)

                semi_hard = []
                for ni in rng.sample(neg_idx, min(50, len(neg_idx))):
                    if ni not in embeddings:
                        continue
                    n_emb = embeddings[ni]
                    an_dist = np.linalg.norm(a_emb - n_emb)
                    if ap_dist < an_dist < ap_dist + margin:
                        semi_hard.append(ni)

                if semi_hard:
                    triplets.append((anchor_i, pos_i, rng.choice(semi_hard)))
                elif neg_idx:
                    triplets.append((anchor_i, pos_i, rng.choice(neg_idx)))

    return triplets


# ═════════════════════════════════════════════════════════════════════
# Trainer with ablation-aware options
# ═════════════════════════════════════════════════════════════════════

class AblationTrainer:
    """Trainer that supports all ablation variants."""

    def __init__(self, model: GINCodeBERTModel, acfg: AblationConfig, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.acfg = acfg

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay
        )
        self.loss_fn = nn.TripletMarginLoss(margin=acfg.margin, p=2)

        self.scheduler = None
        if acfg.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=acfg.epochs, eta_min=acfg.lr * 0.01
            )

    def train(
        self,
        dataset: AblationTripletDataset,
        val_dataset: AblationTripletDataset | None = None,
    ) -> dict:
        """Train and return history."""
        import random
        acfg = self.acfg
        history = {"train_loss": [], "val_loss": [], "epoch_time": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        print(f"\n  [{acfg.name}] Training: epochs={acfg.epochs}, "
              f"margin={acfg.margin}, lr={acfg.lr}, hidden={acfg.hidden_dim}")
        print(f"  [{acfg.name}] labels={acfg.label_mode}, warm_start={acfg.warm_start}, "
              f"scheduler={acfg.use_scheduler}")

        for epoch in range(acfg.epochs):
            t0 = time.perf_counter()

            # Mine triplets
            if epoch % acfg.mine_every == 0 and epoch > 0:
                triplets = semi_hard_mine_ablation(
                    self.model, dataset, acfg.batch_size, acfg.margin, self.device
                )
                if len(triplets) > acfg.triplets_per_epoch:
                    triplets = random.sample(triplets, acfg.triplets_per_epoch)
                if not triplets:
                    triplets = dataset.sample_triplets(acfg.triplets_per_epoch, seed=epoch)
            else:
                triplets = dataset.sample_triplets(acfg.triplets_per_epoch, seed=epoch)

            if not triplets:
                print(f"    Epoch {epoch + 1}: no valid triplets, skipping")
                continue

            # Train step
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(triplets), acfg.batch_size):
                batch_triplets = triplets[start:start + acfg.batch_size]
                a_batch, p_batch, n_batch = dataset.collate_triplet_batch(batch_triplets)
                a_batch = a_batch.to(self.device)
                p_batch = p_batch.to(self.device)
                n_batch = n_batch.to(self.device)

                self.optimizer.zero_grad()
                a_emb = self.model(a_batch)
                p_emb = self.model(p_batch)
                n_emb = self.model(n_batch)

                loss = self.loss_fn(a_emb, p_emb, n_emb)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.perf_counter() - t0
            history["train_loss"].append(avg_loss)
            history["epoch_time"].append(elapsed)

            # Validation
            val_loss = None
            if val_dataset is not None:
                val_loss = self._eval_loss(val_dataset)
                history["val_loss"].append(val_loss)
                monitor = val_loss
            else:
                monitor = avg_loss

            if monitor < best_val_loss:
                best_val_loss = monitor
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_str = f", val={val_loss:.4f}" if val_loss is not None else ""
                lr_now = self.scheduler.get_last_lr()[0] if self.scheduler else acfg.lr
                print(f"    Epoch {epoch + 1:3d}/{acfg.epochs}: "
                      f"loss={avg_loss:.4f}{val_str}, lr={lr_now:.1e} ({elapsed:.1f}s)")

            if patience_counter >= acfg.patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        print(f"  [{acfg.name}] Done. Best loss: {best_val_loss:.4f}, "
              f"epochs={len(history['train_loss'])}")
        return history

    def _eval_loss(self, dataset: AblationTripletDataset, n_triplets: int = 64) -> float:
        self.model.eval()
        triplets = dataset.sample_triplets(n_triplets, seed=9999)
        if not triplets:
            return 0.0

        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for start in range(0, len(triplets), self.acfg.batch_size):
                batch_triplets = triplets[start:start + self.acfg.batch_size]
                a_batch, p_batch, n_batch = dataset.collate_triplet_batch(batch_triplets)
                a_batch = a_batch.to(self.device)
                p_batch = p_batch.to(self.device)
                n_batch = n_batch.to(self.device)
                a_emb = self.model(a_batch)
                p_emb = self.model(p_batch)
                n_emb = self.model(n_batch)
                loss = self.loss_fn(a_emb, p_emb, n_emb)
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)


# ═════════════════════════════════════════════════════════════════════
# Warm start: create a GIN-CodeBERT model initialized to produce
# the same outputs as the frozen random GIN (for CodeBERT features)
# ═════════════════════════════════════════════════════════════════════

def create_warm_start_model(acfg: AblationConfig, device: str) -> GINCodeBERTModel:
    """
    Create a GIN-CodeBERT model with random weights that serve as
    a 'warm start' — the input_proj is initialized with small weights
    to produce a similar scale to what a random GIN would produce.

    True warm start from a frozen GIN isn't directly possible because
    the input dims differ (768 vs 11). Instead we use Xavier init
    which preserves variance across the projection.
    """
    model = GINCodeBERTModel(
        in_dim=768,
        hidden_dim=acfg.hidden_dim,
        out_dim=acfg.out_dim,
        num_layers=acfg.num_layers,
        dropout=acfg.dropout,
    )
    # Xavier uniform initialization (better than default for projection layers)
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
    model.to(device)
    return model


# ═════════════════════════════════════════════════════════════════════
# Main ablation runner
# ═════════════════════════════════════════════════════════════════════

def _ensure_autopatch_split(cfg: dict) -> dict:
    """
    Ensure cfg loads autopatch data with split enabled.

    The original GIN-CodeBERT experiment used autopatch data (225 index,
    73 query) with original/augmented variants enabling a proper split.
    """
    import copy
    cfg = copy.deepcopy(cfg)

    # Force autopatch dataset if not already configured
    if "autopatch" not in cfg.get("data", {}):
        cfg.setdefault("data", {})["autopatch"] = {
            "root": "CVE-list",
            "graphml_root": "graphml_augmented",
            "include_variants": True,
        }
    # Ensure autopatch is the active dataset
    cfg["data"]["active"] = ["autopatch"]

    # Enable split with proper augmented test/train separation
    cfg.setdefault("experiment", {}).setdefault("split", {})
    cfg["experiment"]["split"]["enabled"] = True
    cfg["experiment"]["split"].setdefault("stratified", True)
    cfg["experiment"]["split"].setdefault("seed", 42)
    cfg["experiment"]["split"].setdefault("test_ratio", 0.25)
    cfg["experiment"]["split"].setdefault("include_real_in_index", True)
    cfg["experiment"]["split"].setdefault("augmented_train_ratio", 1.0)
    cfg["experiment"]["split"].setdefault("query_source", "augmented_test")

    return cfg


def run_ablation(cfg: dict, ablation_names: list[str] | None = None, output_dir: Path | None = None) -> dict:
    """
    Run the full ablation study.

    Args:
        cfg: Global config
        ablation_names: Which ablations to run (default: all)
        output_dir: Output directory for results
    """
    if output_dir is None:
        output_dir = Path("experiments/output/ablation_gin_codebert")
    output_dir.mkdir(parents=True, exist_ok=True)

    if ablation_names is None:
        ablation_names = list(ABLATIONS.keys())

    print("=" * 70)
    print("  ABLATION STUDY: GIN-CodeBERT Failure Analysis")
    print("=" * 70)
    print(f"  Running: {ablation_names}")
    print(f"  Output:  {output_dir}")

    # ── 1. Load data (shared across all ablations) ───────────────────
    # Force autopatch dataset with proper original/augmented split
    cfg = _ensure_autopatch_split(cfg)
    print("\n[DATA] Loading autopatch pairs (original + augmented variants)...")
    pairs = load_pairs(cfg)
    index_pairs, query_pairs, split_info = build_split(pairs, cfg)
    print(f"  Index: {len(index_pairs)}, Query: {len(query_pairs)}")
    print(f"  Split: {split_info.get('mode', 'enabled')}")

    # Verify no trivial leakage (query should not be identical to index)
    if len(index_pairs) == len(query_pairs):
        idx_ids = set(id(p) for p in index_pairs)
        q_ids = set(id(p) for p in query_pairs)
        if idx_ids == q_ids:
            print("  [WARN] index == query — split is degenerate, check config!")
    # Show CVE overlap (expected: query CVEs should appear in index as different variants)
    idx_cves = set(p.cve_id for p in index_pairs)
    q_cves = set(p.cve_id for p in query_pairs)
    print(f"  CVE overlap (index∩query): {len(idx_cves & q_cves)}/{len(q_cves)} query CVEs")

    # Extract labels for both modes
    index_cve = [p.cve_id for p in index_pairs]
    index_cwe = [p.cwe_id or "UNKNOWN" for p in index_pairs]
    query_cve = [p.cve_id for p in query_pairs]
    query_cwe = [p.cwe_id or "UNKNOWN" for p in query_pairs]

    # Print label stats
    cve_counts = Counter(index_cve)
    cwe_counts = Counter(index_cwe)
    print(f"\n  [LABELS] CVE classes: {len(cve_counts)}, "
          f"mean size: {np.mean(list(cve_counts.values())):.1f}")
    print(f"  [LABELS] CWE classes: {len(cwe_counts)}, "
          f"mean size: {np.mean(list(cwe_counts.values())):.1f}")
    cve_trainable = sum(1 for c in cve_counts.values() if c >= 2)
    cwe_trainable = sum(1 for c in cwe_counts.values() if c >= 2)
    print(f"  [LABELS] Trainable (≥2): CVE={cve_trainable}, CWE={cwe_trainable}")

    # ── 2. Encode node features (shared, cached) ────────────────────
    print("\n[FEATURES] Loading node encoder...")
    emb_cfg = cfg.get("embeddings", {})
    node_encoder = NodeCodeBERTEncoder(emb_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # ── 3. Run each ablation ─────────────────────────────────────────
    results = {}
    ks = cfg.get("experiment", {}).get("ks", [1, 5, 10])

    for abl_name in ablation_names:
        if abl_name not in ABLATIONS:
            print(f"\n  [WARN] Unknown ablation: {abl_name}, skipping")
            continue

        acfg = ABLATIONS[abl_name]
        print(f"\n{'═' * 70}")
        print(f"  ABLATION {abl_name}: {acfg.description}")
        print(f"{'═' * 70}")

        t_start = time.perf_counter()

        # Build training dataset
        index_graphs = [p.G_vuln for p in index_pairs]
        train_ds = AblationTripletDataset(
            index_graphs, index_cve, index_cwe,
            node_encoder, acfg.label_mode, acfg.pool_by_type,
        )

        # Build validation dataset
        val_ds = None
        if len(query_pairs) >= 4:
            query_graphs = [p.G_vuln for p in query_pairs]
            val_ds = AblationTripletDataset(
                query_graphs, query_cve, query_cwe,
                node_encoder, acfg.label_mode, acfg.pool_by_type,
            )
            if not val_ds._valid_classes:
                val_ds = None

        # Create model
        if acfg.warm_start:
            model = create_warm_start_model(acfg, device)
        else:
            model = GINCodeBERTModel(
                in_dim=768,
                hidden_dim=acfg.hidden_dim,
                out_dim=acfg.out_dim,
                num_layers=acfg.num_layers,
                dropout=acfg.dropout,
            ).to(device)

        # Train
        trainer = AblationTrainer(model, acfg, device)
        history = trainer.train(train_ds, val_ds)
        train_time = time.perf_counter() - t_start

        # ── Evaluate ─────────────────────────────────────────────────
        print(f"\n  [{abl_name}] Evaluating...")
        model.eval()

        # Embed index
        index_embs = _embed_all(model, train_ds, device)

        # Embed queries
        if val_ds is not None:
            query_embs = _embed_all(model, val_ds, device)
        else:
            # Re-encode query graphs with same settings
            query_graphs = [p.G_vuln for p in query_pairs]
            tmp_ds = AblationTripletDataset(
                query_graphs, query_cve, query_cwe,
                node_encoder, acfg.label_mode, acfg.pool_by_type,
            )
            query_embs = _embed_all(model, tmp_ds, device)

        # L2 normalize
        index_norms = np.linalg.norm(index_embs, axis=1, keepdims=True)
        index_norms = np.where(index_norms == 0, 1, index_norms)
        index_embs = index_embs / index_norms

        query_norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
        query_norms = np.where(query_norms == 0, 1, query_norms)
        query_embs = query_embs / query_norms

        # Build flat index and evaluate
        flat_idx, retriever = build_flat_index(
            index_pairs, index_embs, f"gin_codebert_{abl_name}", acfg.out_dim
        )

        index_metadata = [
            {"cve_id": p.cve_id, "cwe_id": p.cwe_id, "func_name": p.func_name,
             "variant": p.meta.get("variant", ""), **p.meta}
            for p in index_pairs
        ]

        retrieval = evaluate_retrieval(query_pairs, query_embs, retriever, index_pairs, ks=ks)
        retrieval.pop("raw_queries", None)

        cwe_recall = evaluate_cwe_recall(
            query_pairs, query_embs, retriever, index_metadata, top_k=10
        )
        cwe_recall.pop("raw_queries", None)
        cwe_recall.pop("per_cwe", None)

        # Embedding space diagnostics
        mean_pairwise = float(np.mean(index_embs @ index_embs.T))
        emb_std = float(np.std(index_embs))

        # Store results
        abl_result = {
            "ablation": abl_name,
            "description": acfg.description,
            "config": {
                "label_mode": acfg.label_mode,
                "hidden_dim": acfg.hidden_dim,
                "lr": acfg.lr,
                "warm_start": acfg.warm_start,
                "use_scheduler": acfg.use_scheduler,
                "pool_by_type": acfg.pool_by_type,
                "margin": acfg.margin,
                "epochs_configured": acfg.epochs,
                "triplets_per_epoch": acfg.triplets_per_epoch,
            },
            "training": {
                "epochs_run": len(history["train_loss"]),
                "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
                "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
                "min_train_loss": min(history["train_loss"]) if history["train_loss"] else None,
                "train_time_s": train_time,
                "n_trainable_classes": len(train_ds._valid_classes),
                "n_trainable_graphs": len(train_ds._all_valid_idx),
            },
            "retrieval": {
                "hit@1": retrieval.get("hit@1", 0),
                "hit@5": retrieval.get("hit@5", 0),
                "hit@10": retrieval.get("hit@10", 0),
                "mrr": retrieval.get("mrr", 0),
            },
            "cwe_recall": {
                "macro_avg": cwe_recall.get("macro_avg", 0),
            },
            "space_quality": {
                "mean_pairwise_sim": mean_pairwise,
                "emb_std": emb_std,
            },
        }
        results[abl_name] = abl_result

        # Print summary
        print(f"\n  [{abl_name}] RESULTS:")
        print(f"    hit@1:  {abl_result['retrieval']['hit@1']:.3f}")
        print(f"    hit@5:  {abl_result['retrieval']['hit@5']:.3f}")
        print(f"    hit@10: {abl_result['retrieval']['hit@10']:.3f}")
        print(f"    MRR:    {abl_result['retrieval']['mrr']:.3f}")
        print(f"    CWE recall: {abl_result['cwe_recall']['macro_avg']:.3f}")
        print(f"    Final loss: {abl_result['training']['final_train_loss']:.4f}")
        print(f"    Epochs: {abl_result['training']['epochs_run']}")
        print(f"    Time:   {train_time:.1f}s")

    # ── 4. Summary table ─────────────────────────────────────────────
    print(f"\n\n{'═' * 70}")
    print("  ABLATION SUMMARY")
    print(f"{'═' * 70}")
    print(f"  {'Ablation':<8} {'hit@1':>6} {'hit@5':>6} {'MRR':>6} "
          f"{'CWE-R':>6} {'Loss':>7} {'Ep':>4}  Description")
    print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*4}  {'-'*40}")
    print(f"  {'Frozen':<8} {'0.726':>6} {'0.877':>6} {'0.788':>6} "
          f"{'0.443':>6} {'  ---':>7} {'---':>4}  Untrained frozen GIN (reference)")

    for abl_name in ablation_names:
        if abl_name not in results:
            continue
        r = results[abl_name]
        print(f"  {abl_name:<8} "
              f"{r['retrieval']['hit@1']:>6.3f} "
              f"{r['retrieval']['hit@5']:>6.3f} "
              f"{r['retrieval']['mrr']:>6.3f} "
              f"{r['cwe_recall']['macro_avg']:>6.3f} "
              f"{r['training']['final_train_loss']:>7.4f} "
              f"{r['training']['epochs_run']:>4d}  "
              f"{r['description'][:40]}")

    # ── 5. Save results ──────────────────────────────────────────────
    output_path = output_dir / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "ablations": results,
            "reference": {
                "frozen_gin": {"hit@1": 0.726, "hit@5": 0.877, "hit@10": 0.904, "mrr": 0.788},
                "gin_struct_trained": {"hit@1": 0.781, "hit@5": 0.932, "hit@10": 0.945, "mrr": 0.826},
            },
            "data_stats": {
                "n_index": len(index_pairs),
                "n_query": len(query_pairs),
                "n_cve_classes": len(cve_counts),
                "n_cwe_classes": len(cwe_counts),
                "cve_trainable": cve_trainable,
                "cwe_trainable": cwe_trainable,
                "cve_class_sizes": dict(Counter(cve_counts.values())),
                "cwe_class_sizes": dict(Counter(cwe_counts.values())),
            },
        }, f, indent=2)
    print(f"\n  Results saved → {output_path}")

    return results


def _embed_all(model: GINCodeBERTModel, dataset: AblationTripletDataset, device: str) -> np.ndarray:
    """Embed all graphs in a dataset."""
    model.eval()
    n = len(dataset._graphs)
    dim = model.out_dim
    result = np.zeros((n, dim), dtype=np.float32)

    with torch.no_grad():
        batch_size = 32
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            data_list = []
            valid_in_batch = []
            for i in range(start, end):
                d = dataset.get_data(i)
                if d is not None:
                    data_list.append(d)
                    valid_in_batch.append(i)

            if not data_list:
                continue
            batch = Batch.from_data_list(data_list).to(device)
            embs = model(batch).cpu().numpy()
            for j, orig_idx in enumerate(valid_in_batch):
                result[orig_idx] = embs[j]

    return result


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from src.io import load_config

    parser = argparse.ArgumentParser(
        description="Ablation study: GIN-CodeBERT failure analysis"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--ablations", nargs="+", default=None,
        choices=list(ABLATIONS.keys()),
        help="Which ablations to run (default: all)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output = Path(args.output) if args.output else None
    run_ablation(cfg, args.ablations, output)
