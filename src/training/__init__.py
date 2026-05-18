"""
Triplet loss trainer for the GIN-CodeBERT model.

Trains a GINCodeBERTModel using triplet margin loss with online
semi-hard negative mining. Uses CWE-ID as the class label for
triplet construction.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.embeddings.gin_model import GINCodeBERTModel, nx_to_pyg_codebert
from src.embeddings.node_codebert import NodeCodeBERTEncoder


class TripletDataset:
    """
    Manages graph data and online triplet mining by CWE-ID.

    Groups graphs by CWE class. Each iteration yields (anchor, positive, negative)
    as PyG Data objects with precomputed CodeBERT node features.
    """

    def __init__(
        self,
        graphs: list[nx.MultiDiGraph],
        labels: list[str],
        node_encoder: NodeCodeBERTEncoder,
    ):
        self._graphs = graphs
        self._labels = labels
        self._node_encoder = node_encoder

        # Precompute node features and PyG data
        print("  [triplet] Encoding node features...")
        self._pyg_data: list[Data | None] = []
        for i, G in enumerate(graphs):
            feats = node_encoder.encode_graph(G)
            data = nx_to_pyg_codebert(G, feats)
            self._pyg_data.append(data)
            if (i + 1) % 20 == 0:
                print(f"    {i + 1}/{len(graphs)} graphs encoded")
        print(f"  [triplet] All {len(graphs)} graphs encoded")

        # Build class-to-indices mapping
        self._class_to_idx: dict[str, list[int]] = {}
        for i, label in enumerate(labels):
            if self._pyg_data[i] is None:
                continue
            self._class_to_idx.setdefault(label, []).append(i)

        # Filter classes with at least 2 samples (needed for positive pairs)
        self._valid_classes = [
            c for c, idxs in self._class_to_idx.items() if len(idxs) >= 2
        ]
        self._all_valid_idx = [
            i for c in self._valid_classes for i in self._class_to_idx[c]
        ]
        print(f"  [triplet] {len(self._valid_classes)} classes with ≥2 samples, "
              f"{len(self._all_valid_idx)} valid graphs")

    def sample_triplets(self, n_triplets: int, seed: int | None = None) -> list[tuple[int, int, int]]:
        """
        Sample n_triplets (anchor_idx, positive_idx, negative_idx) tuples.

        Positive = same CWE class, Negative = different CWE class.
        """
        rng = random.Random(seed)
        triplets = []

        for _ in range(n_triplets):
            # Pick anchor class and anchor
            anchor_class = rng.choice(self._valid_classes)
            anchor_idx = rng.choice(self._class_to_idx[anchor_class])

            # Pick positive from same class (different sample)
            pos_candidates = [i for i in self._class_to_idx[anchor_class] if i != anchor_idx]
            if not pos_candidates:
                continue
            pos_idx = rng.choice(pos_candidates)

            # Pick negative from different class
            neg_class = rng.choice([c for c in self._valid_classes if c != anchor_class])
            neg_idx = rng.choice(self._class_to_idx[neg_class])

            triplets.append((anchor_idx, pos_idx, neg_idx))

        return triplets

    def get_data(self, idx: int) -> Data:
        return self._pyg_data[idx]

    def collate_triplet_batch(
        self, triplets: list[tuple[int, int, int]]
    ) -> tuple[Batch, Batch, Batch]:
        """Collate triplets into three PyG Batches (anchor, positive, negative)."""
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


def semi_hard_mine(
    model: GINCodeBERTModel,
    dataset: TripletDataset,
    batch_size: int,
    margin: float,
    device: str,
) -> list[tuple[int, int, int]]:
    """
    Online semi-hard negative mining.

    For each (anchor, positive) pair, find a negative that is farther
    than the positive but within the margin boundary.
    Falls back to random negatives if no semi-hard exists.
    """
    model.eval()
    # Embed all valid graphs
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
                a_emb = embeddings[anchor_i]
                p_emb = embeddings[pos_i]
                ap_dist = np.linalg.norm(a_emb - p_emb)

                # Find semi-hard negatives
                semi_hard = []
                for ni in rng.sample(neg_idx, min(50, len(neg_idx))):
                    n_emb = embeddings[ni]
                    an_dist = np.linalg.norm(a_emb - n_emb)
                    if ap_dist < an_dist < ap_dist + margin:
                        semi_hard.append(ni)

                if semi_hard:
                    triplets.append((anchor_i, pos_i, rng.choice(semi_hard)))
                elif neg_idx:
                    triplets.append((anchor_i, pos_i, rng.choice(neg_idx)))

    return triplets


class TripletTrainer:
    """
    Trains a GINCodeBERTModel with triplet margin loss.

    Args:
        model: The GIN model to train
        cfg: Configuration dict (gin_codebert section)
        device: torch device string
    """

    def __init__(self, model: GINCodeBERTModel, cfg: dict, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

        train_cfg = cfg.get("gin_codebert", {}).get("training", {})
        self.lr = train_cfg.get("lr", 1e-3)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.margin = train_cfg.get("margin", 0.3)
        self.epochs = train_cfg.get("epochs", 50)
        self.batch_size = train_cfg.get("batch_size", 16)
        self.triplets_per_epoch = train_cfg.get("triplets_per_epoch", 256)
        self.patience = train_cfg.get("patience", 10)
        self.mine_every = train_cfg.get("mine_every", 5)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)

    def train(
        self,
        dataset: TripletDataset,
        val_dataset: TripletDataset | None = None,
    ) -> dict[str, Any]:
        """
        Train the model. Returns training history.
        """
        history = {"train_loss": [], "val_loss": [], "epoch_time": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        print(f"\n  [trainer] Starting training: {self.epochs} epochs, "
              f"margin={self.margin}, lr={self.lr}")
        print(f"  [trainer] {self.triplets_per_epoch} triplets/epoch, "
              f"batch_size={self.batch_size}, patience={self.patience}")

        for epoch in range(self.epochs):
            t0 = time.perf_counter()

            # Mine triplets (semi-hard every N epochs, random otherwise)
            if epoch % self.mine_every == 0 and epoch > 0:
                triplets = semi_hard_mine(
                    self.model, dataset, self.batch_size, self.margin, self.device
                )
                if len(triplets) > self.triplets_per_epoch:
                    triplets = random.sample(triplets, self.triplets_per_epoch)
            else:
                triplets = dataset.sample_triplets(self.triplets_per_epoch, seed=epoch)

            if not triplets:
                print(f"    Epoch {epoch + 1}: no valid triplets, skipping")
                continue

            # Train
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(triplets), self.batch_size):
                batch_triplets = triplets[start:start + self.batch_size]
                anchor_batch, pos_batch, neg_batch = dataset.collate_triplet_batch(batch_triplets)

                anchor_batch = anchor_batch.to(self.device)
                pos_batch = pos_batch.to(self.device)
                neg_batch = neg_batch.to(self.device)

                self.optimizer.zero_grad()
                a_emb = self.model(anchor_batch)
                p_emb = self.model(pos_batch)
                n_emb = self.model(neg_batch)

                loss = self.loss_fn(a_emb, p_emb, n_emb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.perf_counter() - t0
            history["train_loss"].append(avg_loss)
            history["epoch_time"].append(elapsed)

            # Validation
            val_loss = None
            if val_dataset is not None:
                val_loss = self._eval_loss(val_dataset)
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
            else:
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_str = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
                print(f"    Epoch {epoch + 1:3d}/{self.epochs}: "
                      f"loss={avg_loss:.4f}{val_str} ({elapsed:.1f}s)")

            if patience_counter >= self.patience:
                print(f"    Early stopping at epoch {epoch + 1} (patience={self.patience})")
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        print(f"  [trainer] Training complete. Best loss: {best_val_loss:.4f}")
        return history

    def _eval_loss(self, dataset: TripletDataset, n_triplets: int = 64) -> float:
        """Evaluate average triplet loss on a dataset."""
        self.model.eval()
        triplets = dataset.sample_triplets(n_triplets, seed=9999)
        if not triplets:
            return 0.0

        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for start in range(0, len(triplets), self.batch_size):
                batch_triplets = triplets[start:start + self.batch_size]
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

    def save_checkpoint(self, path: Path, extra: dict | None = None) -> None:
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "in_dim": self.model.in_dim,
                "hidden_dim": self.model.hidden_dim,
                "out_dim": self.model.out_dim,
                "num_layers": self.model.num_layers,
                "dropout": self.model.dropout,
            },
        }
        if extra:
            state["extra"] = extra
        torch.save(state, path)
        print(f"  [trainer] Checkpoint saved → {path}")

    @staticmethod
    def load_checkpoint(path: Path, device: str = "cpu") -> GINCodeBERTModel:
        """Load a trained model from checkpoint."""
        state = torch.load(path, map_location=device, weights_only=False)
        cfg = state["model_config"]
        model = GINCodeBERTModel(
            in_dim=cfg["in_dim"],
            hidden_dim=cfg["hidden_dim"],
            out_dim=cfg["out_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
        model.load_state_dict(state["model_state_dict"])
        model.to(device)
        model.eval()
        return model
