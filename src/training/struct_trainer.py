"""
Triplet training infrastructure for the structural GIN (node-type features).

Uses CVE-ID / dir_name as the grouping label for triplet mining:
- Positive pairs: same CVE (different variants of the same vulnerability)
- Negative pairs: different CVE

This aligns training directly with the retrieval evaluation metric.
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

from src.embeddings.gin_struct_model import GINStructModel
from src.embeddings.wl import nx_to_pyg


class StructTripletDataset:
    """
    Manages graph data and triplet mining by CVE-ID.

    Groups graphs by CVE. Positives = same CVE (e.g., original + variants).
    Negatives = different CVE.
    """

    def __init__(
        self,
        graphs: list[nx.MultiDiGraph],
        labels: list[str],
    ):
        self._graphs = graphs
        self._labels = labels

        # Convert all graphs to PyG format
        print("  [struct_triplet] Converting graphs to PyG...")
        self._pyg_data: list[Data | None] = []
        for G in graphs:
            data = nx_to_pyg(G)
            self._pyg_data.append(data)
        print(f"  [struct_triplet] {len(graphs)} graphs converted")

        # Build class-to-indices mapping
        self._class_to_idx: dict[str, list[int]] = {}
        for i, label in enumerate(labels):
            if self._pyg_data[i] is None:
                continue
            if self._pyg_data[i].x.shape[0] < 2:
                continue
            self._class_to_idx.setdefault(label, []).append(i)

        # Filter classes with at least 2 samples (needed for positive pairs)
        self._valid_classes = [
            c for c, idxs in self._class_to_idx.items() if len(idxs) >= 2
        ]
        self._all_valid_idx = [
            i for c in self._valid_classes for i in self._class_to_idx[c]
        ]
        # Classes with only 1 sample (can only be negatives)
        self._singleton_classes = [
            c for c, idxs in self._class_to_idx.items() if len(idxs) == 1
        ]
        self._singleton_idx = [
            i for c in self._singleton_classes for i in self._class_to_idx[c]
        ]
        # All valid indices for negative sampling (including singletons)
        self._all_neg_pool = self._all_valid_idx + self._singleton_idx

        print(f"  [struct_triplet] {len(self._valid_classes)} classes with ≥2 samples, "
              f"{len(self._all_valid_idx)} valid graphs, "
              f"{len(self._singleton_classes)} singletons")

    def sample_triplets(self, n_triplets: int, seed: int | None = None) -> list[tuple[int, int, int]]:
        """
        Sample (anchor_idx, positive_idx, negative_idx) tuples.

        Positive = same CVE, Negative = different CVE.
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
            neg_candidates = [i for i in self._all_neg_pool if self._labels[i] != anchor_class]
            if not neg_candidates:
                continue
            neg_idx = rng.choice(neg_candidates)

            triplets.append((anchor_idx, pos_idx, neg_idx))

        return triplets

    def get_data(self, idx: int) -> Data:
        return self._pyg_data[idx]

    def collate_triplet_batch(
        self, triplets: list[tuple[int, int, int]]
    ) -> tuple[Batch, Batch, Batch]:
        """Collate triplets into three PyG Batches."""
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


def semi_hard_mine_struct(
    model: GINStructModel,
    dataset: StructTripletDataset,
    batch_size: int,
    margin: float,
    device: str,
) -> list[tuple[int, int, int]]:
    """
    Online semi-hard negative mining for structural GIN.

    For each valid (anchor, positive) pair, find negatives that satisfy:
      dist(anchor, positive) < dist(anchor, negative) < dist(anchor, positive) + margin
    """
    model.eval()
    all_idx = dataset._all_valid_idx + dataset._singleton_idx
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

                # Sample candidates and find semi-hard
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
                    # Fallback to random hard negative
                    triplets.append((anchor_i, pos_i, rng.choice(neg_idx)))

    return triplets


class StructTripletTrainer:
    """
    Trains a GINStructModel with triplet margin loss.

    Uses CVE-level grouping for positive/negative pairs.
    """

    def __init__(self, model: GINStructModel, cfg: dict, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

        train_cfg = cfg.get("gin_struct", {}).get("training", {})
        self.lr = train_cfg.get("lr", 5e-4)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.margin = train_cfg.get("margin", 0.5)
        self.epochs = train_cfg.get("epochs", 100)
        self.batch_size = train_cfg.get("batch_size", 32)
        self.triplets_per_epoch = train_cfg.get("triplets_per_epoch", 512)
        self.patience = train_cfg.get("patience", 15)
        self.mine_every = train_cfg.get("mine_every", 5)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )
        self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)

    def train(
        self,
        dataset: StructTripletDataset,
        val_dataset: StructTripletDataset | None = None,
    ) -> dict[str, Any]:
        """Train the model. Returns training history."""
        history = {"train_loss": [], "val_loss": [], "epoch_time": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        print(f"\n  [struct_trainer] Starting training: {self.epochs} epochs, "
              f"margin={self.margin}, lr={self.lr}")
        print(f"  [struct_trainer] {self.triplets_per_epoch} triplets/epoch, "
              f"batch_size={self.batch_size}, patience={self.patience}")

        for epoch in range(self.epochs):
            t0 = time.perf_counter()

            # Mine triplets
            if epoch % self.mine_every == 0 and epoch > 0:
                triplets = semi_hard_mine_struct(
                    self.model, dataset, self.batch_size, self.margin, self.device
                )
                if len(triplets) > self.triplets_per_epoch:
                    triplets = random.sample(triplets, self.triplets_per_epoch)
                if not triplets:
                    triplets = dataset.sample_triplets(self.triplets_per_epoch, seed=epoch)
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

                # Skip NaN/Inf batches
                if not torch.isfinite(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

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

            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_str = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
                lr_now = self.scheduler.get_last_lr()[0]
                print(f"    Epoch {epoch + 1:3d}/{self.epochs}: "
                      f"loss={avg_loss:.4f}{val_str}, lr={lr_now:.2e} ({elapsed:.1f}s)")

            if patience_counter >= self.patience:
                print(f"    Early stopping at epoch {epoch + 1} (patience={self.patience})")
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        print(f"  [struct_trainer] Training complete. Best loss: {best_val_loss:.4f}")
        return history

    def _eval_loss(self, dataset: StructTripletDataset, n_triplets: int = 128) -> float:
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
                "hidden_dim": self.model.hidden_dim,
                "out_dim": self.model.out_dim,
                "num_layers": self.model.num_layers,
                "dropout": self.model.dropout,
                "frozen_compat": self.model.frozen_compat,
            },
        }
        if extra:
            state["extra"] = extra
        torch.save(state, path)
        print(f"  [struct_trainer] Checkpoint saved → {path}")

    @staticmethod
    def load_checkpoint(path: Path, device: str = "cpu") -> GINStructModel:
        """Load a trained model from checkpoint."""
        state = torch.load(path, map_location=device, weights_only=False)
        cfg = state["model_config"]
        model = GINStructModel(
            hidden_dim=cfg["hidden_dim"],
            out_dim=cfg["out_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            frozen_compat=cfg.get("frozen_compat", False),
        )
        model.load_state_dict(state["model_state_dict"])
        model.to(device)
        model.eval()
        return model
