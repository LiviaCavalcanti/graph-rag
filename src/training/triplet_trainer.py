"""
Triplet loss trainer for the GIN-CodeBERT model.

Trains a GINCodeBERTModel using triplet margin loss with online
semi-hard negative mining. Uses CWE-ID as the class label for
triplet construction.
"""

from src.training import (
    GINCodeBERTModel,
    NodeCodeBERTEncoder,
    TripletDataset,
    TripletTrainer,
    semi_hard_mine,
)

__all__ = [
    "GINCodeBERTModel",
    "NodeCodeBERTEncoder",
    "TripletDataset",
    "TripletTrainer",
    "semi_hard_mine",
]
