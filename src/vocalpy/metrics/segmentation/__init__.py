"""Metrics for segmentation algorithms."""
from . import ir
from .ir import precision, recall, fscore

__all__ = [
    "ir",
    "fscore",
    "precision",
    "recall",
]
