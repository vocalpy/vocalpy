"""Metrics for segmentation algorithms."""

from . import ir
from .ir import fscore, precision, recall

__all__ = [
    "ir",
    "fscore",
    "precision",
    "recall",
]
