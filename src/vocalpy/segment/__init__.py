"""Algorithms for segmentation.

These include algorithms for segmenting audio
and/or spectrograms into line segments
with start and stop times,
and algorithms that find bounding boxes in spectrograms
with high and low frequencies in addition
to the start and stop times.
"""

from .ava import JOURJINEETAL2023, PETERSONETAL2023, AvaParams, ava
from .meansquared import MeanSquaredParams, meansquared

__all__ = [
    "ava",
    "AvaParams",
    "meansquared",
    "MeanSquaredParams",
    "JOURJINEETAL2023",
    "PETERSONETAL2023",
]
