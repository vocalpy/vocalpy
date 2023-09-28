"""Algorithms for segmentation.

These include algorithms for segmenting audio
and/or spectrograms into line segments
with start and stop times,
and algorithms that find bounding boxes in spectrograms
with high and low frequencies in addition
to the start and stop times.

.. autosummary::
   :toctree: generated/

   audio_amplitude
   ava
"""
from . import ava
from .audio_amplitude import audio_amplitude

__all__ = [
    "audio_amplitude",
    "ava",
]
