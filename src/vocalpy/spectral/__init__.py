"""Spectral representations."""

from . import sat, soundsig
from .sat import sat_multitaper
from .soundsig import soundsig_spectro

__all__ = [
    "sat_multitaper",
    "sat",
    "soundsig",
    "soundsig_spectro",
]
