"""Functions for making spectrograms.

.. autosummary::
   :toctree: generated/

   butter_bandpass
   butter_bandpass_filter
   spectrogram

Notes
-----
Filters adapted from SciPy cookbook:
https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
Spectrogram adapted from code by Kyle Kastner and Tim Sainburg:
https://github.com/timsainb/python_spectrograms_and_inversion
"""
from __future__ import annotations

import scipy.signal

__all__ = ["butter_bandpass", "butter_bandpass_filter"]


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y
