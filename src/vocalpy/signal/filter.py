"""Filters for signal processing.

Notes
-----
Adapted from SciPy cookbook:
https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
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
