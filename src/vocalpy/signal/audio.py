from __future__ import annotations

import numbers
import warnings

import numpy as np
import numpy.typing as npt


def smooth(data: npt.NDArray, samplerate: int, smooth_win: int = 2) -> npt.NDArray:
    """Filter raw audio and smooth signal
    used to calculate amplitude.

    Returns a numpy.ndarray instead of :class:`vocalpy.Audio`,
    since this transformation is destructive.

    Parameters
    ----------
    data : numpy.ndarray
        An audio signal as a :class:`numpy.ndarray`.
        Typically, the `data` attribute from a :class:`vocalpy.Audio` instance.
    samplerate : int
        The sampling rate.
        Typically, the `samplerate` attribute from a :class:`vocalpy.Audio` instance.
    freq_cutoffs : list
        Cutoff frequencies for bandpass filter.
        Two-element sequence of integers,
        (low frequency cutoff, high frequency cutoff).
        Default is [500, 10000].
        If None, bandpass filter is not applied.
    smooth_win : integer
        Size of smoothing window in milliseconds. Default is 2.

    Returns
    -------
    audio_smoothed : numpy.ndarray
        The `vocalpy.Audio.data` after smoothing.

    Rectifies audio signal by squaring, then smooths by taking
    the average within a window of size sm_win.
    This is a very literal translation from the Matlab function SmoothData.m
    by Evren Tumer. Uses the Thomas-Santana algorithm.
    """
    data = np.array(data)
    if issubclass(data.dtype.type, numbers.Integral):
        while np.any(np.abs(data) > np.sqrt(np.iinfo(data.dtype).max)):
            warnings.warn(
                f"Values in `data` would overflow when squaring because of dtype, {data.dtype};"
                f"casting to a larger integer dtype to avoid",
                stacklevel=2,
            )
            # make a new dtype string, endianness + type, plus the current itemsize squared
            new_dtype_str = str(data.dtype)[:-1] + str(int(str(data.dtype)[-1]) ** 2)
            data = data.astype(np.dtype(new_dtype_str))
    squared = np.power(data, 2)
    len = np.round(samplerate * smooth_win / 1000).astype(int)
    h = np.ones((len,)) / len
    smooth = np.convolve(squared, h)
    offset = round((smooth.shape[-1] - data.shape[-1]) / 2)
    return smooth[offset : data.shape[-1] + offset]
