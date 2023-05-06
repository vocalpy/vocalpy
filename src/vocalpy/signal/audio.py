from __future__ import annotations

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

    Applies a bandpass filter with the frequency cutoffs in spect_params,
    then rectifies the signal by squaring, and lastly smooths by taking
    the average within a window of size sm_win.
    This is a very literal translation from the Matlab function SmoothData.m
    by Evren Tumer. Uses the Thomas-Santana algorithm.
    """
    squared = np.power(data, 2)
    len = np.round(samplerate * smooth_win / 1000).astype(int)
    h = np.ones((len,)) / len
    smooth = np.convolve(squared, h)
    offset = round((smooth.shape[-1] - data.shape[-1]) / 2)
    return smooth[offset : data.shape[-1] + offset]
