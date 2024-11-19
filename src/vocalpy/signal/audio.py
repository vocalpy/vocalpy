"""Signal processing functions for audio."""

from __future__ import annotations

import numbers
import warnings
from typing import Sequence

import numpy as np
import numpy.typing as npt
import scipy.signal

from ..sound import Sound


def bandpass_filtfilt(
    sound: Sound, freq_cutoffs: Sequence[int] = (500, 10000)
) -> Sound:
    """Filter audio with band-pass filter, then perform zero-phase
    filtering with :func:`scipy.signal.filtfilt`.

    Parameters
    ----------
    sound: vocalpy.Sound
        An audio signal.
    freq_cutoffs : Sequence
        Cutoff frequencies for bandpass filter.
        List or tuple with two elements, default is ``(500, 10000)``.

    Returns
    -------
    sound : vocalpy.Sound
        New audio instance
    """
    if freq_cutoffs[0] <= 0:
        raise ValueError(
            "Low frequency cutoff {} is invalid, must be greater than zero.".format(
                freq_cutoffs[0]
            )
        )

    nyquist_rate = sound.samplerate / 2
    if freq_cutoffs[1] >= nyquist_rate:
        raise ValueError(
            f"High frequency cutoff ({freq_cutoffs[1]}) is invalid, must be less than Nyquist rate: {nyquist_rate}."
        )

    if sound.data.shape[-1] < 387:
        numtaps = 64
    elif sound.data.shape[-1] < 771:
        numtaps = 128
    elif sound.data.shape[-1] < 1539:
        numtaps = 256
    else:
        numtaps = 512

    cutoffs = np.asarray(
        [freq_cutoffs[0] / nyquist_rate, freq_cutoffs[1] / nyquist_rate]
    )
    # code on which this is based, bandpass_filtfilt.m, says it uses Hann(ing)
    # window to design filter, but default for matlab's fir1
    # is actually Hamming
    # note that first parameter for scipy.signal.firwin is filter *length*
    # whereas argument to matlab's fir1 is filter *order*
    # for linear FIR, filter length is filter order + 1
    b = scipy.signal.firwin(numtaps + 1, cutoffs, pass_zero=False)
    a = np.zeros((numtaps + 1,))
    a[0] = 1  # make an "all-zero filter"
    padlen = np.max((b.shape[-1] - 1, a.shape[-1] - 1))
    filtered = scipy.signal.filtfilt(b, a, sound.data, padlen=padlen)
    return Sound(data=filtered, samplerate=sound.samplerate)


def meansquared(
    sound: Sound, freq_cutoffs=(500, 10000), smooth_win: int = 2
) -> npt.NDArray:
    """Convert audio to a Root-Mean-Square-like trace.

    This function first applies a band-pass filter, and then
    rectifies the audio signal by squaring. Finally, it smooths by taking
    the average within a window of size ``smooth_win``.

    Parameters
    ----------
    sound: vocalpy.Sound
        An audio signal. Multi-channel is supported.
    freq_cutoffs : Iterable
        Cutoff frequencies for bandpass filter.
        List or tuple with two elements, default is ``(500, 10000)``.
    smooth_win : integer
        Size of smoothing window, in milliseconds. Default is ``2``.

    Returns
    -------
    meansquared : numpy.ndarray
        The ``vocalpy.Sound.data`` after squaring and smoothing.

    See Also
    --------
    vocalpy.segment.meansquared
    """
    sound = bandpass_filtfilt(sound, freq_cutoffs)

    data = np.array(sound.data)
    if issubclass(data.dtype.type, numbers.Integral):
        while np.any(np.abs(data) > np.sqrt(np.iinfo(data.dtype).max)):
            warnings.warn(
                f"Values in `data` would overflow when squaring because of dtype, {data.dtype};"
                f"casting to a larger integer dtype to avoid",
                stacklevel=2,
            )
            # make a new dtype string, endianness + type, plus the current itemsize squared
            new_dtype_str = str(data.dtype)[:-1] + str(
                int(str(data.dtype)[-1]) ** 2
            )
            data = data.astype(np.dtype(new_dtype_str))
    squared = np.power(data, 2)
    len = np.round(sound.samplerate * smooth_win / 1000).astype(int)
    h = np.ones((len,)) / len
    # to convolve per channel, seems like the best we can do is a list comprehension
    # followed by re-building the array.
    # see https://scipy-cookbook.readthedocs.io/items/ApplyFIRFilter.html
    smooth = np.array([np.convolve(squared_, h) for squared_ in squared])
    # next two lines are basically the same as np.convolve(mode='valid'), but off by one
    # I think I did it this way to exactly replicate code in evsonganaly
    offset = round((smooth.shape[-1] - data.shape[-1]) / 2)
    return smooth[:, offset : data.shape[-1] + offset]  # noqa: E203
