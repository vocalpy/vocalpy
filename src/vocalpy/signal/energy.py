"""Functions that compute the energy of a signal.

Notes
-----
We use the term *energy* because these functions 
are used for *energy-based segmentation*, as 
defined in [1]_, or *energy-based detections*, 
as defined in [2]_.

Functions in :mod:`vocalpy.segment` call  these 
functions to get a framewise

You may also see terms like "amplitude thresholding" 
or "envelope".

Strictly speaking, the best term might be 
"energy spectral density"

References
----------
"""
from __future__ import annotations

import numbers
import warnings
from typing import Sequence

import numpy as np
import numpy.typing as npt
import scipy.signal

from ..sound import Sound



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
