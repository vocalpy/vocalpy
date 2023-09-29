"""Signal processing functions for audio."""
from __future__ import annotations

import numbers
import warnings

import numpy as np
import numpy.typing as npt

from ..audio import Audio


def smoothed_energy(audio: Audio, smooth_win: int = 2) -> npt.NDArray:
    """Convert audio to energy
    and smooth by taking a moving average
    with a rectangular window.

    Parameters
    ----------
    audio: vocalpy.Audio
        An audio signal.
    smooth_win : integer
        Size of smoothing window, in milliseconds. Default is 2.

    Returns
    -------
    audio_smoothed : numpy.ndarray
        The `vocalpy.Audio.data` after smoothing.

    Rectifies audio signal by squaring, then smooths by taking
    the average within a window of size ``sm_win``.
    """
    data = np.array(audio.data)
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
    len = np.round(audio.samplerate * smooth_win / 1000).astype(int)
    h = np.ones((len,)) / len
    smooth = np.convolve(squared, h)
    offset = round((smooth.shape[-1] - data.shape[-1]) / 2)
    return smooth[offset : data.shape[-1] + offset]  # noqa: E203
