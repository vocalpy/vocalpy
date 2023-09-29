"""Signal processing functions for audio."""
from __future__ import annotations

import numbers
import warnings

import numpy as np
import numpy.typing as npt
import scipy.signal

from ..audio import Audio


def bandpass_filtfilt(audio: Audio, freq_cutoffs: tuple[int, int] = (500, 10000)) -> Audio:
    """Apply bandpass filter to audio, then perform zero-phase
    filtering with :func:`scipy.signal.filtfilt` function.

    Parameters
    ----------
    audio : vocalpy.Audio
        An instance of :class:`vocalpy.Audio`.
    freq_cutoffs : list
        Cutoff frequencies for bandpass filter.
        Tuple of two integers, low frequency
        and high frequency cutoff.
        Default is [500, 10000].

    Returns
    -------
    audio : vocalpy.Audio
        With pre-processing applied to the `data` attribute.
    """
    if freq_cutoffs[0] <= 0:
        raise ValueError(f"Low frequency cutoff {freq_cutoffs[0]} is invalid, " "must be greater than zero.")

    nyquist_rate = audio.samplerate / 2
    if freq_cutoffs[1] >= nyquist_rate:
        raise ValueError(
            f"High frequency cutoff {freq_cutoffs[1]} is invalid, " f"must be less than Nyquist rate, {nyquist_rate}."
        )

    data = audio.data
    if data.shape[-1] < 387:
        numtaps = 64
    elif data.shape[-1] < 771:
        numtaps = 128
    elif data.shape[-1] < 1539:
        numtaps = 256
    else:
        numtaps = 512

    cutoffs = np.asarray([freq_cutoffs[0] / nyquist_rate, freq_cutoffs[1] / nyquist_rate])
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
    filtdata = scipy.signal.filtfilt(b, a, data, padlen=padlen)
    return Audio(data=filtdata, samplerate=audio.samplerate, path=audio.path)


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
