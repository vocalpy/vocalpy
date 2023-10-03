"""Signal processing functions for audio."""
from __future__ import annotations

import numbers
import warnings

import numpy as np
import numpy.typing as npt
import scipy.signal

from ..audio import Audio


def bandpass_filtfilt(audio: Audio, freq_cutoffs=(500, 10000)) -> Audio:
    """Filter audio with band-pass filter, then perform zero-phase
    filtering with :func:`scipy.signal.filtfilt`.

    Parameters
    ----------
    audio: vocalpy.Audio
        An audio signal.
    freq_cutoffs : Iterable
        Cutoff frequencies for bandpass filter.
        List or tuple with two elements, default is ``(500, 10000)``.

    Returns
    -------
    audio : vocalpy.Audio
        New audio instance
    """
    if freq_cutoffs[0] <= 0:
        raise ValueError("Low frequency cutoff {} is invalid, " "must be greater than zero.".format(freq_cutoffs[0]))

    nyquist_rate = audio.samplerate / 2
    if freq_cutoffs[1] >= nyquist_rate:
        raise ValueError(
            f"High frequency cutoff ({freq_cutoffs[1]}) is invalid, " f"must be less than Nyquist rate: {nyquist_rate}."
        )

    if audio.data.shape[-1] < 387:
        numtaps = 64
    elif audio.data.shape[-1] < 771:
        numtaps = 128
    elif audio.data.shape[-1] < 1539:
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
    filtered = scipy.signal.filtfilt(b, a, audio.data, padlen=padlen)
    return Audio(data=filtered, samplerate=audio.samplerate)


def meansquared(audio: Audio, freq_cutoffs=(500, 10000), smooth_win: int = 2) -> npt.NDArray:
    """Convert audio to a Root-Mean-Square-like trace

    First applied a band-pass filter
    Rectifies audio signal by squaring, then smooths by taking
    the average within a window of size ``smooth_win``.

    Parameters
    ----------
    audio: vocalpy.Audio
        An audio signal.
    freq_cutoffs : Iterable
        Cutoff frequencies for bandpass filter.
        List or tuple with two elements, default is ``(500, 10000)``.
    smooth_win : integer
        Size of smoothing window, in milliseconds. Default is ``2``.

    Returns
    -------
    meansquared : numpy.ndarray
        The ``vocalpy.Audio.data`` after squaring and smoothing.
    """
    audio = bandpass_filtfilt(audio, freq_cutoffs)

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
