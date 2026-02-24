"""Signal processing functions for audio."""

from __future__ import annotations

from typing import Sequence

import numpy as np
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
