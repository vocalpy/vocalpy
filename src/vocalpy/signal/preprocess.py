from __future__ import annotations

import numpy as np
import scipy.signal

from ..audio import Audio


# TODO: make class versions? Or don't do that because pytorch ended up undoing it, for some reason?
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
        raise ValueError("Low frequency cutoff {} is invalid, " "must be greater than zero.".format(freq_cutoffs[0]))

    nyquist_rate = audio.samplerate / 2
    if freq_cutoffs[1] >= nyquist_rate:
        raise ValueError(
            "High frequency cutoff {} is invalid, "
            "must be less than Nyquist rate, {}.".format(freq_cutoffs[1], nyquist_rate)
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
    return Audio(data=filtdata, samplerate=audio.samplerate, source_path=audio.source_path)
