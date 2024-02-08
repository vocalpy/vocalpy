"""Spectral methods for Sound Analysis Tools for Matlab (SAT).

Code adapted from [1]_, [2]_, and [3]_.

.. [1] `Sound Analysis Tools <http://soundanalysispro.com/matlab-sat>`_ for Matlab (SAT) by Ofer Tchernichovski
.. [2] `birdsonganalysis <https://github.com/PaulEcoffet/birdsonganalysis>`_  by Paul Ecoffet
.. [3] `avn <https://github.com/theresekoch/avn/blob/main/avn/acoustics.py>`_
   by Therese Koch, specifically the acoustics module
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import librosa
import numpy as np
import numpy.typing as npt
import scipy.signal.windows

if TYPE_CHECKING:
    from .. import Audio, Spectrogram


def sat(
    audio: Audio, n_fft=400, hop_length=40, freq_range=0.5
) -> tuple[Spectrogram, npt.NDArray, npt.NDArray, float, npt.NDArray, npt.NDArray]:
    """Compute spectral representations needed to extract predefined acoustic features
    with :func:`vocalpy.features.sat.similarity_features`

    Parameters
    ----------
    audio : vocalpy.Audio
        Audio loaded from a file.
    n_fft : int
        FFT window size.
    hop_length : int
        Number of audio samples between adjacent STFT columns.
    freq_range : float
        Range of frequencies to use, given as a value
        between zero and one.
        Default is 0.5, which means
        "Use the first half of the frequencies,
        from zero to :math:`f_s/4`
        (half the Nyquist frequency)".

    Returns
    -------
    power_spectrogram : vocalpy.Spectrogram
        A multi-taper spectrogram computed using DPSS tapers
        as described in [1]_. See [2]_ and [3]_ for rationale.
    cepstrogram : numpy.ndarray
        The cepstra computed for every window.
    quefrencies : numpy.ndarray
        The quefrencies for the cepstrogram.
    max_freq : float
        The maximum frequency to use, computed with
        ``freq_range`` and the audio sampling rate.
    dSdt : numpy.ndarray
        Derivative of spectrogram with respect to time,
        computed as described in [3]_.
    dSdf : numpy.ndarray
        Derivative of spectrogram with respect to frequency,
        computed as described in [3]_.

    See Also
    --------
    :func:`vocalpy.features.sat`
    :func:`vocalpy.similarity.sat`

    Notes
    -----
    Code adapted from [4]_, [5]_, and [6]_.

    References
    ----------
    .. [1] Babadi, Behtash, and Emery N. Brown.
           "A review of multitaper spectral analysis."
            IEEE Transactions on Biomedical Engineering 61.5 (2014): 1555-1564.
    .. [2] Tchernichovski, Ofer, et al.
           "A procedure for an automated measurement of song similarity."
           Animal behaviour 59.6 (2000): 1167-1176.
    .. [3] Sound Analysis Pro manual: `http://soundanalysispro.com/manual`_.
    .. [4] `Sound Analysis Tools <http://soundanalysispro.com/matlab-sat>`_ for Matlab (SAT) by Ofer Tchernichovski
    .. [5] `birdsonganalysis <https://github.com/PaulEcoffet/birdsonganalysis>`_  by Paul Ecoffet
    .. [6] `avn <https://github.com/theresekoch/avn/blob/main/avn/acoustics.py>`_ by Therese Koch,
           specifically the ``acoustics`` module.
    """
    if not 0.0 < freq_range <= 1.0:
        raise ValueError(
            f"`freq_range` must be a float greater than zero and less than or equal to 1.0, but was: {freq_range}. "
            f"Please specify a value between zero and one inclusive specifying the percentage of the frequencies "
            f"to use when extracting features with a frequency range"
        )

    # ---- make power spec
    audio_pad = np.pad(audio.data, pad_width=n_fft // 2)
    windows = librosa.util.frame(audio_pad, frame_length=n_fft, hop_length=hop_length, axis=0)
    tapers = scipy.signal.windows.dpss(n_fft, 1.5, Kmax=2)
    windows1 = windows * tapers[0, :]
    windows2 = windows * tapers[1, :]

    spectra1 = np.fft.fft(windows1, n=n_fft)
    spectra2 = np.fft.fft(windows2, n=n_fft)
    power_spectrogram = (np.abs(spectra1) + np.abs(spectra2)) ** 2
    f = librosa.fft_frequencies(sr=audio.samplerate, n_fft=n_fft)
    power_spectrogram = power_spectrogram.T[: f.shape[-1], :]

    # make power spectrum into Spectrogram
    t = librosa.frames_to_time(np.arange(windows.shape[0]), sr=audio.samplerate, hop_length=hop_length, n_fft=n_fft)
    from .. import Spectrogram

    power_spectrogram = Spectrogram(data=power_spectrogram, frequencies=f, times=t)

    spectra1_for_cepstrum = np.copy(spectra1)
    # next line is a fancy way of adding eps to zero values
    # so we don't get the enigmatic divide-by-zero error, and we don't get np.inf values
    # see https://github.com/numpy/numpy/issues/21560
    spectra1_for_cepstrum[spectra1_for_cepstrum == 0.] += np.finfo(spectra1_for_cepstrum.dtype).eps
    cepstrogram = np.fft.ifft(
        np.log(np.abs(spectra1_for_cepstrum)), n=n_fft
    ).real
    cepstrogram = cepstrogram.T
    quefrencies = np.array(np.arange(n_fft)) / audio.samplerate

    # freq_range means "use first `freq_range` percent of frequencies"
    max_freq_idx = int(np.floor(f.shape[0] * freq_range))
    max_freq = f[max_freq_idx]

    spectra1 = spectra1[:, :max_freq_idx]
    spectra2 = spectra2[:, :max_freq_idx]
    # time derivative of spectrum
    dSdt = (-spectra1.real * spectra2.real) - (spectra1.imag * spectra2.imag)
    dSdt = dSdt.T
    # frequency derivative of spectrum
    dSdf = (spectra1.imag * spectra2.real) - (spectra1.real * spectra2.imag)
    dSdf = dSdf.T

    return power_spectrogram, cepstrogram, quefrencies, max_freq, dSdt, dSdf
