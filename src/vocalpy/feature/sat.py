"""Feature extraction for Sound Analysis Toolbox (SAT).

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
import xarray as xr

if TYPE_CHECKING:
    from .. import Audio, Spectrogram

from .. import spectral

# get small number to avoid potential divide by zero errors
EPS = np.finfo(np.double).eps


def goodness_of_pitch(cepstrogram: npt.NDArray, quefrencies: npt.NDArray, max_F0: float = 1830.0) -> npt.NDArray:
    """Calculate goodness of pitch

    Finds the max in each column of ``cepstrogram``
    between ``quefrencies`` greater than ``1 / max_F0``
    and less than ``int(np.floor(len(cepstrogram) / 2))``.

    Parameters
    ----------
    cepstrogram : numpy.ndarray
        Cepstrogram returned by :func:`vocalpy.spectral.sat`,
        matrix with dimensions (quefrencies, time bins).
    quefrencies : numpy.ndarray
        The quefrencies for the cepstrogram.
    max_F0 : float
        Maximum frequency to consider,
        that becomes the lowest ``quefrency``
        used when computing goodness of pitch.

    Returns
    -------
    values : numpy.ndarray
        The goodness of pitch values for each column in ``cepstrogram``.

    Notes
    -----
    Goodness of pitch is an estimate of harmonic periodicity of a signal.
    Higher values indicate a more periodic sound (like a harmonic stack), whereas
    lower values indicate less periodic sounds (like noise). Formally, it is the
    peak of the cepstrum of the signal for fundamental frequencies below `max_F0`.

    Code adapted from [1]_, [2]_, and [3]_.
    Docs adapted from [1]_ and [3]_.

    References
    ----------
    .. [1] `Sound Analysis Tools <http://soundanalysispro.com/matlab-sat>`_ for Matlab (SAT) by Ofer Tchernichovski
    .. [2] `birdsonganalysis <https://github.com/PaulEcoffet/birdsonganalysis>`_  by Paul Ecoffet
    .. [3] `avn <https://github.com/theresekoch/avn/blob/main/avn/acoustics.py>`_
       by Therese Koch, specifically the acoustics module
    """
    if max_F0 <= 0:
        raise ValueError(f"`max_F0` must be greater than zero but was: {max_F0}")
    quefrency_cutoff = 1 / max_F0
    if quefrency_cutoff > quefrencies.max():
        raise ValueError(
            f"`max_F0` of {max_F0} gives a quefrency cut-off of {quefrency_cutoff}, "
            f"but this is greater than the max value in `quefrencies`: {quefrencies.max()}"
        )
    min_quef_idx = np.min(np.argwhere(quefrencies > quefrency_cutoff)) - 1
    max_quef_idx = int(np.floor(len(cepstrogram) / 2))
    return np.max(cepstrogram[min_quef_idx:max_quef_idx, :], axis=0)


def mean_frequency(power_spectrogram: Spectrogram, min_freq: float = 380.0, max_freq: float = 11025.0) -> npt.NDArray:
    """Calculate mean frequency.

    Finds the mean for each column in ``power_spectrogram``,
    between the frequencies specified by ``min_freq`` and ``max_freq``.
    To find the mean, the frequencies are weighted by their power
    in ``power_spectrogram`` and then divided by the sum of that power.

    Parameters
    ----------
    power_spectrogram : vocalpy.Spectrogram
        Spectrogram, returned by
        :func:`vocalpy.spectral.sat`.
    min_freq : float
        The minimum frequency to consider in ``power_spectrogram``.
    max_freq : float
        The maximum frequency to consider in ``power_spectrogram``.
        Returned by :func`:vocalpy.spectral.sat`, computing using the
        ``freq_range`` argument to that function.

    Returns
    -------
    values : numpy.ndarray
        The mean frequency of each column in ``power_spectrogram``.

    Notes
    -----
    This is one way to estimate the pitch of a signal.
    It is the center of the distribution of power across frequencies in the signal.
    For another estimate of pitch, see :func:`vocalpy.features.sat.pitch`.

    Code adapted from [1]_, [2]_, and [3]_.
    Docs adapted from [1]_ and [3]_.

    References
    ----------
    .. [1] `Sound Analysis Tools <http://soundanalysispro.com/matlab-sat>`_ for Matlab (SAT) by Ofer Tchernichovski
    .. [2] `birdsonganalysis <https://github.com/PaulEcoffet/birdsonganalysis>`_  by Paul Ecoffet
    .. [3] `avn <https://github.com/theresekoch/avn/blob/main/avn/acoustics.py>`_
       by Therese Koch, specifically the acoustics module

    See Also
    --------
    pitch
    """
    freq_inds = (power_spectrogram.frequencies > min_freq) & (power_spectrogram.frequencies < max_freq)
    P = power_spectrogram.data[freq_inds, :]
    P[P == 0.0] = EPS
    frequencies = power_spectrogram.frequencies[freq_inds]
    return np.sum(P * frequencies[:, np.newaxis], axis=0) / np.sum(P, axis=0)


def frequency_modulation(dSdt: npt.NDArray, dSdf: npt.NDArray) -> npt.NDArray:
    """Calculate frequency modulation.

    Parameters
    ----------
    dSdt : numpy.ndarray
        Derivative of spectrogram with respect to time, returned by
        :func:`vocalpy.spectral.sat`.
    dSdf : numpy.ndarray
        Derivative of spectrogram with respect to frequency, returned by
        :func:`vocalpy.spectral.sat`.

    Returns
    -------
    values : numpy.ndarray
        The frequency modulation for each column in ``dSdt`` and ``dSdf``.

    Notes
    -----
    Frequency modulation is a measure of the variance of the frequency composition over time
    of a signal [4]_.

    Code adapted from [1]_, [2]_, and [3]_.
    Docs adapted from [1]_ and [3]_.

    References
    ----------
    .. [1] `Sound Analysis Tools <http://soundanalysispro.com/matlab-sat>`_ for Matlab (SAT) by Ofer Tchernichovski
    .. [2] `birdsonganalysis <https://github.com/PaulEcoffet/birdsonganalysis>`_  by Paul Ecoffet
    .. [3] `avn <https://github.com/theresekoch/avn/blob/main/avn/acoustics.py>`_
       by Therese Koch, specifically the acoustics module
    .. [4] Bradbury, Jack W., and Sandra Lee Vehrencamp. Principles of animal communication. Vol. 132.
       Sunderland, MA: Sinauer Associates, 1998.
    """
    return np.arctan(np.max(dSdt, axis=0) / (np.max(dSdf, axis=0) + EPS))


def amplitude_modulation(dSdt: npt.NDArray) -> npt.NDArray:
    """Calculates the amplitude modulation.

    Parameters
    ----------
    dSdt : numpy.ndarray
        Derivative of spectrogram with respect to time, returned by
        :func:`vocalpy.spectral.sat`.

    Returns
    -------
    values : numpy.ndarray
        The amplitude modulation for each column in ``dSdt``.

    Notes
    -----
    Amplitude modulation is a measure of the variance in amplitude
    over time of a signal [4]_.

    Code adapted from [1]_, [2]_, and [3]_.
    Docs adapted from [1]_ and [3]_.

    References
    ----------
    .. [1] `Sound Analysis Tools <http://soundanalysispro.com/matlab-sat>`_ for Matlab (SAT) by Ofer Tchernichovski
    .. [2] `birdsonganalysis <https://github.com/PaulEcoffet/birdsonganalysis>`_  by Paul Ecoffet
    .. [3] `avn <https://github.com/theresekoch/avn/blob/main/avn/acoustics.py>`_
       by Therese Koch, specifically the acoustics module
    .. [4] Bradbury, Jack W., and Sandra Lee Vehrencamp. Principles of animal communication. Vol. 132.
       Sunderland, MA: Sinauer Associates, 1998.
    """
    return np.sum(dSdt, axis=0)


def entropy(power_spectrogram: Spectrogram, min_freq: float = 380.0, max_freq: float = 11025.0) -> npt.NDArray:
    """Calculate Wiener entropy

    Computes the Wiener entropy for each column in ``power_spectrogram``,
    between the frequencies specified by ``min_freq`` and ``max_freq``.

    Returns:
        np.array: array containing the  of each frame in the song interval.

    Parameters
    ----------
    power_spectrogram : vocalpy.Spectrogram
        Spectrogram, returned by
        :func:`vocalpy.spectral.sat`.
    min_freq : float
        The minimum frequency to consider in ``power_spectrogram``.
    max_freq : float
        The maximum frequency to consider in ``power_spectrogram``.
        Returned by :func`:vocalpy.spectral.sat`, computing using the
        ``freq_range`` argument to that function.

    Returns
    -------
    values : numpy.ndarray
        The log-scaled Weiner entropy for every column in ``power_spectrogram``.

    Notes
    ------
    Wiener entropy is a measure of the uniformity of power spread across frequency bands in a frame of audio.
    The output of this function is log-scaled Wiener entropy, which can range in value from 0 to negative
    infinity. A score close to 0 indicates broadly spread power across frequency bands, i.e. a less structured
    sound like white noise. A large negative score indicates low uniformity across frequency bands, i.e. a more
    structured sound like a harmonic stack or pure tone.

    Code adapted from [1]_, [2]_, and [3]_.
    Docs adapted from [1]_ and [3]_.

    References
    ----------
    .. [1] `Sound Analysis Tools <http://soundanalysispro.com/matlab-sat>`_ for Matlab (SAT) by Ofer Tchernichovski
    .. [2] `birdsonganalysis <https://github.com/PaulEcoffet/birdsonganalysis>`_  by Paul Ecoffet
    .. [3] `avn <https://github.com/theresekoch/avn/blob/main/avn/acoustics.py>`_
       by Therese Koch, specifically the acoustics module
    """
    freq_inds = (power_spectrogram.frequencies > min_freq) & (power_spectrogram.frequencies < max_freq)
    P = power_spectrogram.data[freq_inds, :]
    P[P == 0.0] = EPS
    # calculate entropy for current frame
    sum_log = np.sum(np.log(P), axis=0)
    log_sum = np.log(np.sum(P, axis=0) / (P.shape[0] - 1))
    return sum_log / (P.shape[0] - 1) - log_sum


def amplitude(
    power_spectrogram: Spectrogram, min_freq: float = 380.0, max_freq: float = 11025.0, baseline: float = 70.0
) -> npt.NDArray:
    """Calculate amplitude.

    Sums each column in ``power_spectrogram.data``
    for frequencies between ``min_freq`` and ``max_freq``.
    This gives the energy, that is then converted
    to decibels with ``10 * np.log10``. The ``baseline``
    is added to these values.

    Parameters
    ----------
    power_spectrogram : vocalpy.Spectrogram
        Spectrogram, returned by
        :func:`vocalpy.spectral.sat`.
    min_freq : float
        The minimum frequency to consider in ``power_spectrogram``.
    max_freq : float
        The maximum frequency to consider in ``power_spectrogram``.
        Returned by :func`:vocalpy.spectral.sat`, computing using the
        ``freq_range`` argument to that function.
    baseline : float
        The baseline value added, in decibels.
        The default is 70.0 dB, the value used by SAT
        and SAP.

    Returns
    -------
    values : numpy.ndarray

    Notes
    -----
    Code adapted from [1]_, [2]_, and [3]_.
    Docs adapted from [1]_ and [3]_.

    References
    ----------
    .. [1] `Sound Analysis Tools <http://soundanalysispro.com/matlab-sat>`_ for Matlab (SAT) by Ofer Tchernichovski
    .. [2] `birdsonganalysis <https://github.com/PaulEcoffet/birdsonganalysis>`_  by Paul Ecoffet
    .. [3] `avn <https://github.com/theresekoch/avn/blob/main/avn/acoustics.py>`_
       by Therese Koch, specifically the acoustics module
    """
    freq_inds = (power_spectrogram.frequencies > min_freq) & (power_spectrogram.frequencies < max_freq)
    P = power_spectrogram.data[freq_inds, :]
    P[P == 0.0] = EPS
    return 10 * np.log10(np.sum(P, axis=0)) + baseline


def pitch(
    audio: Audio,
    fmin: float = 380.0,
    fmax_yin: float = 8000.0,
    frame_length: int = 400,
    hop_length: int = 40,
    trough_threshold: float = 0.1,
):
    """Estimates the fundamental frequency (or pitch) using the YIN algorithm [1]_.

    The pitch is computed using :func:`librosa.yin`.

    Parameters
    ----------
    audio : vocalpy.Audio
        A :class:`vocalpy.Audio` instance.
    fmin : float
        Minimum frequency in Hertz.
        The recommended minimum is ``librosa.note_to_hz('C2')`` (~65 Hz)
        though lower values may be feasible.
    fmax_yin : float
        Maximum frequency in Hertz.
        Default is 8000.
    frame_length : int
        length of the frames in samples.
        Default is 400.
    hop_length : int
        number of audio samples between adjacent YIN predictions.
        If ``None``, defaults to ``frame_length // 4``.
        Default is 40.
    trough_threshold: float
        Absolute threshold for peak estimation.
        A float greater than 0.

    Returns
    -------
    pitch: np.array
        Time series of fundamental frequency in Hertz.

    Notes
    -----
    For more information on the YIN algorithm for fundamental frequency estimation,
    please refer to the documentation for :func:`librosa.yin`.

    Code adapted from [2]_, [3]_, and [4]_.
    Docs adapted from [2]_ and [4]_.

    References
    ----------
    .. [1] De Cheveigné, Alain, and Hideki Kawahara.
           “YIN, a fundamental frequency estimator for speech and music.”
           The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.
    .. [2] `Sound Analysis Tools <http://soundanalysispro.com/matlab-sat>`_ for Matlab (SAT) by Ofer Tchernichovski
    .. [3] `birdsonganalysis <https://github.com/PaulEcoffet/birdsonganalysis>`_  by Paul Ecoffet
    .. [4] `avn <https://github.com/theresekoch/avn/blob/main/avn/acoustics.py>`_
           by Therese Koch, specifically the acoustics module
    """
    return librosa.yin(
        audio.data,
        fmin=fmin,
        fmax=fmax_yin,
        sr=audio.samplerate,
        frame_length=frame_length,
        hop_length=hop_length,
        trough_threshold=trough_threshold,
    )


def similarity_features(
    audio: Audio,
    n_fft=400,
    hop_length=40,
    freq_range=0.5,
    min_freq: float = 380.0,
    amp_baseline: float = 70.0,
    max_F0: int = 1830.0,
    fmax_yin: float = 8000.0,
    trough_threshold: float = 0.1,
) -> xr.DataSet:
    """Extract all features used to compute similarity with SAT.

    Calls :func:`vocalpy.spectral.sat` to get spectral representations
    of the :class:`vocalpy.Audio`, then extracts all features
    from those spectral representations.

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
    min_freq : float
        Minimum frequency to consider when extracting features.
    amp_baseline : float
        The baseline value added, in decibels, to the amplitude feature.
        The default is 70.0 dB, the value used by SAT and SAP.
    max_F0 : float
        Maximum frequency to consider,
        that becomes the lowest ``quefrency``
        used when computing goodness of pitch.
    fmax_yin : float
        Maximum frequency in Hertz when computing pitch with YIN algorithm.
        Default is 8000.
    trough_threshold: float
        Absolute threshold for peak estimation.
        A float greater than 0.
        Used by :func:`pitch`.

    Returns
    -------
    features : xarray.Dataset
        An xarray.Dataset where the data variables are the features,
        and the coordinate is the time for each time bin.
    """
    # pitch, goodness, AM, FM, entropy
    power_spectrogram, cepstrogram, quefrencies, max_freq, dSdt, dSdf = spectral.sat(
        audio, n_fft, hop_length, freq_range
    )
    amp_ = amplitude(power_spectrogram, min_freq, max_freq, amp_baseline)
    pitch_ = pitch(
        audio, min_freq, fmax_yin, frame_length=n_fft, hop_length=hop_length, trough_threshold=trough_threshold
    )
    goodness_ = goodness_of_pitch(cepstrogram, quefrencies, max_F0)
    FM = frequency_modulation(dSdt, dSdf)
    AM = amplitude_modulation(dSdt)
    entropy_ = entropy(power_spectrogram, min_freq, max_freq)
    return xr.Dataset(
        {
            "amplitude": (["time"], amp_),
            "pitch": (["time"], pitch_),
            "goodness_of_pitch": (["time"], goodness_),
            "frequency_modulation": (["time"], FM),
            "amplitude_modulation": (["time"], AM),
            "entropy": (["time"], entropy_),
        },
        coords={"time": power_spectrogram.times},
    )
