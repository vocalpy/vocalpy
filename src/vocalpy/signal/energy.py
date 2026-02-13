"""Functions that compute the energy of a signal.

Functions in :mod:`vocalpy.segment` call these 
functions to get a framewise measurement of energy 
in an audio signal. The frame is typically 
a fixed-size audio window, or a set of acoustic 
features computed from such a window, such as a 
single spectrum in an STFT or some other 
set of acoustic features.

Notes
-----
We use the term *energy* because these functions 
are used for *energy-based segmentation*, as 
defined in [1]_, or *energy-based detections*, 
as discussed in [2]_.

You may also see terms like *amplitude*,
as in "amplitude thresholding*,  
or *envelope*. Strictly speaking, the best term might be 
"energy spectral density" or "power spectral density" [3]_.

References
----------
.. [1] Kemp, T., Schmidt, M., Whypphal, M., & Waibel, A. (2000, June).
   Strategies for automatic segmentation of audio data.
   In 2000 ieee international conference on acoustics, speech, and signal processing.
   proceedings (cat. no. 00ch37100) (Vol. 3, pp. 1423-1426). IEEE.

.. [2] Araya‐Salas, M., Smith‐Vidaurre, G., Chaverri, G., Brenes, J. C., Chirino, 
   F., Elizondo‐Calvo, J., & Rico‐Guevara, A. (2023). 
   ohun: An R package for diagnosing and optimizing automatic sound event detection. 
   Methods in Ecology and Evolution, 14(9), 2259-2271.

.. [3] https://en.wikipedia.org/wiki/Spectral_density#Energy_spectral_density
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numbers
import warnings

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter
from scipy.signal import stft

from .audio import bandpass_filtfilt

if TYPE_CHECKING:
    from .. import Sound
    from ..spectrogram import Spectrogram


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


def ava(
    sound: Sound,
    nperseg: int = 1024,
    noverlap: int = 512,
    min_freq: int = 30e3,
    max_freq: int = 110e3,
    spect_min_val: float | None = None,
    spect_max_val: float | None = None,
    use_softmax_amp: bool = True,
    temperature: float = 0.5,
    smoothing_timescale: float = 0.007,
    scale: bool = True,
    scale_val: int | float = 2**15,
    scale_dtype: npt.DTypeLike = np.int16,
    epsilon: float = 1e-9,
    return_spect: bool = False,
) -> tuple[npt.NDArray, float] | tuple[npt.NDArray, float, Spectrogram]:
    """Compute spectral power using algorithm
    from ``ava`` package.

    Computes Short-Time Fourier Transform of :class:`vocalpy.Sound`, 
    then estimates spectral power density by summing power 
    across frequencies.

    Segments audio by generating a spectrogram from it,
    summing power across frequencies, and then thresholding
    this summed spectral power as if it were an amplitude trace.

    Parameters
    ----------
    sound : vocalpy.Sound
        Sound loaded from an audio file.
    nperseg : int
        Number of samples per segment for Short-Time Fourier Transform.
        Default is 1024.
    noverlap : int
        Number of samples to overlap per segment
        for Short-Time Fourier Transform.
        Default is 512.
    min_freq : int
        Minimum frequency. Spectrogram is "cropped"
        below this frequency (instead of, e.g.,
        bandpass filtering). Default is 30e3.
    max_freq : int
        Maximum frequency. Spectrogram is "cropped"
        above this frequency (instead of, e.g.,
        bandpass filtering). Default is 110e3.
    spect_min_val : float, optional
        Expected minimum value of spectrogram
        after transforming to the log of the
        magnitude. Used for a min-max scaling:
        :math:`(s - s_{min} / (s_{max} - s_{min})`
        where ``spect_min_val`` is :math:`s_{min}`.
        Default is None, in which case
        the minimum value of the spectrogram is used.
    spect_max_val : float, optional
        Expected maximum value of spectrogram
        after transforming to the log of the
        magnitude. Used for a min-max scaling:
        :math:`(s - s_{min} / (s_{max} - s_{min})`
        where ``spect_min_val`` is :math:`s_{min}`.
        Default is None, in which case
        the maximum value of the spectrogram is used.
    use_softmax_amp : bool
        If True, compute summed spectral power from spectrogram
        with a softmax operation on each column.
        Default is True.
    temperature : float
        Temperature for softmax. Only used if ``use_softmax_amp`` is True.
    smoothing_timescale : float
        Timescale to use when smoothing summed spectral power
        with a gaussian filter.
        The window size will be ``dt - smoothing_timescale / samplerate``,
        where ``dt`` is the size of a time bin in the spectrogram.
    scale : bool
        If True, scale the ``sound.data``.
        Default is True.
        This is needed to replicate the behavior of ``ava``,
        which assumes the audio data is loaded as 16-bit integers.
        Since the default for :class:`vocalpy.Sound` is to load sounds
        with a numpy dtype of float64, this function defaults to
        multiplying the ``sound.data`` by 2**15,
        and then casting to the int16 dtype.
        This replicates the behavior of the ``ava`` function,
        given data with dtype float64.
        If you have loaded a sound with a dtype of int16,
        then set this to False.
    scale_val :
        Value to multiply the ``sound.data`` by, to scale the data.
        Default is 2**15.
        Only used if ``scale`` is ``True``.
        This is needed to replicate the behavior of ``ava``,
        which assumes the audio data is loaded as 16-bit integers.
    scale_dtype : numpy.dtype
        Numpy Dtype to cast ``sound.data`` to, after scaling.
        Default is ``np.int16``.
        Only used if ``scale`` is ``True``.
        This is needed to replicate the behavior of ``ava``,
        which assumes the audio data is loaded as 16-bit integers.:class:`vocalpy.Spectrogram`
    return_spect : bool
        If True, return the computed STFT as a 
        :class:`vocalpy.Spectrogram` (note that this 
        function log transforms the STFT output).
        Default is False.

    Returns
    -------
    amps : npt.NDArray
        Amplitude envelope; the smoothed summed spectral 
        power density.
    dt : float
        The size of a time step (one element in `amps`) in seconds.
    spect : vocalpy.Spectrogram
        The log-transformed spectrogram that was used to compute `amps`.

    Notes
    -----
    Code is adapted from [2]_.
    Default parameters are taken from example script here:
    https://github.com/pearsonlab/autoencoded-vocal-analysis/blob/master/examples/mouse_sylls_mwe.py
    Note that example script suggests tuning these parameters using functionality built into it,
    that we do not replicate here.

    Versions of this algorithm were also used to segment
    rodent vocalizations in [4]_ (see code in [5]_)
    and [6]_ (see code in [7]_).

    References
    ----------
    .. [1] Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021).
      Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires.
      eLife, 10:e67855. https://doi.org/10.7554/eLife.67855

    .. [2] https://github.com/pearsonlab/autoencoded-vocal-analysis

    .. [3] Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021).
      Data from: Low-dimensional learned feature spaces quantify individual
      and group differences in vocal repertoires. Duke Research Data Repository.
      https://doi.org/10.7924/r4gq6zn8w

    .. [4] Nicholas Jourjine, Maya L. Woolfolk, Juan I. Sanguinetti-Scheck, John E. Sabatini,
      Sade McFadden, Anna K. Lindholm, Hopi E. Hoekstra,
      Two pup vocalization types are genetically and functionally separable in deer mice,
      Current Biology, 2023 https://doi.org/10.1016/j.cub.2023.02.045

    .. [5] https://github.com/nickjourjine/peromyscus-pup-vocal-evolution/blob/main/src/segmentation.py

    .. [6] Peterson, Ralph Emilio, Aman Choudhri, Catalin MItelut, Aramis Tanelus, Athena Capo-Battaglia,
      Alex H. Williams, David M. Schneider, and Dan H. Sanes.
      "Unsupervised discovery of family specific vocal usage in the Mongolian gerbil."
      bioRxiv (2023): 2023-03.

    .. [7] https://github.com/ralphpeterson/gerbil-vocal-dialects/blob/main/vocalization_segmenting.py
    """
    if sound.data.shape[0] > 1:
        raise ValueError(
            f"The ``sound`` has {sound.data.shape[0]} channels, but segmentation is not implemented "
            "for sounds with multiple channels. This is because there can be a different number of segments "
            "per channel, which cannot be represented as a rectangular array. To segment each channel, "
            "first split the channels into separate ``vocalpy.Sound`` instances, then pass each to this function."
            "For example,\n"
            ">>> sound_channels = [sound_ for sound_ in sound]  # split with a list comprehension\n"
            ">>> channel_segments = [vocalpy.segment.meansquared(sound_) for sound_ in sound_channels]\n"
        )

    data = np.squeeze(
        sound.data, axis=0
    )  # get rid of channels dim so we operate on scalars in main loop

    if scale:
        data = (data * scale_val).astype(scale_dtype)

    # ---- compute spectrogram
    f, t, spect = stft(
        data, sound.samplerate, nperseg=nperseg, noverlap=noverlap
    )
    i1 = np.searchsorted(f, min_freq)
    i2 = np.searchsorted(f, max_freq)
    f, spect = f[i1:i2], spect[i1:i2]
    spect = np.log(np.abs(spect) + epsilon)
    spect -= spect_min_val
    spect /= spect_max_val - spect_min_val
    spect = np.clip(spect, 0.0, 1.0)

    # we determine `dt` here in case we need it for `amps`
    # we also use it below to remove segments shorter than the minimum allowed value
    dt = t[1] - t[0]

    # ---- calculate amplitude and smooth.
    if use_softmax_amp:
        temp = np.exp(spect / temperature)
        temp /= np.sum(temperature, axis=0) + epsilon
        amps = np.sum(np.multiply(spect, temp), axis=0)
    else:
        amps = np.sum(spect, axis=0)
    amps = gaussian_filter(amps, smoothing_timescale / dt)

    if return_spect:
        from ..spectrogram import Spectrogram
        spect = Spectrogram(spect, f, t)
        return amps, dt, spect
    else:
        return amps, dt
