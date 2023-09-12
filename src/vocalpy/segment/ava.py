from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter
from scipy.signal import stft


EPSILON = 1e-9


def spectrogram(data: npt.NDArray, samplerate: int, nperseg: int = 1024, noverlap: int = 512,
                    min_freq: int = 30e3, max_freq: int =  110e3,
                    spect_min_val: float = 2.0, spect_max_val: float = 6.0
                    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compute a spectrogram the same way the ``ava`` package does.

    This is the default function used to generate spectrograms by
    :func:`vocalpy.segment.ava.segment`.

    Parameters
    ----------
    data : numpy.ndarray
        Audio data.
    samplerate : int
        Sampling rate for audio.
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
    spect_min_val : float
        Expected minimum value of spectrogram
        after transforming to the log of the
        magnitude. Used for a min-max scaling:
        :math:`(s - s_{min} / (s_{max} - s_{min})`
        where ``spect_min_val`` is :math:`s_{min}`.
        Default is 2.0.
    spect_max_val : float
        Expected maximum value of spectrogram
        after transforming to the log of the
        magnitude. Used for a min-max scaling:
        :math:`(s - s_{min} / (s_{max} - s_{min})`
        where ``spect_min_val`` is :math:`s_{min}`.
        Default is 6.0.

    Returns
    -------
    spect : numpy.ndarray
        Spectrogram, a matrix with shape ``(f, t)``.
    f : numpy.ndarray
        Vector of frequencies, same size as
        the ``f`` dimension of ``spec``.
    t : numpy.ndarray
        Vector of times, same size as
        the ``t`` dimension of ``spec``.

    Notes
    -----
    Default parameters are taking from example script here:
    https://github.com/pearsonlab/autoencoded-vocal-analysis/blob/master/examples/mouse_sylls_mwe.py
    Note that example script suggests tuning these parameters using functionality built into it,
    that we do not replicate here.
    """
    if not len(data) >= nperseg:
        raise ValueError(
            f"length of `audio`` {(len(data))} must be greater than or equal to ``nperseg``: {nperseg}"
        )

    f, t, spect = stft(data, samplerate, nperseg=nperseg, noverlap=noverlap)
    min_freq_ind = np.searchsorted(f, min_freq)
    max_freq_ind = np.searchsorted(f, max_freq)
    f, spect = f[min_freq_ind:max_freq_ind], spect[min_freq_ind:max_freq_ind]
    spect = np.log(np.abs(spect) + EPSILON)
    # This is like min-max scaling between 0. and 1,
    # but notice that we could end up with all values set to zero
    # depending on range of spectrogram before transformation.
    # An alternative would be to do ``spect = (spect - spect.min()) / (spect.max() - spect.min())``.
    spect = (spect - spect_min_val) / (spect_max_val - spect_min_val)
    spect = np.clip(spect, 0., 1.)
    return spect, f, t


def softmax(arr: npt.NDArray, t=0.5):
    """Softmax along first array dimension. Not numerically stable."""
    temp = np.exp(arr / t)
    temp /= np.sum(temp, axis=0) + EPSILON
    return np.sum(np.multiply(arr, temp), axis=0)


def segment(
    data: npt.NDArray, samplerate: int,
    spect_callback: Callable | None = None,
    thresh_lowest: float = 0.1, thresh_min: float = 0.2, thresh_max: float = 0.3,
    min_dur: float = 0.03, max_dur: float = 0.2,
    use_softmax_amp: bool = True, temperature: float = 0.5,
    smoothing_timescale: float = 0.007,
):
    """Find segments in audio, using algorithm
    from ``ava`` package.

    Segments audio by generating a spectrogram from it,
    summing power across frequencies, and then thresholding
    this summed spectral power as if it were an amplitude trace.

    The spectral power is segmented with three thresholds,
    ``thresh_lowest``, ``thresh_min``, and ``thresh_max``, where
    ``thresh_lowest <= thresh_min <= thresh_max``.
    The segmenting algorithm works as follows:
    first detect all local maxima that exceed ``thresh_max``.
    Then for each local maximum, find onsets and offsets.
    An offset is detected wherever a local maxima
    is followed by a subsequent local minimum
    in the summed spectral power less than ``thresh_min``,
    or when the power is less than ``thresh_lowest``.
    Onsets are located in the same way,
    by looking for a preceding local minimum
    less than ``thresh_min``, or any value less than ``thresh_lowest``.

    Parameters
    ----------
    audio : numpy.ndarray
        Raw audio samples.
    samplerate : int
        Sampling rate for audio.
    spect_callback : callable, optional
        Function used to compute spectrogram.
        If None, then :func:`vocalpy.segment.ava_segmenter.ava_spectrogram`
        is used with the default parameters.
    thresh_max : float
        Threshold used to find local maxima.
    thresh_min : float
        Threshold used to find local minima,
        in relation to local maxima.
        Used to find onsets and offsets of segments.
    thresh_lowest : float
        Lowest threshold used to find onsets and offsets
        of segments.
    min_dur : float
        Minimum duration of a segment, in seconds.
    max_dur : float
        Maximum duration of a segment, in seconds.
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

    Returns
    -------
    onsets_s : numpy.ndarray
        Vector of onset times of segments, in seconds.
    offsets_s : numpy.ndarray
        Vector of offset times of segments, in seconds.

    Notes
    -----
    This algorithm works well for isolated calls in short sound clips.
    For examples, see the mouse data in [3]_,
    from the dataset associated with [1]_.

    Code is adapted from [2]_.

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
    if spect_callback is None:
        spect_callback = spectrogram_ava
    spect, f, t = spect_callback(data, samplerate)

    dt = t[1] - t[0]
    # Calculate amplitude and smooth.
    if use_softmax_amp:
        amps = softmax(spect, t=temperature)
    else:
        amps = np.sum(spect, axis=0)
    amps = gaussian_filter(amps, smoothing_timescale / dt)

    # Find local maxima greater than thresh_max.
    local_maxima = []
    # replace this with a convolution to find local maxima?
    # actually I think we want `find_peaks`
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    for i in range(1, len(amps)-1,1):
        if amps[i] > thresh_max and amps[i] == np.max(amps[i - 1:i + 2]):
            local_maxima.append(i)

    # Then search to the left and right for onsets and offsets.
    onsets, offsets = [], []
    for local_max in local_maxima:
        # skip this local_max if we are in a region we already declared a syllabel
        if len(offsets) > 1 and local_max < offsets[-1]:
            continue

        # first find onset
        i = local_max - 1
        while i > 0:  # could we do ``while i > min(0, i - max_syl_length)`` to speed up?
            # this if-else can be a single `if` with an `or`
            # and then I think we can remove the `if len(onsets)` blocks
            if amps[i] < thresh_lowest:
                onsets.append(i)
                break
            elif amps[i] < thresh_min and amps[i] == np.min(amps[i-1:i+2]):
                onsets.append(i)
                break
            i -= 1

        # if we found multiple o        nsets because of if/else, then only keep one
        if len(onsets) != len(offsets) + 1:
            onsets = onsets[:len(offsets)]
            continue

        # then find offset
        i = local_max + 1
        while i < len(amps):  # could we do ``while i > min(amps, i + max_syl_length)`` to speed up?
            # this if-else can be a single `if` with an `or`
            # and then I think we can remove the `if len(onsets)` blocks
            if amps[i] < thresh_lowest:
                offsets.append(i)
                break
            elif amps[i] < thresh_min and amps[i] == np.min(amps[i-1:i+2]):
                offsets.append(i)
                break
            i += 1

        # if we found multiple offsets because of if/else, then only keep one
        # shouldn't this change offsets though?
        if len(onsets) != len(offsets):
            onsets = onsets[:len(offsets)]
            continue

    # Throw away syllables that are too long or too short.
    min_dur_samples = int(np.floor(min_dur / dt))
    max_dur_samples = int(np.ceil(max_dur / dt))
    new_onsets = []
    new_offsets = []
    for i in range(len(offsets)):
        t1, t2 = onsets[i], offsets[i]
        if t2 - t1 + 1 <= max_dur_samples and t2 - t1 + 1 >= min_dur_samples:
            new_onsets.append(t1 * dt)
            new_offsets.append(t2 * dt)

    return np.array(new_onsets), np.array(new_offsets)
