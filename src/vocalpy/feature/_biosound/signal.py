"""Signal-processing functions from ``soundsig`` [1]_.
Adapted under MIT license.

.. [1] https://github.com/theunissenlab/soundsig
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal


def lowpass_filter(
    data: npt.NDArray,
    samplerate: int,
    cutoff_freq: int,
    filter_order: int = 5,
    rescale: bool = False,
) -> npt.NDArray:
    """Apply a low-pass filter to a sound.

    Parameters
    ----------
    data : numpy.ndarray
        The data from one channel of a :class:`~vocalpy.Sound`.
    samplerate : int
        The sampling rate of the ``data``.
    cutoff_freq : int
        The cutoff frequency of the filter, in Hz.
    filter_order
        The order of the filter.
    rescale

    Returns
    -------
    filtered : numpy.ndarray
        Low-pass filtered ``data``.
    """
    nyq = samplerate / 2.0
    b, a = scipy.signal.butter(filter_order, cutoff_freq / nyq)
    filtered_s = scipy.signal.filtfilt(b, a, data)
    if rescale:
        filtered_s /= filtered_s.max()
        filtered_s *= data.max()
    return filtered_s


def gaussian_window(N: int, nstd: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Generate a Gaussian window.

    Generates a window of length N and standard deviation nstd.

    Parameters
    ----------
    N : int
        Length of window.
    nstd : int
        Number of standard deviations in window.

    Returns
    -------
    gauss_t : numpy.ndarray
    gauss_window : numpy.ndarray
    """
    hnwinlen = (N + (1 - N % 2)) // 2
    gauss_t = np.arange(-hnwinlen, hnwinlen + 1, 1.0)
    gauss_std = float(N) / float(nstd)
    gauss_window = np.exp(-(gauss_t**2) / (2.0 * gauss_std**2)) / (
        gauss_std * np.sqrt(2 * np.pi)
    )
    return gauss_t, gauss_window


def correlation_function(
    s1: npt.NDArray,
    s2: npt.NDArray,
    lags: npt.NDArray,
    mean_subtract: bool = True,
    normalize: bool = True,
) -> npt.NDArray:
    """Computes the cross-correlation function between signals s1 and s2.

    Parameters
    ----------
    s1 : numpy.ndarray
        The first signal.
    s2 : numpy.ndarray
        The second signal.
    lags : numpy.ndarray
        An array of integers indicating the lags.
        The lags are in units of sample period.
    mean_subtract : bool
        If True, subtract the mean of s1 from s1, and the mean of s2 from s2,
        which is the standard thing to do. Default is True
    normalize : bool
        If True, then divide the correlation function
        by the product of standard deviations of s1 and s2.

    Returns
    -------
    cf : numpy.ndarray
        The cross correlation function evaluated at the lags.

    Notes
    -----
    The cross correlation function is computed as:

    ``cf(k) = sum_over_t( (s1(t) - s1.mean()) * (s2(t+k) - s2.mean()) ) / s1.std()*s2.std()``
    """

    assert len(s1) == len(
        s2
    ), "Signals must be same length! len(s1)=%d, len(s2)=%d" % (
        len(s1),
        len(s2),
    )
    assert np.sum(np.isnan(s1)) == 0, "There are NaNs in s1"
    assert np.sum(np.isnan(s2)) == 0, "There are NaNs in s2"

    s1_mean = 0
    s2_mean = 0
    if mean_subtract:
        s1_mean = s1.mean()
        s2_mean = s2.mean()

    s1_std = s1.std(ddof=1)
    s2_std = s2.std(ddof=1)
    s1_centered = s1 - s1_mean
    s2_centered = s2 - s2_mean
    N = len(s1)

    if N < lags.max():
        raise ValueError(
            "The maximum lag is larger than the length of the signal `s1`: "
            f"the length of signal `s1` is {N}, lags.max()={lags.max()}"
        )

    cf = np.zeros([len(lags)])
    for k, lag in enumerate(lags):
        if lag == 0:
            cf[k] = np.dot(s1_centered, s2_centered) / N
        elif lag > 0:
            cf[k] = np.dot(s1_centered[:-lag], s2_centered[lag:]) / (N - lag)
        elif lag < 0:
            cf[k] = np.dot(
                s1_centered[np.abs(lag) :], s2_centered[:lag]  # noqa : E203
            ) / (N + lag)

    if normalize:
        cf /= s1_std * s2_std

    return cf
