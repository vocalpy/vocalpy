"""Functions for working with sound, from ``soundsig`` [1]_.
Adapted under MIT license.

.. [1] https://github.com/theunissenlab/soundsig
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.signal

from .signal import lowpass_filter


def temporal_envelope(
    data: npt.NDArray,
    samplerate: int,
    cutoff_freq: int = 200,
    filter_order: int = 4,
    resample_rate: int | None = None,
) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
    """Get the temporal envelope from the sound pressure waveform.

    Parameters
    ----------
    data : numpy.ndarray
        The data from one channel of a :class:`~vocalpy.Sound`.
    samplerate : int
        The sampling rate of the ``data``.
    cutoff_freq : int
        The cutoff frequency of the low-pass filter used to create the envelope.
    filter_order : int
        The order of the filter.
    resample_rate : int, None
        Rate to use when resampling.

    Returns
    -------
    env : numpy.ndarray
        The temporal envelope of the signal.
    """
    # rectify a zeroed version
    srect = np.abs(data - np.mean(data))

    if cutoff_freq is not None:
        srect = lowpass_filter(
            srect, samplerate, cutoff_freq, filter_order=filter_order
        )
        srect[srect < 0] = 0

    if resample_rate is not None:
        lensound = len(srect)
        t = (np.array(range(lensound), dtype=float)) / samplerate
        lenresampled = int(round(float(lensound) * resample_rate / samplerate))
        srectresampled, tresampled = scipy.signal.resample(
            srect, lenresampled, t=t, axis=0, window=None
        )
        return srectresampled, tresampled
    else:
        return srect
