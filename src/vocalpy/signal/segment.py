from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..sequence import Sequence
from .audio import smooth


def audio_amplitude(
    data: npt.NDArray,
    samplerate: int,
    smooth_win: int = 2,
    threshold: int = 5000,
    min_dur: float = 0.02,
    min_silent_dur: float = 0.002,
) -> Sequence | None:
    """Segment audio into a sequence of units.

    Applies a threshold below which is considered silence,
    and then finds all continuous periods above the threshold
    that are bordered by silent gaps.
    All such periods are considered a :class:`vocalpy.Unit`.
    This function returns a :class:`vocalpy.Sequence`
    of such units.

    Note that before segmenting,
    the audio is first smoothed with
    :func:`vocalpy.signal.segment.smooth`.

    Parameters
    ----------
    data : numpy.ndarray
        An audio signal as a :class:`numpy.ndarray`.
        Typically, the `data` attribute from a :class:`vocalpy.Audio` instance.
    samplerate : int
        The sampling rate.
        Typically, the `samplerate` attribute from a :class:`vocalpy.Audio` instance.
    smooth_win : integer
        Size of smoothing window in milliseconds. Default is 2.
    threshold : int
        Value above which amplitude is considered part of a segment.
        Default is 5000.
    min_dur : float
        Minimum duration of a segment, in seconds.
        Default is 0.02, i.e. 20 ms.
    min_silent_dur : float
        Minimum duration of silent gap between segments, in seconds.
        Default is 0.002, i.e. 2 ms.

    Returns
    -------
    sequence : vocalpy.Sequence
        A :class:`vocalpy.Sequence` made up of `vocalpy.Unit` instances.
    """
    smoothed = smooth(data, samplerate, smooth_win)
    above_th = smoothed > threshold
    h = [1, -1]
    # convolving with h causes:
    # +1 whenever above_th changes from 0 to 1
    # and -1 whenever above_th changes from 1 to 0
    above_th_convoluted = np.convolve(h, above_th)

    # always get in units of sample first, then convert to s
    onsets_sample = np.where(above_th_convoluted > 0)[0]
    offsets_sample = np.where(above_th_convoluted < 0)[0]
    onsets_s = onsets_sample / samplerate
    offsets_s = offsets_sample / samplerate

    if onsets_s.shape[0] < 1 or offsets_s.shape[0] < 1:
        return None  # because no onsets or offsets in this file

    # get rid of silent intervals that are shorter than min_silent_dur
    silent_gap_durs = onsets_s[1:] - offsets_s[:-1]  # duration of silent gaps
    keep_these = np.nonzero(silent_gap_durs > min_silent_dur)
    onsets_s = np.concatenate((onsets_s[0, np.newaxis], onsets_s[1:][keep_these]))
    offsets_s = np.concatenate((offsets_s[:-1][keep_these], offsets_s[-1, np.newaxis]))

    # eliminate syllables with duration shorter than min_dur
    unit_durs = offsets_s - onsets_s
    keep_these = np.nonzero(unit_durs > min_dur)
    onsets_s = onsets_s[keep_these]
    offsets_s = offsets_s[keep_these]

    return onsets_s, offsets_s
