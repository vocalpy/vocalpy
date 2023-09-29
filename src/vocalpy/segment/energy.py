from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .. import signal

if TYPE_CHECKING:
    from .. import Audio


def energy(
    audio: Audio,
    smooth_win: int = 2,
    threshold: int = 5000,
    min_dur: float = 0.02,
    min_silent_dur: float = 0.002,
) -> tuple[npt.NDArray, npt.NDArray] | None:
    """Find segments in audio by thresholding the smoothed energy.

    Converts audio to smoothed energy
    using :func:`vocalpy.signal.audio.smoothed_energy`,
    that computes a running average of the squared signal
    by convolving with a window of size ``smooth_win``
    milliseconds.
    Then finds all continuous periods
    above ``threshold``, that are considered candidate segments.
    Candidates are removed that have a duration less than
    ``minimum_dur``, and then any two segments with a silent
    gap between them less than ``min_silent_dur`` are merged
    into a single segment. The segments remaining after this
    post-processing are returned as onset and offset times
    in two numpy arrays.

    Parameters
    ----------
    audio: vocalpy.Audio
        An audio signal.
    smooth_win : integer
        Size of smoothing window in milliseconds. Default is 2.
    threshold : int
        Value above which smoothed energy is considered part of a segment.
        Default is 5000.
    min_dur : float
        Minimum duration of a segment, in seconds.
        Default is 0.02, i.e. 20 ms.
    min_silent_dur : float
        Minimum duration of silent gap between segments, in seconds.
        Default is 0.002, i.e. 2 ms.

    Returns
    -------
    onsets_s : numpy.ndarray
        Vector of onset times of segments, in seconds.
    offsets_s : numpy.ndarray
        Vector of offset times of segments, in seconds.
    """
    energy_smoothed = signal.audio.smoothed_energy(audio, smooth_win)
    above_th = energy_smoothed > threshold
    h = [1, -1]
    # convolving with h causes:
    # +1 whenever above_th changes from 0 to 1
    # and -1 whenever above_th changes from 1 to 0
    above_th_convoluted = np.convolve(h, above_th)

    # always get in units of sample first, then convert to s
    onsets_sample = np.where(above_th_convoluted > 0)[0]
    offsets_sample = np.where(above_th_convoluted < 0)[0]
    onsets_s = onsets_sample / audio.samplerate
    offsets_s = offsets_sample / audio.samplerate

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
