from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
import numpy.typing as npt

from .. import signal

if TYPE_CHECKING:
    from .. import Sound


def meansquared(
    sound: Sound,
    threshold: int = 5000,
    min_dur: float = 0.02,
    min_silent_dur: float = 0.002,
    freq_cutoffs: Iterable = (500, 10000),
    smooth_win: int = 2,
) -> tuple[npt.NDArray, npt.NDArray] | None:
    """Segment audio by thresholding the mean squared signal.

    Converts audio to the mean squared of the signal
    (using :func:`vocalpy.signal.audio.meansquared`).
    Then finds all continuous periods
    in the mean squared signal above ``threshold``.
    These periods are considered candidate segments.
    Candidates are removed that have a duration less than
    ``minimum_dur``; then, any two segments with a silent
    gap between them less than ``min_silent_dur`` are merged
    into a single segment. The segments remaining after this
    post-processing are returned as onset and offset times
    in two NumPy arrays.

    Note that :func:`vocalpy.signal.audio.meansquared`
    first filters the audio, with
    :func:`vocalpy.signal.audio.bandpass_filtfilt`,
    using ``freq_cutoffs``, and then computes
    a running average of the squared signal
    by convolving with a window of size ``smooth_win``
    milliseconds.

    Parameters
    ----------
    sound: vocalpy.Sound
        An audio signal.
    threshold : int
        Value above which mean squared signal is considered part of a segment.
        Default is 5000.
    min_dur : float
        Minimum duration of a segment, in seconds.
        Default is 0.02, i.e. 20 ms.
    min_silent_dur : float
        Minimum duration of silent gap between segments, in seconds.
        Default is 0.002, i.e. 2 ms.
    freq_cutoffs : Iterable
        Cutoff frequencies for bandpass filter.
        List or tuple with two elements, default is ``(500, 10000)``.
    smooth_win : int
        Size of smoothing window in milliseconds. Default is 2.

    Returns
    -------
    onsets_s : numpy.ndarray
        Vector of onset times of segments, in seconds.
    offsets_s : numpy.ndarray
        Vector of offset times of segments, in seconds.
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

    meansquared_ = signal.audio.meansquared(sound, freq_cutoffs, smooth_win)
    # we get rid of the channel dimension *after* calling ``signal.audio.meansquared``
    # because that function *does* work on multi-channel data
    meansquared_ = np.squeeze(meansquared_, axis=0)
    above_th = meansquared_ > threshold
    h = [1, -1]
    # convolving with h causes:
    # +1 whenever above_th changes from 0 to 1
    # and -1 whenever above_th changes from 1 to 0
    above_th_convoluted = np.convolve(h, above_th)

    # always get in units of sample first, then convert to s
    onsets_sample = np.where(above_th_convoluted > 0)[0]
    offsets_sample = np.where(above_th_convoluted < 0)[0]
    onsets_s = onsets_sample / sound.samplerate
    offsets_s = offsets_sample / sound.samplerate

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
