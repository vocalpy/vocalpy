from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..domain_model import Audio, Sequence, Unit


def smooth(data: npt.NDArray, samplerate: int, smooth_win: int = 2) -> npt.NDArray:
    """Filter raw audio and smooth signal
    used to calculate amplitude.

    Parameters
    ----------
    audio : vocalpy.Audio
        An instance of :class:`vocalpy.Audio`. 
    freq_cutoffs : list
        Cutoff frequencies for bandpass filter.
        Two-element sequence of integers, 
        (low frequency cutoff, high frequency cutoff).
        Default is [500, 10000].
        If None, bandpass filter is not applied.
    smooth_win : integer
        Size of smoothing window in milliseconds. Default is 2.

    Returns
    -------
    audio_smoothed : vocalpy.Audio
        With pre-processing applied to the `data` attribute.

    Applies a bandpass filter with the frequency cutoffs in spect_params,
    then rectifies the signal by squaring, and lastly smooths by taking
    the average within a window of size sm_win.
    This is a very literal translation from the Matlab function SmoothData.m
    by Evren Tumer. Uses the Thomas-Santana algorithm.
    """
    squared = np.power(data, 2)
    len = np.round(samplerate * smooth_win / 1000).astype(int)
    h = np.ones((len,)) / len
    smooth = np.convolve(squared, h)
    offset = round((smooth.shape[-1] - data.shape[-1]) / 2)
    return smooth[offset:data.shape[-1] + offset]


def segment(audio: Audio, threshold: int = 5000, min_dur: float = 0.02,
            min_silent_dur: float = 0.002, return_sample: bool = False) -> Sequence:
    """Segment audio into a sequence of units.
    
    Applies a threshold below which is considered silence, 
    and finds all segments 
    (periods below threshold) greater than a minimum sp 
    by finding silent gaps greater than 

    Parameters
    ----------
    smooth : np.ndarray
        Smoothed audio waveform, returned by evfuncs.smooth_data
    samp_freq : int
        Sampling frequency at which audio was recorded. Returned by
        evfuncs.load_cbin.
    threshold : int
        value above which amplitude is considered part of a segment.
        Default is 5000.
    min_dur : float
        minimum duration of a segment.
        In .not.mat files saved by evsonganaly, this value is called "min_dur"
        Default is 0.02, i.e. 20 ms.
    min_silent_dur : float
        minimum duration of silent gap between segment.
        In .not.mat files saved by evsonganaly, this value is called "min_int"
        Default is 0.002, i.e. 2 ms.
    return_sample : bool
        if True, returns the onsets and offsets in sample numbers (from audio signal).
        Default is False.

    Returns
    -------
    sequence : vocalpy.Sequence
        A :class:`vocalpy.Sequence` made up of `vocalpy.Unit` instances.
    """
    audio_data, samplerate = audio.data, audio.samplerate
    smoothed = smooth(audio_data, samplerate)
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
        return None, None  # because no onsets or offsets in this file

    # get rid of silent intervals that are shorter than min_silent_dur
    silent_gap_durs = onsets_s[1:] - offsets_s[:-1]  # duration of silent gaps
    keep_these = np.nonzero(silent_gap_durs > min_silent_dur)
    onsets_s = np.concatenate(
        (onsets_s[0, np.newaxis], onsets_s[1:][keep_these]))
    offsets_s = np.concatenate(
        (offsets_s[:-1][keep_these], offsets_s[-1, np.newaxis]))
    if return_sample:
        onsets_sample = np.concatenate(
            (onsets_sample[0, np.newaxis], onsets_sample[1:][keep_these]))
        offsets_sample = np.concatenate(
            (offsets_sample[:-1][keep_these], offsets_sample[-1, np.newaxis]))

    # eliminate syllables with duration shorter than min_dur
    unit_durs = offsets_s - onsets_s
    keep_these = np.nonzero(unit_durs > min_dur)
    onsets_s = onsets_s[keep_these]
    offsets_s = offsets_s[keep_these]

    onsets_sample = onsets_sample[keep_these]
    offsets_sample = offsets_sample[keep_these]

    units = []
    for onset_s, offset_s, onset_sample, offset_sample in zip(
        onsets_s, offsets_s, onsets_sample, offsets_sample
    ):
        units.append(
            Unit(onset_s=onset_s, offset_s=offset_s, onset_sample=onset_sample, offset_sample=offset_sample)
        )

    return Sequence(units=units)
