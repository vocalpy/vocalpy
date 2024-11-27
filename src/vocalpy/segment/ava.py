"""Find segments in audio, using algorithm from ``ava`` package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter
from scipy.signal import stft

from ..params import Params
from ..segments import Segments

if TYPE_CHECKING:
    from .. import Sound


EPSILON = 1e-9


@dataclass
class AvaParams(Params):
    """Data class that represents parameters
    for :func:`vocalpy.segment.ava`.

    Constants in this module are instances of this class
    that represent parameters used in papers.

    Attributes
    ----------
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
    min_isi_dur : float, optional
        Minimum duration of inter-segment intervals, in seconds.
        If specified, any inter-segment intervals shorter than this value
        will be removed, and the adjacent segments merged.
        Default is None.
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
        which assumes the audio data is loaded as 16-bit integers.

    Examples
    --------
    >>> jourjineetal2023paths = voc.example('jourjine-et-al-2023')
    >>> wav_path = jourjine2023paths[0]
    >>> sound = voc.Sound.read(wav_path)
    >>> onsets, offsets = voc.segment.ava(sound, **voc.segment.ava.JOURJINEETAL2023)
    """

    nperseg: int = 1024
    noverlap: int = 512
    min_freq: float = 20e3
    max_freq: float = 125e3
    spect_min_val: float = 0.8
    spect_max_val: float = 6.0
    thresh_lowest: float = 0.3
    thresh_min: float = 0.3
    thresh_max: float = 0.35
    min_dur: float = 0.015
    max_dur: float = 1.0
    min_isi_dur: float | None = None
    use_softmax_amp: bool = False
    temperature: float = 0.01
    smoothing_timescale: float = 0.00025
    scale: bool = True
    scale_val: int | float = 2**15
    scale_dtype: npt.DTypeLike = np.int16


# from https://github.com/nickjourjine/peromyscus-pup-vocal-evolution/blob/main/scripts/Segmenting%20and%20UMAP.ipynb
JOURJINEETAL2023 = AvaParams(
    nperseg=1024,
    noverlap=512,
    min_freq=20e3,
    max_freq=125e3,
    spect_min_val=0.8,
    spect_max_val=6.0,
    thresh_lowest=0.3,
    thresh_min=0.3,
    thresh_max=0.35,
    min_dur=0.015,
    max_dur=1.0,
    min_isi_dur=0.004,
    use_softmax_amp=False,
    temperature=0.01,
    smoothing_timescale=0.00025,
)


# from https://github.com/ralphpeterson/gerbil-vocal-dialects/blob/main/figure1-audio-segmenting-example.ipynb
PETERSONETAL2023 = AvaParams(
    nperseg=512,
    noverlap=256,
    min_freq=500.0,
    max_freq=62.5e3,
    spect_min_val=-8.0,
    spect_max_val=-7.25,
    thresh_lowest=2,
    thresh_min=5,
    thresh_max=2,
    min_dur=0.03,
    max_dur=0.3,
    use_softmax_amp=False,
    temperature=0.01,
    smoothing_timescale=0.007,
)


def ava(
    sound: Sound,
    nperseg: int = 1024,
    noverlap: int = 512,
    min_freq: int = 30e3,
    max_freq: int = 110e3,
    spect_min_val: float | None = None,
    spect_max_val: float | None = None,
    thresh_lowest: float = 0.1,
    thresh_min: float = 0.2,
    thresh_max: float = 0.3,
    min_dur: float = 0.03,
    max_dur: float = 0.2,
    min_isi_dur: float | None = None,
    use_softmax_amp: bool = True,
    temperature: float = 0.5,
    smoothing_timescale: float = 0.007,
    scale: bool = True,
    scale_val: int | float = 2**15,
    scale_dtype: npt.DTypeLike = np.int16,
) -> Segments:
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
    min_isi_dur : float, optional
        Minimum duration of inter-segment intervals, in seconds.
        If specified, any inter-segment intervals shorter than this value
        will be removed, and the adjacent segments merged.
        Default is None.
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
        which assumes the audio data is loaded as 16-bit integers.

    Returns
    -------
    segments : vocalpy.Segments
        Instance of :class:`vocalpy.Segments` representing
        the segments found.

    Examples
    --------
    >>> jourjineetal2023paths = voc.example('jourjine-et-al-2023')
    >>> wav_path = jourjineetal2023paths[0]
    >>> sound = voc.Sound.read(wav_path)
    >>> params = {**voc.segment.ava.JOURJINEETAL2023}
    >>> del params['min_isi_dur']
    >>> segments = voc.segment.ava(sound, **params)
    >>> spect = voc.spectrogram(sound)
    >>> rows = 3; cols = 4
    >>> import matplotlib.pyplot as plt
    >>> fig, ax_arr = plt.subplots(rows, cols)
    >>> start_inds, stop_inds = segments.start_inds, segments.stop_inds
    >>> ax_to_use = ax_arr.ravel()[:start_inds.shape[0]]
    >>> for start_ind, stop_ind, ax in zip(start_inds, stop_inds, ax_to_use):
    ...     data = sound.data[:, start_ind:stop_ind]
    ...     newsound = voc.Sound(data=data, samplerate=sound.samplerate)
    ...     spect = voc.spectrogram(newsound)
    ...     ax.pcolormesh(spect.times, spect.frequencies, np.squeeze(spect.data))
    >>> for ax in ax_arr.ravel()[:start_inds.shape[0]]:
    ...     ax.set_axis_off()
    >>> for ax in ax_arr.ravel()[start_inds.shape[0]:]:
    ...     ax.remove()

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
    # TODO: return Spectrogram for each Segment when we return Segments
    f, t, spect = stft(
        data, sound.samplerate, nperseg=nperseg, noverlap=noverlap
    )
    i1 = np.searchsorted(f, min_freq)
    i2 = np.searchsorted(f, max_freq)
    f, spect = f[i1:i2], spect[i1:i2]
    spect = np.log(np.abs(spect) + EPSILON)
    spect -= spect_min_val
    spect /= spect_max_val - spect_min_val
    spect = np.clip(spect, 0.0, 1.0)

    # we determine `dt` here in case we need it for `amps`
    # we also use it below to remove segments shorter than the minimum allowed value
    dt = t[1] - t[0]

    # ---- calculate amplitude and smooth.
    if use_softmax_amp:
        temp = np.exp(spect / temperature)
        temp /= np.sum(temperature, axis=0) + EPSILON
        amps = np.sum(np.multiply(spect, temp), axis=0)
    else:
        amps = np.sum(spect, axis=0)
    amps = gaussian_filter(amps, smoothing_timescale / dt)

    # Find local maxima greater than thresh_max.
    local_maxima = []
    for i in range(1, len(amps) - 1, 1):
        if amps[i] > thresh_max and amps[i] == np.max(
            amps[i - 1 : i + 2]  # noqa: E203
        ):
            local_maxima.append(i)

    # Then search to the left and right for onsets and offsets.
    onsets, offsets = [], []
    for local_max in local_maxima:
        # skip this local_max if we are in a region we already declared a segment.
        # Note the next line in the original implementation had an off-by-one error,
        # that is fixed here to avoid occasionally duplicating the first segment. See:
        # https://github.com/pearsonlab/autoencoded-vocal-analysis/issues/12
        if len(offsets) > 0 and local_max < offsets[-1]:
            continue

        # first find onset
        i = local_max - 1
        while (
            i > 0
        ):  # could we do ``while i > min(0, i - max_syl_length)`` to speed up?
            # this if-else can be a single `if` with an `or`
            # and then I think we can remove the `if len(onsets)` blocks
            if amps[i] < thresh_lowest:
                onsets.append(i)
                break
            elif amps[i] < thresh_min and amps[i] == np.min(
                amps[i - 1 : i + 2]  # noqa: E203
            ):
                onsets.append(i)
                break
            i -= 1

        # if we found multiple onsets because of if/else, then only keep one
        if len(onsets) != len(offsets) + 1:
            onsets = onsets[: len(offsets)]
            continue

        # then find offset
        i = local_max + 1
        while i < len(
            amps
        ):  # could we do ``while i > min(amps, i + max_syl_length)`` to speed up?
            # this if-else can be a single `if` with an `or`
            # and then I think we can remove the `if len(onsets)` blocks
            if amps[i] < thresh_lowest:
                offsets.append(i)
                break
            elif amps[i] < thresh_min and amps[i] == np.min(
                amps[i - 1 : i + 2]  # noqa: E203
            ):
                offsets.append(i)
                break
            i += 1

        # if we found multiple offsets because of if/else, then only keep one
        # shouldn't this change offsets though?
        if len(onsets) != len(offsets):
            onsets = onsets[: len(offsets)]
            continue

    # Throw away segments that are too long or too short.
    min_dur_samples = int(np.floor(min_dur / dt))
    max_dur_samples = int(np.ceil(max_dur / dt))
    new_onsets, new_offsets = [], []
    for i in range(len(offsets)):
        t1, t2 = onsets[i], offsets[i]
        if t2 - t1 + 1 <= max_dur_samples and t2 - t1 + 1 >= min_dur_samples:
            new_onsets.append(t1 * dt)
            new_offsets.append(t2 * dt)
    onsets = np.array(new_onsets)
    offsets = np.array(new_offsets)

    if onsets.size == 0 and offsets.size == 0:
        # can't throw any intervals away (next code block)
        # if there's not any intervals, so, return empty Segments
        return Segments(
            np.array([]).astype(int),
            np.array([]).astype(int),
            sound.samplerate,
        )

    # Throw away inter-segment intervals that are too short, as is done in Jourjine et al., 2023
    # Note we do this **after** throwing away segments that are too long or too short.
    # We do this to replicate what was done in Jourjine et al., 2023, where they call `ava.get_onsets_offsets`
    # and then remove inter-syllable intervals less than a specified duration.
    # This means there is the possibility of throwing away some short segments that we might have merged,
    # if we'd removed inter-segment intervals *first*, which is what `vocalpy.segment.meansquared` does.
    if min_isi_dur is not None:
        isi_durs = onsets[1:] - offsets[:-1]
        keep_these = isi_durs > min_isi_dur
        # we don't keep seconds anymore since we're returning samples
        onsets = np.concatenate(
            (onsets[0, np.newaxis], onsets[1:][keep_these])
        )
        offsets = np.concatenate(
            (offsets[:-1][keep_these], offsets[-1, np.newaxis])
        )

    onsets_sample = (onsets * sound.samplerate).astype(int)
    offsets_sample = (offsets * sound.samplerate).astype(int)
    lengths = offsets_sample - onsets_sample
    # Handle edge case where we decide last time bin is offset,
    # and the time of the last time bin in `t` (as computed from `dt`) is greater than duration of sound.
    # Fixes https://github.com/vocalpy/vocalpy/issues/167
    if onsets_sample[-1] + lengths[-1] > sound.samples:
        # set length to be "until the end of the sound"
        lengths[-1] = sound.samples - onsets_sample[-1]
    return Segments(
        start_inds=onsets_sample, lengths=lengths, samplerate=sound.samplerate
    )
