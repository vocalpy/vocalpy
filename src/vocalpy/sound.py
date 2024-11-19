"""Class that represents a sound."""

from __future__ import annotations

import pathlib
import reprlib
import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import soundfile

from ._vendor import evfuncs
from .audio_file import AudioFile

if TYPE_CHECKING:
    from .segments import Segments


class Sound:
    """Class that represents a sound.

    Attributes
    ----------
    data : numpy.ndarray
        The audio signal as a :class:`numpy.ndarray`,
        where the dimensions are (channels, samples).
    samplerate : int
        The sampling rate the audio signal was acquired at, in Hertz.
    channels : int
        The number of channels in the audio signal.
        Determined from the first dimension of ``data``.
    samples : int
        The number of samples in the audio signal.
        Determined from the last dimension of ``data``.
    duration : float
        Duration of the sound in seconds.
        Determined from the last dimension of ``data``
        and the ``samplerate``.

    Examples
    --------

    A :class:`~vocalpy.Sound` is read from a file.

    >>> sound_path = voc.example("bl26lb16.wav", return_path=True)
    >>> sound = voc.Sound.read(sound_path)
    >>> sound
    vocalpy.Sound(data=array([[-0.00... 0.00912476]]), samplerate=32000)

    The :class:`~vocalpy.Sound` class is designed as a
    domain-specific data container with attributes that
    help us avoid cluttering up code with variables
    that track the sampling rate, number of channels,
    and duration of the file.

    >>> sound = voc.example("bl26lb16.wav")
    >>> print(sound.samplerate)
    32000
    >>> print(sound.channels)
    1
    >>> print(sound.duration)
    7.254

    You can :func:`print` a :class:`~vocalpy.Sound`
    to see all the properties that are derived from
    the sampling rate and the shape of the
    underlying data array: the number of channels,
    the number of samples, and the duration in seconds.

    >>> sound = voc.example("bl26lb16.wav")
    >>> print(sound)
    vocalpy.Sound(data=array([[-0.00... 0.00912476]]), samplerate=32000), channels=1, samples=184463, duration=5.764)

    The :mod:`vocalpy` package tries to provide
    functions that take :class:`~vocalpy.Sound` instances as inputs,
    and return other domain-specific types as outputs,
    such as :class:`~vocalpy.Segments`, :class:`~vocalpy.Spectrogram`,
    and :class:`~vocalpy.Features`.
    If instead you need to work with the
    digital audio signal directly as a numpy array,
    you can access it through the :attr:`~vocalpy.Sound.data`
    attribute.

    >>> sound = voc.example("bl26lb16.wav")
    >>> sound_arr = sound.data

    Sound can be written to a file as well,
    in any format supported by :mod:`soundfile`.

    >>> sound = voc.example("bl26lb16.wav")
    >>> sound.write("bl26lb16-copy.wav")

    We can clip a sound to an arbitrary duration
    using the :meth:`~vocalpy.Sound.clip` method.
    This is useful if there are long, relatively silent periods
    before or after the animal sounds that we are interested in.

    >>> sound = voc.example("bl26lb16.wav")
    >>> sound_clip = sound.clip(0.1, 1.5)
    >>> print(sound_clip.duration)
    1.4

    If we want to clip from a start time to the end of the sound,
    we can just specify a time for `start`.

    >>> sound = voc.example("bl26lb16.wav")
    >>> sound_clip = sound.clip(0.5)
    >>> print(sound_clip.duration)
    1.4

    Likewise, if we want to clip from the start of the sound
    we can just specify a time for `stop`.
    Notice that we need to use a keyword argument here,
    since `start` is the first argument to :meth:`~vocalpy.Sound.clip`.

    >>> sound = voc.example("bl26lb16.wav")
    >>> sound_clip = sound.clip(stop=0.5)
    >>> print(sound_clip.duration)
    0.5

    If we want to segment an audio file
    into periods of animal sounds and periods of background,
    we can do that with one of the algorithms in
    :mod:`vocalpy.segment`. This will give us a
    :class:`~vocalpy.Segments` instance that we can then pass into
    the :meth:`~vocalpy.Sound.segment` method to get back
    a :class:`list` of :class:`~vocalpy.Sound` instances,
    one for each segment.

    >>> sound = voc.example("bl26lb16.wav")
    >>> segments = voc.segment.meansquared(sound, threshold=1000, min_dur=0.0002, min_silent_dur=0.004)
    >>> syllables = sound.segment(segments)
    >>> len(syllables)
    26

    You can also index a :class:`~vocalpy.Sound` as you would a
    :class:`numpy.array` and this will give you back a new
    :class:`~vocalpy.Sound`.
    One place where this is useful is when you have multi-channel
    audio, and you only want one channel, or you want to iterate
    over the channels.

    >>> sound = voc.example("fruitfly-song-multichannel.wav")
    >>> a_channel = sound[0, :]
    >>> print(a_channel)
    vocalpy.Sound(data=array([[-0.00...-0.00723267]]), samplerate=10000), channels=1, samples=15000, duration=1.500)
    >>> for channel in sound:
    ...     print(channel)
    vocalpy.Sound(data=array([[-0.00...-0.00723267]]), samplerate=10000), channels=1, samples=15000, duration=1.500)
    vocalpy.Sound(data=array([[ 0.01... 0.00268555]]), samplerate=10000), channels=1, samples=15000, duration=1.500)
    vocalpy.Sound(data=array([[ 0.00...-0.00100708]]), samplerate=10000), channels=1, samples=15000, duration=1.500)

    This works with other methods of indexing,
    as shown below.

    >>> sound = voc.example("bl26lb16.wav")
    >>> print(sound.data.shape)
    >>> decimated = sound[:, ::10]  # keep every 10th sample -- not true downsampling, we don't change the sampling rate

    Note that we are just passing indexing directly
    to the underlying :class:`numpy.array`,
    not re-implementing the API.
    """

    def __init__(
        self,
        data: npt.NDArray,
        samplerate: int,
    ):
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"Sound array `data` should be a numpy array, "
                f"but type was {type(data)}."
            )
        if not (data.ndim == 1 or data.ndim == 2):
            raise ValueError(
                f"Sound array `data` should have either 1 or 2 dimensions, "
                f"but number of dimensions was {data.ndim}."
            )
        if data.ndim == 1:
            data = data[np.newaxis, :]

        if data.shape[0] > data.shape[1]:
            warnings.warn(
                "The ``data`` passed in has more channels than samples: the number of channels (data.shape[0]) "
                f"is {data.shape[0]} and the number of samples (data.shape[1]) is {data.shape[1]}. "
                "You may need to verify you have passed in the data correctly.",
                stacklevel=2,
            )

        self.data = data

        if not isinstance(samplerate, int):
            raise TypeError(
                f"Type of ``samplerate`` must be int but was: {type(samplerate)}"
            )
        if not samplerate > 0:
            raise ValueError(
                f"Value of ``samplerate`` must be a positive integer, but was {samplerate}."
            )
        self.samplerate = samplerate

    @property
    def channels(self):
        return self.data.shape[0]

    @property
    def samples(self):
        return self.data.shape[1]

    @property
    def duration(self):
        return self.data.shape[1] / self.samplerate

    def __repr__(self):
        return (
            f"vocalpy.{self.__class__.__name__}("
            f"data={reprlib.repr(self.data)}, "
            f"samplerate={self.samplerate})"
        )

    def __str__(self):
        return (
            f"vocalpy.{self.__class__.__name__}("
            f"data={reprlib.repr(self.data)}, "
            f"samplerate={self.samplerate}), "
            f"channels={self.channels}, "
            f"samples={self.samples}, "
            f"duration={self.duration:.3f})"
        )

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return all(
            [
                np.array_equal(self.data, other.data),
                self.samplerate == other.samplerate,
            ]
        )

    def __ne__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return not self.__eq__(other)

    @classmethod
    def read(
        cls,
        path: str | pathlib.Path,
        dtype: npt.DTypeLike = np.float64,
        **kwargs,
    ) -> "Self":  # noqa: F821
        """Read audio from ``path``.

        Parameters
        ----------
        path : str, pathlib.Path
            Path to file from which audio data should be read.
        **kwargs : dict, optional
            Other arguments to :func:`soundfile.read`:, refer to
            :module:`soundfile` documentation for details.
            Note that :method:`vocalpy.Sound.read` passes in the argument
            ``always_2d=True``, because we require `Sound.data`
            to always have a "channel" dimension.

        Returns
        -------
        sound : vocalpy.Sound
            A :class:`vocalpy.Sound` instance with ``data``
            read from ``path``.
        """
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Sound file not found at path: {path}")

        if path.name.endswith("cbin"):
            data, samplerate = evfuncs.load_cbin(path)
            if dtype in (np.float32, np.float64):
                # for consistency with soundfile,
                # we scale the cbin int16 data to range [-1.0, 1.0] when we cast to float
                # Next line is from https://stackoverflow.com/a/42544738/4906855, see comments there
                data = data.astype(dtype) / 32768.0
            elif dtype == np.int16:
                pass
            else:
                raise ValueError(
                    f"Invalid ``dtype`` for cbin audio: {dtype}. "
                    "Must be one of {numpy.int16, np.float32, np.float64}"
                )
            # evfuncs always gives us 1-dim, so we add channel dimension
            data = data[np.newaxis, :]
        else:
            data, samplerate = soundfile.read(
                path, always_2d=True, dtype=dtype, **kwargs
            )
            data = data.transpose(
                (1, 0)
            )  # dimensions (samples, channels) -> (channels, samples)

        return cls(data=data, samplerate=samplerate)

    def write(self, path: str | pathlib.Path, **kwargs) -> AudioFile:
        """Write audio data to a file.

        Parameters
        ----------
        path : str, pathlib.Path
            Path to file that audio data should be saved in.
        **kwargs: dict, optional
            Extra arguments to :func:`soundfile.write`.
            Refer to :module:`soundfile` documentation for details.
        """
        path = pathlib.Path(path)
        if path.name.endswith("cbin"):
            raise ValueError(
                "Extension for `path` was 'cbin', but `vocalpy.Sound.write` cannot write to the cbin format. "
                "Audio data from cbin files can be converted to wav as follows:\n"
                ">>> sound.data = sound.data.astype(np.float32) / 32768.0\n"
                "The above converts the int16 values to float values between -1.0 and 1.0. "
                "You can then save the data as a wav file:\n"
                ">>> sound.write('path.wav')\n"
            )
        # next line: swap axes because soundfile expects dimensions to be (samples, channels)
        soundfile.write(
            file=path,
            data=self.data.transpose((1, 0)),
            samplerate=self.samplerate,
            **kwargs,
        )
        return AudioFile(path=path)

    def __iter__(self):
        for channel in self.data:
            yield Sound(
                data=channel[np.newaxis, ...],
                samplerate=self.samplerate,
            )

    def __getitem__(self, key):
        if isinstance(key, (int, tuple, slice)):
            try:
                return Sound(
                    data=self.data[key],
                    samplerate=self.samplerate,
                )
            except IndexError as e:
                raise IndexError(
                    f"Invalid integer or slice for Sound with {self.data.shape[0]} channels: {key}"
                ) from e
        else:
            raise TypeError(
                f"Sound can be indexed with integer or slice, but type was: {type(key)}"
            )

    def segment(self, segments: Segments) -> list[Sound]:
        """Segment a sound, using a set of line :class:`~vocalpy.Segments`.

        Parameters
        ----------
        segments : vocalpy.Segments.
            A :class:`~vocalpy.Segments` instance,
            the output of a segmenting function
            in :mod:`vocalpy.segment`.

        Returns
        -------
        sounds : list
            A list of :class:`~vocalpy.Sound` instances,
            one for every segment in :class:`~vocalpy.Segments`.

        Examples
        --------
        >>> sound = voc.example("bells.wav")
        >>> segments = voc.segment.meansquared(sound)
        >>> syllables = sound.segment(segments)
        >>> len(syllables)
        10

        Notes
        -----
        The :meth`Sound.segment` method is used with the output
        of functions from :mod:`vocalpy.segment`, an instance of
        :class:`~vocalpy.Segments`. If you need to clip a
        :class:`~vocalpy.Sound` at arbitrary times, use the
        :meth:`~vocalpy.Sound.clip` method.

        See Also
        --------
        vocalpy.segment
        Sound.clip
        """
        from .segments import Segments

        if not isinstance(segments, Segments):
            raise TypeError(
                f"`segments` argument should be an instance of `vocalpy.Segments`, but type is: {type(segments)}"
            )
        if segments.samplerate != self.samplerate:
            warnings.warn(
                f"The `samplerate` attribute of `segments, {segments.samplerate}, "
                f"does not equal the `samplerate` of this `Sound`, {self.samplerate}. "
                "You may want to check the source of the segments.",
                stacklevel=2,
            )
        if (
            segments.start_inds[-1] + segments.lengths[-1]
            > self.data.shape[-1]
        ):
            raise ValueError(
                f"The offset of the last segment in `segments`, {segments.start_inds[-1] + segments.lengths[-1]}, "
                f"is greater than the last sample of this `Sound`, {self.data.shape[-1]}"
            )

        sounds_out = []
        for start_ind, length in zip(segments.start_inds, segments.lengths):
            sounds_out.append(
                Sound(
                    data=self.data[
                        :, start_ind : start_ind + length  # noqa : E203
                    ],
                    samplerate=self.samplerate,
                )
            )
        return sounds_out

    def clip(self, start: float = 0.0, stop: float | None = None) -> Sound:
        """Make a clip from this :class:`~vocalpy.Sound` that starts at time
        ``start`` in seconds and ends at time ``stop``.

        Parameters
        ----------
        start : float
            Start time for clip, in seconds.
            Default is 0.
        stop : float, optional.
            Stop time for clip, in seconds.
            Default is None, in which case
            the value will be set to the
            :attr:`~vocalpy.Sound.duration`
            of this :class:`~vocalpy.Sound`.

        Returns
        -------
        clip : vocalpy.Sound
            A new :class:`~vocalpy.Sound` with
            duration ``stop - start``.

        Examples
        --------
        >>> sound = voc.example('bl26lb16.wav')
        >>> clip = sound.clip(1.5, 2.5)
        >>> clip.duration
        1.0

        Notes
        -----
        The :meth:`~vocalpy.Sound.clip` method is used to clip a
        :class:`~vocalpy.Sound` at arbitrary times.
        If you need to segment an audio file into periods of
        animal sounds and periods of background,
        use one of the functions in :mod:`vocalpy.segment`
        to get an instance of :class:`~vocalpy.Segments`,
        that you can then use with the :meth`Sound.segment` method.

        See Also
        --------
        Sound.segment
        """
        if not isinstance(start, (float, np.floating)):
            raise TypeError(
                f"The `start` time for the clip must be a float type, but type was {type(start)}."
            )
        if start < 0.0:
            raise ValueError(
                f"Value for `start` time must be a non-negative number, but was: {start}"
            )
        if start > self.duration:
            raise ValueError(
                f"Value for `start` time, {start}, cannot be greater than this `Sound`'s duration, {self.duration}"
            )
        start_ind = int(start * self.samplerate)

        if stop is None:
            return Sound(
                # don't use stop ind, instead go all the way to the end
                data=self.data[:, start_ind:],
                samplerate=self.samplerate,
            )
        else:
            if not isinstance(stop, (float, np.floating)):
                raise TypeError(
                    f"The `stop` time for the clip must be a float type, but type was {type(start)}."
                )
            if stop < start:
                raise ValueError(
                    f"Value for `stop`, {stop}, is less than value for `start`, {start}. "
                    "Please specify a `stop` time for the clip greater than the `start` time."
                )
            if stop > self.duration:
                raise ValueError(
                    f"Value for `stop` time, {stop}, cannot be greater than this `Sound`'s duration, {self.duration}"
                )
            stop_ind = int(stop * self.samplerate)
            return Sound(
                data=self.data[:, start_ind:stop_ind],
                samplerate=self.samplerate,
            )

    def to_mono(self):
        """Convert a :class:`~vocalpy.Sound` to mono by averaging samples across channels.

        Examples
        --------

        >>> sound = voc.examples("WhiLbl0010")
        >>> print(sound.channels)
        2
        >>> sound_mono = sound.to_mono()
        >>> print(sound.channels)
        1

        Note that feature extraction functions operate on channels independently,
        so it may speed up your analysis to convert multi-channel audio to mono,
        if you do not need to consider channels indepedently.

        >>> import timeit
        >>> import numpy as np
        >>> sound = voc.examples("WhiLbl0010")
        >>> sound_mono = sound.to_mono()
        >>> np.mean(timeit.repeat("voc.feature.biosound(sound)", number=5, globals=globals()))
        np.float64(19.713963174959645)
        >>> np.mean(timeit.repeat("voc.feature.biosound(sound_mono)", number=5, globals=globals()))
        np.float64(9.917085491772742)

        Notes
        -----
        This method uses the :func:`librosa.to_mono` function.
        """
        if self.data.shape[0] == 1:
            return self
        else:
            import librosa

            return Sound(
                data=librosa.to_mono(self.data), samplerate=self.samplerate
            )
