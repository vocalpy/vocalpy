"""Class that represents a sound."""
from __future__ import annotations

import pathlib
import reprlib
import warnings

import numpy as np
import numpy.typing as npt
import soundfile

from ._vendor import evfuncs
from .audio_file import AudioFile


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
    path : pathlib.Path
        The path to the audio file that this
        :class:`vocalpy.Sound` was read from.

    Examples
    --------

    Reading audio from a file

    >>> import vocalpy as voc
    >>> sound = voc.Sound.read("1291.WAV")
    >>> sound
    Sound(data=array([ 0.   ... -0.00115967]), samplerate=44100, channels=1)
    """

    def __init__(
        self,
        data: npt.NDArray,
        samplerate: int,
        path: str | pathlib.Path | None = None,
    ):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Sound array `data` should be a numpy array, " f"but type was {type(data)}.")
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
            raise TypeError(f"Type of ``samplerate`` must be int but was: {type(samplerate)}")
        if not samplerate > 0:
            raise ValueError(f"Value of ``samplerate`` must be a positive integer, but was {samplerate}.")
        self.samplerate = samplerate

        if path is not None:
            path = pathlib.Path(path)
        self.path = path

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
            f"samplerate={reprlib.repr(self.samplerate)}, "
            f"path={self.path})"
        )

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return all(
            [
                np.array_equal(self.data, other.data),
                self.samplerate == other.samplerate,
                self.path == other.path,
            ]
        )

    def __ne__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return not self.__eq__(other)

    @classmethod
    def read(cls, path: str | pathlib.Path, dtype: npt.DTypeLike = np.float64, **kwargs) -> "Self":  # noqa: F821
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
                # Fixes
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
            data, samplerate = soundfile.read(path, always_2d=True, dtype=dtype, **kwargs)
            data = data.transpose((1, 0))  # dimensions (samples, channels) -> (channels, samples)

        return cls(data=data, samplerate=samplerate, path=path)

    def write(self, path: str | pathlib.Path, **kwargs) -> AudioFile:
        """Write audio data to a file.

        Uses the :func:`audiofile.write` function.

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
        soundfile.write(file=path, data=self.data.transpose((1, 0)), samplerate=self.samplerate, **kwargs)
        return AudioFile(path=path)

    def __iter__(self):
        for channel in self.data:
            yield Sound(
                data=channel[np.newaxis, ...],
                samplerate=self.samplerate,
                path=self.path,
            )

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            try:
                return Sound(
                    data=self.data[key],
                    samplerate=self.samplerate,
                    path=self.path,
                )
            except IndexError as e:
                raise IndexError(f"Invalid integer or slice for Sound with {self.data.shape[0]} channels: {key}") from e
        else:
            raise TypeError(f"Sound can be indexed with integer or slice, but type was: {type(key)}")
