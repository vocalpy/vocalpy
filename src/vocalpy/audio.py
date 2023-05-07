from __future__ import annotations

import contextlib
import pathlib
import reprlib

import evfuncs
import numpy as np
import numpy.typing as npt
import soundfile

from .audio_file import AudioFile


def get_channels_from_data(data: npt.NDArray) -> int:
    """Determine the number of audio channels
    from audio data in a Numpy array.

    Parameters
    ----------
    data : numpy.ndarray
        An audio signal as a :class:`numpy.ndarray`.
        Either one-dimensional for data with one audio channel,
        or two-dimensional for data with one or more channels,
        where the dimensions are (samples, channels).

    Returns
    -------
    channels : int
        Number of channels in ``data``.
        Determined as described in the definition of ``data``.
    """
    if data.ndim == 1:
        # handles both soundfile and evfuncs
        channels = 1
    elif data.ndim == 2:
        # multi-channel sound loaded by soundfile will have channels as second dimension
        channels = data.shape[1]
    else:
        raise ValueError(
            "Audio `data` had invalid number of dimensions, unable to determine number of channels. "
            f"Number of dimensions of `data` was: {data.ndim}"
            "The `data` array should have either one dimension (1 channel) or two dimensions."
            "(number of channels will be equal to size of the first dimension, i.e., ``data.shape[0]``)"
        )
    return channels


def is_1d_or_2d_ndarray(data: npt.NDArray) -> None:
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Audio array `data` should be a numpy array, " f"but type was {type(data)}.")

    if not (data.ndim == 1 or data.ndim == 2):
        raise ValueError(
            f"Audio array `data` should have either 1 or 2 dimensions, " f"but number of dimensions was {data.ndim}."
        )


class Audio:
    """Class that represents an audio signal.

    Attributes
    ----------
    data : numpy.ndarray
        The audio signal as a :class:`numpy.ndarray`.
        Either one-dimensional for data with one audio channel,
        or two-dimensional for data with one or more channels,
        where the dimensions are (samples, channels).
    samplerate : int
        The sampling rate the audio signal was acquired at, in Hertz.
    channels : int, optional
        The number of channels in the audio signal.
        If not specified, this will be determined from ``data``
        as specified above. If specified, the value
        must match what is determined from ``data``
        or a ValueError will be raised.
    path : pathlib.Path
        The path to the audio file that this
        :class:`vocalpy.Audio` was read from.

    Examples
    --------

    Reading audio from a file

    >>> import vocalpy as voc
    >>> audio = voc.Audio.read("1291.WAV")
    >>> audio
    Audio(data=array([ 0.   ... -0.00115967]), samplerate=44100, channels=1)
    """

    def __init__(
        self,
        data: npt.NDArray | None = None,
        samplerate: int | None = None,
        path: str | pathlib.Path | None = None,
    ):
        if path:
            path = pathlib.Path(path)
        self.path = path

        if any((data is not None, samplerate is not None)):
            if not all((data is not None, samplerate is not None)):
                raise ValueError("Must provide both `data` and `samplerate`.")
        if data is not None:
            is_1d_or_2d_ndarray(data)
        self._data = data

        if samplerate is not None:
            if not isinstance(samplerate, int) or samplerate < 1:
                raise ValueError("`samplerate` must be a positive integer")
        self._samplerate = samplerate

        if data is not None:
            channels_from_data = get_channels_from_data(self.data)
            self._channels = channels_from_data
        else:
            self._channels = None

    def _read(self, **kwargs):
        if self.path.name.endswith("cbin"):
            data, samplerate = evfuncs.load_cbin(self.path)
        else:
            data, samplerate = soundfile.read(self.path, **kwargs)

        channels = get_channels_from_data(data)

        self._data = data
        self._samplerate = samplerate
        self._channels = channels

    @property
    def data(self):
        if self._data is None:
            self._read()
        return self._data

    @property
    def samplerate(self):
        if self._samplerate is None:
            self._read()
        return self._samplerate

    @property
    def channels(self):
        if self._channels is None:
            self._read()
        return self._channels

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"data={reprlib.repr(self._data)}, "
            f"samplerate={reprlib.repr(self._samplerate)}, "
            f"channels={self._channels}), "
            f"path={self.path}"
        )

    def asdict(self):
        """Convert this :class:`vocalpy.Audio`
        to a :class:`dict`.

        Returns
        -------
        audio_dict : dict
            A :class:`dict` with keys {'data', 'samplerate', 'channels', 'path'} that map
            to the corresponding attributes of this :class:`vocalpy.Audio`.
        """
        return {
            "data": self._data,
            "samplerate": self._samplerate,
            "channels": self._channels,
            "path": self.path,
        }

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return all(
            [
                np.array_equal(self.data, other.data),
                self.samplerate == other.samplerate,
                self.channels == other.channels,
                self.path == other.path,
            ]
        )

    def __ne__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return not self.__eq__(other)

    @classmethod
    def read(cls, path: str | pathlib.Path, **kwargs) -> "vocalpy.Audio":  # noqa: F821
        """Read audio from ``path``.

        Parameters
        ----------
        path : str, pathlib.Path
            Path to file from which audio data should be read.
        **kwargs : dict, optional
            Extra arguments to :func:`soundfile.read`: refer to
            :module:`soundfile` documentation for details.

        Returns
        -------
        audio : vocalpy.Audio
            A :class:`vocalpy.Audio` instance with ``data``
            read from ``path``.
        """
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found at path: {path}")

        if path.name.endswith("cbin"):
            data, samplerate = evfuncs.load_cbin(path)
        else:
            data, samplerate = soundfile.read(path, **kwargs)

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
        soundfile.write(file=path, data=self.data, samplerate=self.samplerate, **kwargs)
        return AudioFile(path=path)

    @contextlib.contextmanager
    def open(self, **kwargs):
        self._read(**kwargs)
        yield
        self._data = None
        self._samplerate = None
        self._channels = None
