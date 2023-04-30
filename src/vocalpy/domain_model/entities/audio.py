from __future__ import annotations

import pathlib
import reprlib

import attrs
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
            'Audio `data` had invalid number of dimensions, unable to determine number of channels. '
            f'Number of dimensions of `data` was: {data.ndim}'
            'The `data` array should have either one dimension (1 channel) or two dimensions.'
            '(number of channels will be equal to size of the first dimension, i.e., ``data.shape[0]``)'
        )
    return channels


@attrs.define
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

    Examples
    --------

    Reading audio from a file

    >>> import vocalpy as voc
    >>> audio = voc.Audio.read("1291.WAV")
    >>> audio
    Audio(data=array([ 0.   ... -0.00115967]), samplerate=44100, channels=1)
    """
    data: npt.NDArray = attrs.field()
    @data.validator
    def is_1d_or_2d(self, attribute, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f'Audio array `data` should be a numpy array, '
                f'but type was {type(value)}.'
            )

        if not (value.ndim == 1 or value.ndim == 2):
            raise ValueError(
                f'Audio array `data` should have either 1 or 2 dimensions, '
                f'but number of dimensions was {value.ndim}.'
            )

    samplerate: int = attrs.field(converter=int, validator=[
        attrs.validators.instance_of(int),
        attrs.validators.gt(0),
    ])
    channels: int | None = attrs.field(converter=attrs.converters.optional(int),
                                       validator=attrs.validators.optional([
                                           attrs.validators.instance_of(int),
                                           attrs.validators.gt(0),
                                       ]),
                                       default=None)
    source_path : pathlib.Path = attrs.field(converter=attrs.converters.optional(pathlib.Path),
                                             validator=attrs.validators.optional(
                                                 attrs.validators.instance_of(pathlib.Path)
                                             ),
                                             default=None)

    def __attrs_post_init__(self):
        channels_from_data = get_channels_from_data(self.data)

        if self.channels is None:
            self.channels = channels_from_data
        else:
            if channels_from_data != self.channels:
                raise ValueError(
                    f'Value specified for channels, {self.channels}, '
                    f'does not match value determined from data: {channels_from_data}'
                )

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'data={reprlib.repr(self.data)}, '
                f'samplerate={reprlib.repr(self.samplerate)}, '
                f'channels={self.channels})'
                )

    def asdict(self):
        """Convert this :class:`vocalpy.Audio`
        to a :class:`dict`.

        Returns
        -------
        audio_dict : dict
            A :class:`dict` with keys {'data', 'samplerate', 'channels', 'source_path'} that map
            to the corresponding attributes of this :class:`vocalpy.Audio`.
        """
        return attrs.asdict(self)

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return all(
            [np.array_equal(self.data, other.data),
             self.samplerate == other.samplerate,
             self.channels == other.channels,
             self.source_path == other.source_path,
             ]
        )

    def __ne__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return not self.__eq__(other)

    @classmethod
    def read(cls, path: str | pathlib.Path, **kwargs) -> 'vocalpy.Audio':
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
            raise FileNotFoundError(
                f'Audio file not found at path: {path}'
            )

        if path.suffix == '.cbin':
            data, samplerate = evfuncs.load_cbin(path)
        else:
            data, samplerate = soundfile.read(path, **kwargs)

        channels = get_channels_from_data(data)

        return cls(
            data=data,
            samplerate=samplerate,
            channels=channels,
            source_path=path
        )

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
        soundfile.write(file=path,
                        data=self.data,
                        samplerate=self.samplerate,
                        **kwargs)
        return AudioFile(path=path)

