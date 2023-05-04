from __future__ import annotations

import pathlib
import reprlib

import attrs
import numpy as np
import numpy.typing as npt

from . import validators
from .audio_file import AudioFile
from .spectrogram_file import SpectrogramFile

VALID_SPECT_FILE_EXTENSION = ".npz"


@attrs.define
class Spectrogram:
    r"""Class that represents a spectrogram.

    Attributes
    ----------
    data : numpy.ndarray
        The spectrogram itself, typically a matrix
        :math:`f \times t` where there are :math:`f`
        frequency bins and :math:`t` time bins.
        Must have at least 2 dimensions.
    frequencies : numpy.ndarray
        A vector of size :math:`f` where values are frequencies
        at center of frequency bins in spectrogram.
    times : numpy.ndarray
        Vector of size :math:`t` where values are times
        at center of time bins in spectrogram.
    source_path : pathlib.Path, optional
        Path to .npz file that spectrogram was loaded from.
        Optional, added automatically by the :meth:`~vocalpy.Spectrogram.read` method.
    source_audio_path : pathlib.Path, optional
        Path to audio file from which spectrogram was generated.
        Optional, default None. Can be specified as argument
        to :meth:`~vocalpy.Spectrogram.read` method.

    Examples
    --------
    An example of creating a spectrogram with toy data,
    to demonstrate the expected dimensions of the arrays.

    >>> import numpy as np
    >>> import vocalpy as voc
    >>> times = np.arange(data.shape[1]) / 32000  # 32000 is the sampling rate, to convert to seconds
    >>> frequencies = np.linspace(0, 10000, data.shape[0])
    >>> spect = voc.Spectrogram(data, frequencies, times)
    >>> print(spect)
    Spectrogram(data=array([[0.354... 0.81988536]]), frequencies=array([    0....000.        ]), times=array([0.0000...3.121875e-02]))  # noqa: E501

    An example of reading a spectrogram from an npz file.

    >>> spect = voc.Spectrogram.read("llb3_0066_2018_04_23_17_31_55.wav.npz")
    >>> spect
    Spectrogram(s=array([[0.561... 0.        ]]), f=array([[    0...50.        ]]), t=array([[0.000...6053968e+01]]),
    spect_path=PosixPath('llb3_0066_2018_04_23_17_31_55.wav.mat'), audio_path=None)
    """
    data: npt.NDArray = attrs.field()

    @data.validator
    def validate_data(self, attribute, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Spectrogram array `data` should be a numpy array, " f"but type was {type(value)}.")

        if value.ndim < 2:
            raise ValueError(
                f"Spectrogram array `data` should have at least 2 dimensions, "
                f"but number of dimensions was {value.ndim}."
            )

    frequencies: npt.NDArray = attrs.field(validator=validators.is_1d_ndarray)
    times: npt.NDArray = attrs.field(validator=validators.is_1d_ndarray)

    source_path: pathlib.Path = attrs.field(
        converter=attrs.converters.optional(pathlib.Path),
        validator=attrs.validators.optional(attrs.validators.instance_of(pathlib.Path)),
        default=None,
    )
    source_audio_path: pathlib.Path = attrs.field(
        converter=attrs.converters.optional(pathlib.Path),
        validator=attrs.validators.optional(attrs.validators.instance_of(pathlib.Path)),
        default=None,
    )

    def __attrs_post_init__(self):
        if not self.data.shape[0] == self.frequencies.shape[0]:
            raise ValueError(
                "Number of rows in spectrogram `data` should match number of elements in "
                f"`frequencies vector, but `data` has {self.data.shape[0]} rows and there are "
                f"{self.frequencies.shape[0]} elements in `frequencies`."
            )

        if not self.data.shape[1] == self.times.shape[0]:
            raise ValueError(
                "Number of columns in spectrogram `data` should match number of elements in "
                f"`times` vector, but `data` has {self.data.shape[1]} columns and there are "
                f"{self.times.shape[0]} elements in `times`."
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"data={reprlib.repr(self.data)}, "
            f"frequencies={reprlib.repr(self.frequencies)}, "
            f"times={reprlib.repr(self.times)})"
        )

    def asdict(self):
        """Convert this :class:`vocalpy.Spectrogram`
        to a :class:`dict`.

        Returns
        -------
        spect_dict : dict
            A :class:`dict` with keys {'data', 'frequencies', 'times'} that map
            to the corresponding attributes of this :class:`vocalpy.Spectrogram`.
        """
        return attrs.asdict(self)

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return all(
            [
                np.array_equal(self.data, other.data),
                np.array_equal(self.frequencies, other.frequencies),
                np.array_equal(self.times, other.times),
            ]
        )

    def __ne__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return not self.__eq__(other)

    @classmethod
    def read(cls, path: str | pathlib.Path, source_audio_path: str | pathlib.Path | None = None):
        """Read spectrogram and associated arrays from a Numpy npz file
        at the given ``path``.

        Parameters
        ----------
        path : str, pathlib.Path
            The path to the file containing the spectrogram ``s``
            and associated arrays ``f`` and ``t``.

        Returns
        -------
        spect : vocalpy.Spectrogram
            An instance of :class:`vocalpy.Spectrogram`
            containing the arrays loaded from ``path``.
        """
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File with spectrogram not found at path specified:\n{path}")

        if not path.suffix == VALID_SPECT_FILE_EXTENSION:
            raise ValueError(
                f"Invalid extension for path: '{path.suffix}'. "
                f"Should be a numpy array path with extension '{VALID_SPECT_FILE_EXTENSION}'."
            )

        spect_file_dict = np.load(str(path))

        kwargs = {}
        for key in ("data", "frequencies", "times"):
            try:
                kwargs[key] = np.array(spect_file_dict[key])
            except KeyError as e:
                raise KeyError(f"Did not find key '{key}' in path: {path}") from e

        return cls(source_path=path, source_audio_path=source_audio_path, **kwargs)

    def write(self, path: [str, pathlib.Path]):
        """Write this :class:`vocalpy.Spectrogram`
        to a Numpy .npz file at the given ``path``.

        Parameters
        ----------
        path : str, pathlib.Path
            The path to where the path should be saved
            containing the spectrogram ``data``
            and associated arrays ``frequencies`` and ``times``.
        """
        # TODO: deal with extension here
        path = pathlib.Path(path)
        np.savez(path, data=self.data, frequencies=self.frequencies, times=self.times)
        if self.source_audio_path:
            source_audio_file = AudioFile(path=self.source_audio_path)
        else:
            source_audio_file = None
        return SpectrogramFile(path=path, source_audio_file=source_audio_file)
