"""Class that represents a spectrogram."""

from __future__ import annotations

import pathlib
import reprlib

import attrs
import numpy as np
import numpy.typing as npt

from .. import validators
from ..spectrogram_file import SpectrogramFile

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
        Must have either 2 dimensions (frequencies, times)
        or 3 dimensions (channels, frequencies, times).
    frequencies : numpy.ndarray
        A vector of size :math:`f` where values are frequencies
        at center of frequency bins in spectrogram.
    times : numpy.ndarray
        Vector of size :math:`t` where values are times
        at center of time bins in spectrogram.

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
    Spectrogram(data=array([[0.561... 0.        ]]), fequencies=array([[    0...50.        ]]), times=array([[0.000...6053968e+01]]))
    """

    data: npt.NDArray = attrs.field()

    @data.validator
    def validate_data(self, attribute, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Spectrogram array `data` should be a numpy array, "
                f"but type was {type(value)}."
            )

        if value.ndim not in (2, 3):
            raise ValueError(
                "Spectrogram array `data` must have either 2 dimensions (frequencies, times) "
                "or 3 dimensions (channels, frequencies, times), "
                f"but number of dimensions was {value.ndim}."
            )

    frequencies: npt.NDArray = attrs.field(
        validator=validators.attrs.is_1d_ndarray
    )
    times: npt.NDArray = attrs.field(validator=validators.attrs.is_1d_ndarray)

    def __attrs_post_init__(self):
        if self.data.ndim == 2:
            # "canonicalize" to always have 3 dimensions, (channels, frequencies, times)
            self.data = self.data[np.newaxis, ...]

        if not self.data.shape[1] == self.frequencies.shape[0]:
            raise ValueError(
                "Number of rows in spectrogram `data` should match number of elements in "
                f"`frequencies vector, but `data` has {self.data.shape[0]} rows and there are "
                f"{self.frequencies.shape[0]} elements in `frequencies`."
            )
        if not self.data.shape[2] == self.times.shape[0]:
            raise ValueError(
                "Number of columns in spectrogram `data` should match number of elements in "
                f"`times` vector, but `data` has {self.data.shape[1]} columns and there are "
                f"{self.times.shape[0]} elements in `times`."
            )

    def __repr__(self):
        return (
            f"vocalpy.{self.__class__.__name__}("
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
    def read(cls, path: str | pathlib.Path):
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
            raise FileNotFoundError(
                f"File with spectrogram not found at path specified:\n{path}"
            )

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
                raise KeyError(
                    f"Did not find key '{key}' in path: {path}"
                ) from e

        return cls(**kwargs)

    def write(self, path: [str, pathlib.Path]) -> SpectrogramFile:
        """Write this :class:`vocalpy.Spectrogram`
        to a Numpy npz file at the given ``path``.

        Parameters
        ----------
        path : str, pathlib.Path
            The path to where the path should be saved
            containing the spectrogram ``data``
            and associated arrays ``frequencies`` and ``times``.
            If this path does not already end with the extension
            ".npz", that extension will be added
            (by :func:`numpy.savez`).

        Returns
        -------
        spectrogram_file : SpectrogramFile
            An instance of :class:`SpectrogramFile`
            representing the saved spectrogram.
        """
        path = pathlib.Path(path)
        np.savez(
            path,
            data=self.data,
            frequencies=self.frequencies,
            times=self.times,
        )
        return SpectrogramFile(path=path)

    def __iter__(self):
        for channel in self.data:
            yield Spectrogram(
                data=channel[np.newaxis, ...],
                frequencies=self.frequencies,
                times=self.times,
            )

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            try:
                return Spectrogram(
                    data=self.data[key],
                    frequencies=self.frequencies,
                    times=self.times,
                )
            except IndexError as e:
                raise IndexError(
                    f"Invalid integer or slice for Spectrogram with {self.data.shape[0]} channels: {key}"
                ) from e
        else:
            raise TypeError(
                f"Spectrogram can be indexed with integer or slice, but type was: {type(key)}"
            )
