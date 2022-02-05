import pathlib
import reprlib
from typing import Union

import numpy as np
import scipy.io

from .. import validators


SPECT_FILE_LOADING_FUNCTIONS = {
    'mat': scipy.io.loadmat,
    'npz': np.load,
}

VALID_SPECT_FILE_FORMATS = tuple(SPECT_FILE_LOADING_FUNCTIONS.keys())

VALID_KEYS = ('s', 't', 'f', 'audio_path')


def are_key_map_keys_valid(key_map: dict) -> bool:
    """tests if all keys in `key_map` are in `spectrogram.VALID_KEYS`"""
    keys = list(key_map.keys())
    return all([key in VALID_KEYS for key in keys])


def default_key_map() -> dict:
    """maps the valid keys to themselves"""
    default = {k: k for k in VALID_KEYS}
    return default


def add_default_key_mappings(key_map: dict) -> dict:
    """add default key mappings for any not specified"""
    key_map_copy = key_map.copy()
    default = default_key_map()
    for k, v in default.items():
        if k not in key_map_copy:
            key_map_copy[k] = v
    return key_map_copy


class Spectrogram:
    """class that represents a spectrogram saved in a file.

    Attributes
    ----------
    s : numpy.ndarray
        spectrogram, an (m x n) matrix
    f : numpy.ndarray
        vector of size m where values are frequencies at center of frequency bins in spectrogram
    t : numpy.ndarray
        vector of size n where values are times at center of time bins in spectrogram
    spect_path : str, pathlib.Path
        path to a file containing saved arrays.
    format : str
        format of file, one of {'npz', 'mat'}.
    key_map : dict
        that maps standard keys for accessing
        arrays in file to a different set of keys.
        Standard keys are {'s', 'f', 't', 'audio_path'} (defined below).
        Defaults to None, in which case the default mapping is used:
        `{'s': 's', 'f': 'f', 't': 't', 'audio_path': None}`.
    audio_path : pathlib.Path
        path to audio file from which spectrogram was generated. Optional, default is None.

    Examples
    --------
    >>> spect = vocalpy.Spectrogram.from_file("llb3_0066_2018_04_23_17_31_55.wav.mat")
    >>> spect
    Spectrogram(s=array([[0.561... 0.        ]]), f=array([[    0...50.        ]]), t=array([[0.000...6053968e+01]]),
    spect_path=PosixPath('llb3_0066_2018_04_23_17_31_55.wav.mat'), audio_path=None)
    """

    def __init__(self,
                 spect_path: Union[str, pathlib.Path] = None,
                 format: str = None,
                 s: np.ndarray = None,
                 t: np.ndarray = None,
                 f: np.ndarray = None,
                 audio_path: Union[str, pathlib.Path] = None,
                 key_map: dict = None):
        if format not in VALID_SPECT_FILE_FORMATS:
            raise ValueError(
                f'invalid extension for spectrogram file format: {format}. '
                f'Valid extensions are: {VALID_SPECT_FILE_FORMATS}'
            )

        if key_map:
            if not are_key_map_keys_valid(key_map):
                raise ValueError(
                    f'invalid key map: {key_map}.\n Valid keys for mapping are: {VALID_KEYS}'
                )
            key_map = add_default_key_mappings(key_map)
        if key_map is None:
            key_map = default_key_map()

        if spect_path is not None:
            spect_path = pathlib.Path(spect_path)

        self.spect_path = spect_path
        self.format = format
        self.key_map = key_map

        self._s = s
        self._f = t
        self._t = f
        self._audio_path = audio_path

    def __repr__(self):
        if any([getattr(self, attr) is None for attr in ('_s', '_t', '_f')]):
            self._load()

        return (f'{self.__class__.__name__}('
                f's={reprlib.repr(self._s)}, '
                f'f={reprlib.repr(self._f)}, '
                f't={reprlib.repr(self._t)}, '
                f'spect_path={self.spect_path!r}, '
                f'audio_path={self._audio_path!r})'
                )

    def _load(self):
        """function that lazy loads"""
        if not self.spect_path.exists():
            raise FileNotFoundError(
                f"did not find spectrogram file at path specified: {self.spect_path}"
            )

        spect_file_dict = SPECT_FILE_LOADING_FUNCTIONS[self.format](self.spect_path)

        try:
            s = np.array(spect_file_dict[self.key_map['s']])
        except KeyError as e:
            raise KeyError(
                f"Did not find spectrogram using key '{self.key_map['s']}' "
                f"in file loaded from spect_path: {self.spect_path}"
            ) from e

        if s.ndim < 2:
            raise ValueError(
                f'spectrogram `s` should have at least 2 dimensions, '
                f'but number of dimensions was {s.ndim}'
            )

        try:
            t = np.array(spect_file_dict[self.key_map['t']])
        except KeyError as e:
            raise KeyError(
                f"Did not find time bins vector using key '{self.key_map['t']}' "
                f"in file loaded from spect_path: {self.spect_path}"
            ) from e

        if not validators.is_1d_or_row_or_column(t):
            raise ValueError(
                f'time bins vector `t` should be a 1-dimensional array, '
                f'but number of dimensions was {t.ndim}'
            )

        try:
            f = np.array(spect_file_dict[self.key_map['f']])
        except KeyError as e:
            raise KeyError(
                f"Did not find frequency bins vector using key '{self.key_map['f']}' "
                f"in file loaded from spect_path: {self.spect_path}"
            ) from e

        if not validators.is_1d_or_row_or_column(f):
            raise ValueError(
                f'frequency bins vector `f` should be a 1-dimensional array, '
                f'but number of dimensions was {t.ndim}'
            )

        try:
            audio_path = spect_file_dict[self.key_map['audio_path']]
            if isinstance(audio_path, np.ndarray):
                audio_path = np.squeeze(audio_path).item()
            audio_path = pathlib.Path(audio_path)
        except KeyError:  # for audio_path only we default to None if it is not found
            audio_path = None

        self._s = s
        self._t = t
        self._f = f
        self._audio_path = audio_path

    @property
    def s(self) -> np.array:
        """spectrogram, a matrix"""
        if self._s is None:
            self._load()
        return self._s

    @property
    def t(self) -> np.array:
        """vector of times at the center of each time bin"""
        if self._t is None:
            self._load()
        return self._t

    @property
    def f(self) -> np.array:
        """vector of frequencies at the center of each frequency bin"""
        if self._f is None:
            self._load()
        return self._f

    @property
    def audio_path(self) -> pathlib.Path:
        """path to audio file from which spectrogram was generated"""
        if self._audio_path is None:
            self._load()
        return self._audio_path

    @classmethod
    def from_file(cls, spect_path: Union[str, pathlib.Path], format: str = None, key_map: dict = None):
        spect_path = pathlib.Path(spect_path)
        if format is None:
            format = spect_path.suffix[1:]  # [1:] to remove period from extension
            if format not in VALID_SPECT_FILE_FORMATS:
                raise ValueError(
                    f'invalid extension for spectrogram file format: {format}. '
                    f'Valid extensions are: {VALID_SPECT_FILE_FORMATS}'
                )
        if key_map:
            if not are_key_map_keys_valid(key_map):
                raise ValueError(
                    f'invalid key map: {key_map}.\n Valid keys for mapping are: {VALID_KEYS}'
                )
            key_map = add_default_key_mappings(key_map)
        else:
            key_map = default_key_map()
        return cls(spect_path, format, key_map)

    @classmethod
    def from_mat(cls, mat_path: [str, pathlib.Path], key_map: dict = None):
        return cls.from_file(mat_path, 'mat', key_map)

    @classmethod
    def from_npz(cls, npz_path: [str, pathlib.Path], key_map: dict = None):
        return cls.from_file(npz_path, 'npz', key_map)

    def to_file(self, spect_path: [str, pathlib.Path], key_map: dict = None, exist_ok: bool = False):
        spect_path = pathlib.Path(spect_path)
        if spect_path.exists() and not exist_ok:
            raise FileExistsError(
                f'spect_path already exists:\n{spect_path}.\nTo overwrite, set exist_ok argument to False.'
            )

        # in most(?) cases this won't be true
        # since we don't expect to be using `to_file` for something loaded from a file,
        # but we need to handle it gracefully
        if any([getattr(self, attr) is None for attr in ('_s', '_t', '_f')]):
            self._load()

        np.savez(
            spect_path, s=self._s, f=self._f, t=self._t
        )
