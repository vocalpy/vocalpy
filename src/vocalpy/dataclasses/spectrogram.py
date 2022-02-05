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

VALID_KEYS = ('s', 'f', 't', 'audio_path')


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
                 s: np.ndarray,
                 t: np.ndarray,
                 f: np.ndarray,
                 spect_path: Union[str, pathlib.Path] = None,
                 audio_path: Union[str, pathlib.Path] = None):
        self.s = s
        self.f = t
        self.t = f
        self.spect_path = spect_path
        self.audio_path = audio_path

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

    @property
    def s(self) -> np.array:
        """spectrogram, a matrix"""
        return self._s

    @s.setter
    def s(self, value: np.array):
        value = np.array(value)
        if value.ndim < 2:
            raise ValueError(
                f'spectrogram `s` should have at least 2 dimensions, '
                f'but number of dimensions was {value.ndim}'
            )
        self._s = value

    @property
    def t(self) -> np.array:
        """vector of times at the center of each time bin"""
        return self._t

    @t.setter
    def t(self, value: np.array):
        value = np.array(value)
        if not validators.is_1d_or_row_or_column(value):
            raise ValueError(
                f'time bins vector `t` should be a 1-dimensional array, '
                f'but number of dimensions was {value.ndim}'
            )
        self._t = value

    @property
    def f(self) -> np.array:
        """vector of frequencies at the center of each frequency bin"""
        return self._f

    @f.setter
    def f(self, value: np.array):
        value = np.array(value)
        if not validators.is_1d_or_row_or_column(value):
            raise ValueError(
                f'frequency bins vector `f` should be a 1-dimensional array, '
                f'but number of dimensions was {value.ndim}'
            )
        self._f = value

    @property
    def spect_path(self) -> pathlib.Path:
        """path to a file containing saved arrays"""
        return self._spect_path

    @spect_path.setter
    def spect_path(self, path: Union[str, pathlib.Path]):
        if path is not None:
            path = pathlib.Path(path)
        self._spect_path = path

    @property
    def audio_path(self) -> pathlib.Path:
        """path to audio file from which spectrogram was generated"""
        return self._audio_path

    @audio_path.setter
    def audio_path(self, path: Optional[pathlib.Path] = None):
        if path is not None:
            path = pathlib.Path(path)
        self._audio_path = path

    @classmethod
    def from_file(cls, spect_path: Union[str, pathlib.Path], format: str = None, key_map: dict = None):
        spect_path = pathlib.Path(spect_path)
        if not spect_path.exists():
            raise FileNotFoundError(
                f"did not find spectrogram file at path specified: {spect_path}"
            )

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

        spect_file_dict = SPECT_FILE_LOADING_FUNCTIONS[format](spect_path)

        try:
            s = np.array(spect_file_dict[key_map['s']])
        except KeyError as e:
            raise KeyError(
                f"Did not find spectrogram using key '{key_map['s']}' "
                f"in file loaded from spect_path: {spect_path}"
            ) from e

        try:
            t = np.array(spect_file_dict[key_map['t']])
        except KeyError as e:
            raise KeyError(
                f"Did not find time bins vector using key '{key_map['t']}' "
                f"in file loaded from spect_path: {spect_path}"
            ) from e

        try:
            f = np.array(spect_file_dict[key_map['f']])
        except KeyError as e:
            raise KeyError(
                f"Did not find frequency bins vector using key '{key_map['f']}' "
                f"in file loaded from spect_path: {spect_path}"
            ) from e

        if not validators.is_1d_or_row_or_column(f):
            raise ValueError(
                f'frequency bins vector `f` should be a 1-dimensional array, '
                f'but number of dimensions was {t.ndim}'
            )

        try:
            audio_path = spect_file_dict[key_map['audio_path']]
            if isinstance(audio_path, np.ndarray):
                audio_path = np.squeeze(audio_path).item()
            audio_path = pathlib.Path(audio_path)
        except KeyError:  # for audio_path only we default to None if it is not found
            audio_path = None

        return cls(s, f, t, spect_path, audio_path)

    @classmethod
    def from_mat(cls, mat_path: [str, pathlib.Path], key_map: dict = None):
        """load a spectrogram from a Matlab .mat file"""
        return cls.from_file(mat_path, 'mat', key_map)

    @classmethod
    def from_npz(cls, npz_path: [str, pathlib.Path], key_map: dict = None):
        """load a spectrogram from a Numpy .npz file"""
        return cls.from_file(npz_path, 'npz', key_map)

    def to_file(self, spect_path: [str, pathlib.Path], key_map: dict = None, exist_ok: bool = False):
        """save a spectrogram to a Numpy .npz file"""
        spect_path = pathlib.Path(spect_path)
        if spect_path.exists() and not exist_ok:
            raise FileExistsError(
                f'spect_path already exists:\n{spect_path}.\nTo overwrite, set exist_ok argument to True.'
            )

        if key_map:
            if not are_key_map_keys_valid(key_map):
                raise ValueError(
                    f'invalid key map: {key_map}.\n Valid keys for mapping are: {VALID_KEYS}'
                )
            key_map = add_default_key_mappings(key_map)
        else:
            key_map = default_key_map()

        np.savez(
            spect_path, s=self._s, f=self._f, t=self._t
        )
