"""adapted from https://github.com/geopandas/geopandas/blob/master/geopandas/array.py
under BSD license: https://github.com/geopandas/geopandas/blob/master/LICENSE.txt"""
from pathlib import Path
from collections.abc import Iterable
import numbers


import numpy as np
import pandas as pd
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
)


@pd.api.extensions.register_extension_dtype
class PathDtype(ExtensionDtype):
    type = Path
    name = "Path"
    na_value = np.nan

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(
                "'construct_from_string' expects a string, got {}".format(type(string))
            )
        elif string == cls.name:
            return cls()
        else:
            raise TypeError(
                "Cannot construct a '{}' from '{}'".format(cls.__name__, string)
            )

    @classmethod
    def construct_array_type(cls):
        return PathArray


def is_path_dtype(obj):
    dtype = getattr(obj, 'dtype', obj)
    if dtype is None:
        return False
    else:
        return isinstance(dtype, PathDtype)


class PathArray(ExtensionArray):
    """Class wrapping a numpy array of pathlib.Path objects"""

    _dtype = PathDtype()

    def __init__(self, paths):
        if isinstance(paths, list) or isinstance(paths, tuple):
            paths = self.convert_ndarray(paths)

        if not isinstance(paths, np.ndarray):
            raise TypeError(
                "'path' should be array of pathlib.Path objects."
            )
        elif not paths.ndim == 1:
            raise ValueError(
                "'paths' should be a 1-dimensional array of pathlib.Path objects."
            )
        self._paths = paths

    @staticmethod
    def convert_ndarray(paths):
        return np.array(
            [Path(path) for path in paths]
        )

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return self._paths.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return self._paths[idx]

        # validate and convert IntegerArray/BooleanArray
        # to numpy array, pass-through non-array-like indexers
        idx = pd.api.indexers.check_array_indexer(self, idx)
        if isinstance(idx, (Iterable, slice)):
            return PathArray(self._paths[idx])
        else:
            raise TypeError("Index type not supported", idx)

    def __setitem__(self, key, value):
        key = pd.api.indexers.check_array_indexer(self, key)

        if isinstance(value, pd.Series):
            value = value.values  # get underlying array

        if isinstance(value, (list, tuple, np.ndarray)):
            value = self.__init__(value)

        if isinstance(value, PathArray):
            if isinstance(key, numbers.Integral):
                raise ValueError("cannot set a single element with an array")
            self._paths[key] = value._paths

        elif isinstance(value, Path) or pd.isna(value):
            self._paths[key] = value

        else:
            raise TypeError(
                f"Value should be either a Path or None, got {value}"
            )

    @classmethod
    def from_dir(cls, dir, globstr='*'):
        dir_path = Path(dir)
        if not dir_path.is_dir():
            raise NotADirectoryError(
                f'not recognized as a directory: {dir}'
            )
        paths = sorted(dir_path.glob(globstr))
        return cls(paths)
