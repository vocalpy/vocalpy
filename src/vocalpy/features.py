"""Class that represents features extracted from sound."""

from __future__ import annotations

import pathlib

import xarray as xr


class Features:
    """Class that represents features extracted from sound.

    Attributes
    ----------
    data : xarray.Dataset

    See Also
    --------
    vocalpy.feature
    vocalpy.FeatureExtractor
    """

    def __init__(self, data: xr.Dataset):
        if not isinstance(data, xr.Dataset):
            raise TypeError(
                f"Features `data` must be an instance of `xarray.Dataset`, but type was: {type(data)}"
            )
        self.data = data

    def __repr__(self):
        return f"vocalpy.Features(data=\n{self.data!r}\n)"

    @classmethod
    def read(cls, path: str | pathlib.Path):
        """Read :class:`Features` from a file.

        Parameters
        ----------
        path : str, pathlib.Path
            Path to file with features.

        Returns
        -------
        features : vocalpy.Features
            Instance of :class:`Features` with data
            from ``path`` loaded into it.
        """
        data = xr.open_dataset(path)
        return cls(data=data)

    def write(self, path: str | pathlib.Path):
        """Write :class:`Features` to a file.

        Parameters
        ----------
        path : str, pathlib.Path
            Path to where file with features should be saved.
        """
        self.data.to_netcdf(path)
