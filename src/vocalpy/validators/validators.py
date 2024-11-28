"""Validation functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["have_same_dtype", "is_1d_ndarray", "is_valid_boundaries_array"]


def is_1d_ndarray(y: npt.NDArray, name: str | None = None) -> bool:
    """Validates that ``y`` is a
    1-dimensional :class:`numpy.ndarray`.

    Parameters
    ----------
    y: numpy.ndarray
        Array to be validated.
    name: str, optional
        Name of array in calling function.
        Used in any error message if supplied.

    Returns
    -------
    is_1d_ndarray: bool
        True if ``y`` is valid.

    Examples
    --------
    >>> y = np.array([0, 1, 2])
    >>> vocalpy.validators.is_1d_ndarray(y)
    True
    """
    if name:
        name += " "
    else:
        name = ""

    if not isinstance(y, np.ndarray):
        raise TypeError(
            f"Input {name}should be a numpy array, but type was: {type(y)}"
        )

    if not len(y.shape) == 1:
        raise ValueError(
            f"Input {name}should be a 1-dimensional numpy array, "
            f"but number of dimensions was: {y.ndim}"
        )
    return True


def is_valid_boundaries_array(y: npt.NDArray, name: str | None = None) -> bool:
    """Validates that ``y`` is a valid array of boundaries
    found with a segmentation algorithm,
    e.g., onsets or offsets of segments returned by :func:`vocalpy.segment.meansquared`.

    To be a valid array of boundaries, ``y`` must meet the following conditions:
    - Be a one dimensional numpy array, as validated with
      :func:`vocalpy.validators.is_1d_ndarray`.
    - Have a dtype that is a float or int, e.g. ``np.float64`` or ``np.int32``.
    - Have values that are all non-negative, i.e. ``np.all(y >= 0.0)``
    - Have values that are strictly increasing, i.e. ``np.all(y[1:] > y[:-1])``

    An empty array or an array with a single value are also considered valid,
    as long as the dtype is correct and any value is non-negative.

    Parameters
    ----------
    y: numpy.ndarray
        Array to be validated.
    name: str, optional
        Name of array in calling function.
        Used in any error message if supplied.

    Returns
    -------
    is_valid_boundaries_array: bool
        True if ``y`` is valid.

    Examples
    --------
    >>> vocalpy.validators.is_valid_boundaries_array(np.array([1, 2, 3], dtype=np.int16))
    True
    >>> vocalpy.validators.is_valid_boundaries_array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    True
    >>> vocalpy.validators.is_valid_boundaries_array(np.array([], dtype=np.float32))
    True
    >>> vocalpy.validators.is_valid_boundaries_array(np.array([1.0], dtype=np.float32))
    True
    """
    is_1d_ndarray(y, name)

    if name:
        name += " "
    else:
        name = ""

    if not (
        issubclass(y.dtype.type, np.floating)
        or issubclass(y.dtype.type, np.integer)
    ):
        raise TypeError(
            f"Dtype of boundaries array {name}must be either float or int but was: {y.dtype}"
        )

    if not np.all(y >= 0.0):
        raise ValueError(
            f"Values of boundaries array {name}must all be non-negative"
        )

    if y.size <= 1:
        # It's a valid boundary array but there's no boundaries or just one boundary,
        # so we don't check that values are strictly increasing
        return True

    if not np.all(y[1:] > y[:-1]):
        raise ValueError(
            f"Values of boundaries array {name}must be strictly increasing"
        )

    return True


def have_same_dtype(
    arr1: npt.NDArray,
    arr2: npt.NDArray,
    name1: str | None = None,
    name2: str | None = None,
) -> bool:
    """Validates that two arrays, ``arr1`` and ``arr2``, have the same :class:`~numpy.dtype`.

    Parameters
    ----------
    arr1 : numpy.ndarray
        First array to be validated.
    arr2 : numpy.ndarray
        Second array to be validated.
    name1 : str, optional
        Name of first array in calling function.
        Used in any error message if both ``name1`` and ``name2`` are supplied.
    name2 : str, optional
        Name of second array in calling function.
        Used in any error message if both ``name1`` and ``name2`` are supplied.

    Returns
    -------
    have_same_dtype : bool
        True if ``arr1`` and ``arr2`` have the same :class:`~numpy.dtype`.
    """
    if not arr1.dtype == arr2.dtype:
        if name1 and name2:
            names = f"{name1} and {name2} "
        else:
            names = ""

        raise ValueError(
            f"Two arrays {names}must have the same dtype, but dtypes were: {arr1.dtype} and {arr2.dtype}"
        )

    return True
