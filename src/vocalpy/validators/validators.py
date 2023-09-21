"""Validation functions."""
import numpy as np
import numpy.typing as npt


def is_1d_ndarray(y: npt.NDArray, name:str=None) -> None:
    """Validates that ``y`` is a
    1-dimensional :class:`numpy.ndarray`.
    """
    if name:
        name += " "
    if not isinstance(y, np.ndarray):
        raise TypeError(f"Input {name}should be a numpy array, but type was: {type(y)}")

    if not len(y.shape) == 1:
        raise ValueError(
            f"Input {name}should be a 1-dimensional numpy array, "
            f"but number of dimensions was: {y.ndim}"
        )


def is_valid_boundaries_array(y: npt.NDArray, name:str=None):
    is_1d_ndarray(y, name)

    if name:
        name += " "

    # TODO: dtype should be float or int

    if not np.all(y >= 0.0):
        raise ValueError(
            f"Values of boundaries array {name}must all be non-negative"
        )

    if not np.all(y[1:] > y[:-1]):
        raise ValueError(
            f"Values of boundaries array {name}must be strictly increasing"
        )