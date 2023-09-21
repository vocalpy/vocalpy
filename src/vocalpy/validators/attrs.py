"""Validation functions.

Includes validators used by attrs classes,
as well as more general validation functions
that may be used e.g. as pre-conditions for a function.
"""
import numpy as np


def is_1d_ndarray(instance, attribute, value):
    """An :mod:`attrs` validator that
    validates that the value for an attribute is a
    1-dimensional :class:`numpy.ndarray`.
    """
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{attribute} of {instance} should be a numpy array, " f"but type was: {type(value)}")

    if not value.ndim == 1:
        raise ValueError(
            f"{attribute} of {instance} should be a 1-dimensional numpy array, "
            f"but number of dimensions was: {value.ndim}"
        )
