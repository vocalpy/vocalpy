"""Validators used by attrs classes.

.. autosummary::
   :toctree: generated
"""

from . import validators


def is_1d_ndarray(instance, attribute, value):
    """An :mod:`attrs` validator that
    validates that the value for an attribute is a
    1-dimensional :class:`numpy.ndarray`.
    """
    try:
        validators.is_1d_ndarray(value)
    except TypeError as e:
        raise TypeError(
            f"{attribute} of {instance} should be a numpy array, "
            f"but type was: {type(value)}"
        ) from e
    except ValueError as e:
        raise ValueError(
            f"{attribute} of {instance} should be a 1-dimensional numpy array, "
            f"but number of dimensions was: {value.ndim}"
        ) from e
