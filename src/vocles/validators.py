"""input validation

adapted in part from scikit-learn under license
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/validation.py
"""
import numpy as np


def is_1d_or_row_or_column(y):
    """validate that a vector is 1-dimensional,
    or is a row or column vector that is safe to ravel
    to make it 1-dimensional

    Parameters
    ----------
    y : array-like
    """
    shape = np.shape(y)
    if len(shape) == 1 or (len(shape) == 2 and any([size == 1 for size in shape])):
        return True
