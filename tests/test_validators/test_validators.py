import numpy as np
import pytest

import vocalpy.validators


@pytest.mark.parametrize(
    'y',
    [
        np.array([1, 2, 3]),
        np.array([1.0, 2.0, 3.0]),
    ]
)
def test_is_1d_ndarray(y):
    assert vocalpy.validators.is_1d_ndarray(y) is True


@pytest.mark.parametrize(
    'y',
    [
        [1, 2, 3],
        (1, 2, 3),
        [1.0, 2.0, 3.0],
        (1, 2, 3),
    ]
)
def test_is_1d_ndarray_raises_type_error(y):
    with pytest.raises(TypeError):
        vocalpy.validators.is_1d_ndarray(y)


@pytest.mark.parametrize(
    'y',
    [
        # 0d
        np.array(1),
        np.array(1.0),
        # 2d
        np.array([[1, 2, 3]]),
        np.array([[1.0, 2.0, 3.0]]),
        # 3d
        np.array([[[1, 2, 3]]]),
        np.array([[[1.0, 2.0, 3.0]]]),
    ]
)
def test_is_1d_ndarray_raises_value_error(y):
    with pytest.raises(ValueError):
        vocalpy.validators.is_1d_ndarray(y)


@pytest.mark.parametrize(
    'y',
    [
        np.array([1, 2, 3]),
        np.array([1.0, 2.0, 3.0]),
        # ---- edge cases
        # empty arrays are valid boundary arrays,
        # e.g. when we segment but don't find any boundaries
        np.array([], dtype=np.float32),
        np.array([], dtype=np.int16),
        # a single boundary is still a valid boundary array
        # e.g for segmenting algorithms that threshold a distance measure
        np.array([1]),
        np.array([1.0]),
    ]
)
def test_is_valid_boundaries_array(y):
    assert vocalpy.validators.is_valid_boundaries_array(y) is True


@pytest.mark.parametrize(
    'y',
    [
        # list of ints
        [1, 2, 3],
        # tuple of ints
        (1, 2, 3),
        # list of float
        [1.0, 2.0, 3.0],
        # tuple of float
        (1, 2, 3),
        # invalid dtype
        np.array(list('abcde'))
    ]
)
def test_is_valid_boundaries_array_raises_type_error(y):
    with pytest.raises(TypeError):
        vocalpy.validators.is_valid_boundaries_array(y)


@pytest.mark.parametrize(
    'y',
    [
        # 0d
        np.array(1),
        np.array(1.0),
        # 2d
        np.array([[1, 2, 3]]),
        np.array([[1.0, 2.0, 3.0]]),
        # 3d
        np.array([[[1, 2, 3]]]),
        np.array([[[1.0, 2.0, 3.0]]]),
        # has negative values
        np.array([[[-1, 0, 2, 3]]]),
        np.array([[[-1.0, 0.0, 1.0, 2.0, 3.0]]]),
        # is not monotonically increasing
        np.array([[[1, 2, 3]]])[::-1],
        np.array([[[1.0, 2.0, 3.0]]])[::-1],
    ]
)
def test_is_valid_boundaries_array_raises_value_error(y):
    with pytest.raises(ValueError):
        vocalpy.validators.is_valid_boundaries_array(y)


@pytest.mark.parametrize(
    'arr1, arr2',
    [
        (np.array([1, 2, 3]), np.array([4, 5, 6])),
        (np.array([1., 2., 3.]), np.array([4., 5., 6.])),
    ]
)
def test_have_same_dtype(arr1, arr2):
    assert vocalpy.validators.have_same_dtype(arr1, arr2) is True



@pytest.mark.parametrize(
    'arr1, arr2',
    [
        # (int, float)
        (np.array([1, 2, 3]), np.array([4., 5., 6.])),
        # (float, int)
        (np.array([1., 2., 3.]),  np.array([4, 5, 6])),
    ]
)
def test_have_same_dtype_raises_value_error(arr1, arr2):
    with pytest.raises(ValueError):
        vocalpy.validators.have_same_dtype(arr1, arr2)
