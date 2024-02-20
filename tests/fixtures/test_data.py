from pathlib import Path

import pytest

HERE = Path(__file__).parent


TEST_DATA_ROOT = HERE.joinpath('..', 'data-for-tests')

@pytest.fixture
def test_data_root():
    """Path that points to root of test_data directory"""
    return TEST_DATA_ROOT


SOURCE_TEST_DATA_ROOT = TEST_DATA_ROOT / 'source'


@pytest.fixture
def source_test_data_root(test_data_root):
    """'source' test data, i.e., files **not** created by ``vocalpy``,
    that is, the input data used when vak does create files (csv files, logs,
    neural network checkpoints, etc.)
    """
    return SOURCE_TEST_DATA_ROOT


GENERATED_TEST_DATA_ROOT = TEST_DATA_ROOT / 'generated'


@pytest.fixture
def generated_test_data_root(test_data_root):
    """'generated' test data, i.e., files **not** created by ``vak``, that is,
    the input data used when vak does create files (csv files, logs,
    neural network checkpoints, etc.)
    """
    return GENERATED_TEST_DATA_ROOT


DATA_ROOTS_WITH_SUBDIRS = GENERATED_TEST_DATA_ROOT / 'root_with_subdirs_to_test_paths'
