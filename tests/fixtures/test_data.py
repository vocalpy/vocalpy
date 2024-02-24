from pathlib import Path

import pytest

HERE = Path(__file__).parent


PROJ_ROOT = (HERE / '..' / '..').resolve()
# next line, use `resolve` + `relative_to` so all paths we compute with this are relative to project root,
# which matters e.g. when we have fixtures that use paths to files, we don't want them to be
# absolute for a specific system (such as, my computer)
TEST_DATA_ROOT = HERE.joinpath('..', 'data-for-tests').resolve().relative_to(PROJ_ROOT)


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


