import pytest

from .test_data import SOURCE_TEST_DATA_ROOT


# note this root has two sub-directories, 032312 and 032412
AUDIO_CBIN_ANNOT_NOTMAT_ROOT = SOURCE_TEST_DATA_ROOT / 'audio_cbin_annot_notmat' / 'gy6or6'


@pytest.fixture
def audio_cbin_annot_notmat_root():
    return AUDIO_CBIN_ANNOT_NOTMAT_ROOT
