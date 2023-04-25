import pytest

from .annot import ANNOT_PATHS_LIST_NOTMAT
from .audio import AUDIO_LIST_CBIN
from .test_data import SOURCE_TEST_DATA_ROOT

# note we convert iterator to a list so it's not consumed by one unit test then left empty
CBIN_NOTMAT_PAIRS = list(zip(AUDIO_LIST_CBIN, ANNOT_PATHS_LIST_NOTMAT))


@pytest.fixture(params=CBIN_NOTMAT_PAIRS)
def a_cbin_notmat_pair(request):
    return request.param


# note this root has two sub-directories, 032312 and 032412
AUDIO_CBIN_ANNOT_NOTMAT_ROOT = SOURCE_TEST_DATA_ROOT / 'audio_cbin_annot_notmat' / 'gy6or6'


@pytest.fixture
def audio_cbin_annot_notmat_root():
    return AUDIO_CBIN_ANNOT_NOTMAT_ROOT


SPECT_MAT_ANNOT_YARDEN_ROOT = SOURCE_TEST_DATA_ROOT / 'spect_mat_annot_yarden' / 'llb3'


@pytest.fixture
def spect_mat_annot_yarden_root():
    return SPECT_MAT_ANNOT_YARDEN_ROOT
