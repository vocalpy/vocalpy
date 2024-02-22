"""fixtures relating to array files containing spectrograms"""
import inspect

import pytest

import vocalpy

from .test_data import GENERATED_TEST_DATA_ROOT, SOURCE_TEST_DATA_ROOT

# ---- source data
SPECT_DIR_MAT = SOURCE_TEST_DATA_ROOT / 'spect_mat_annot_yarden' / 'llb3' / 'spect'


@pytest.fixture
def spect_dir_mat(source_test_data_root):
    return SPECT_DIR_MAT


SPECT_LIST_MAT = sorted(SPECT_DIR_MAT.glob('*.mat'))


@pytest.fixture
def spect_list_mat():
    return SPECT_LIST_MAT


@pytest.fixture(params=SPECT_LIST_MAT)
def a_mat_spect_path(request):
    return request.param


# ---- generated data
SPECTROGRAM_DIR = GENERATED_TEST_DATA_ROOT / 'spectrogram'
SPECT_DIR_NPZ = SPECTROGRAM_DIR / 'spect_npz'

@pytest.fixture
def spect_dir_npz():
    return SPECT_DIR_NPZ


SPECT_LIST_NPZ = sorted(SPECT_DIR_NPZ.glob('*.npz'))


@pytest.fixture
def spect_list_npz():
    return SPECT_LIST_NPZ


@pytest.fixture(params=SPECT_LIST_NPZ)
def an_npz_spect_path(request):
    return request.param


@pytest.fixture
def specific_spect_dir(spect_dir_mat,
                       spect_dir_npz):
    def _specific_spect_dir(spect_format):
        if spect_format == 'mat':
            return spect_dir_mat
        elif spect_format == 'npz':
            return spect_dir_npz
        else:
            raise ValueError(f'invalid spect_format: {spect_format}')

    return _specific_spect_dir


SPECT_LIST_MAT = sorted(SPECT_DIR_MAT.glob('*.mat'))


@pytest.fixture
def spect_list_mat():
    return SPECT_LIST_MAT


@pytest.fixture
def spect_list_npz(spect_dir_npz):
    return sorted(spect_dir_npz.glob('*.spect.npz'))


@pytest.fixture
def specific_spect_list(spect_list_mat,
                        # spect_list_mat_all_labels_in_labelset,
                        # spect_list_mat_labels_not_in_labelset,
                        spect_list_npz,
                        # spect_list_npz_all_labels_in_labelset,
                        # spect_list_npz_labels_not_in_labelset
                        ):

    def _specific_spect_list(spect_format, qualifier=None):
        MAP = {
            'mat': {
                None: spect_list_mat,
                # 'all_labels_in_labelset': spect_list_mat_all_labels_in_labelset,
                # 'labels_not_in_labelset': spect_list_mat_labels_not_in_labelset
            },
            'npz': {
                None: spect_list_npz,
                # 'all_labels_in_labelset': spect_list_npz_all_labels_in_labelset,
                # 'labels_not_in_labelset': spect_list_npz_labels_not_in_labelset
            }

        }
        return MAP[spect_format][qualifier]

    return _specific_spect_list


@pytest.fixture
def default_spect_params():
    spect_sig = inspect.signature(vocalpy.spectrogram)
    default_n_fft, default_hop_length = spect_sig.parameters['n_fft'], spect_sig.parameters['hop_length']
    return vocalpy.SpectrogramParameters(n_fft=default_n_fft, hop_length=default_hop_length)
