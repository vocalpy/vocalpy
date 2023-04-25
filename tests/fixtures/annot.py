"""fixtures relating to annotation files"""
import crowsetta
import pytest

from .test_data import SOURCE_TEST_DATA_ROOT

ANNOT_FILE_YARDEN = SOURCE_TEST_DATA_ROOT.joinpath('spect_mat_annot_yarden', 'llb3', 'llb3_annot_subset.mat')


@pytest.fixture
def annot_file_yarden():
    return ANNOT_FILE_YARDEN


@pytest.fixture
def annot_list_yarden(annot_file_yarden):
    scribe = crowsetta.Transcriber(format='yarden')
    annot_list = scribe.from_file(annot_file_yarden)
    return annot_list


@pytest.fixture
def labelset_yarden():
    """labelset as it would be loaded from a toml file

    don't return a set because we need to use this to test functions that convert it to a set
    """
    return [str(an_int) for an_int in [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19]]


ANNOT_DIR_NOTMAT = SOURCE_TEST_DATA_ROOT.joinpath('audio_cbin_annot_notmat', 'gy6or6', '032312')


@pytest.fixture
def annot_dir_notmat():
    return ANNOT_DIR_NOTMAT


ANNOT_PATHS_LIST_NOTMAT = sorted(ANNOT_DIR_NOTMAT.glob('*.not.mat'))


@pytest.fixture
def annot_paths_list_notmat():
    return ANNOT_PATHS_LIST_NOTMAT


@pytest.fixture
def annot_list_notmat():
    scribe = crowsetta.Transcriber(format='notmat')
    annot_list = scribe.from_file(ANNOT_PATHS_LIST_NOTMAT)
    return annot_list


ANNOT_FILE_KOUMURA = SOURCE_TEST_DATA_ROOT.joinpath('audio_wav_annot_koumura', 'Bird0', 'Annotation.xml')


@pytest.fixture
def annot_file_koumura(source_test_data_root):
    return ANNOT_FILE_KOUMURA


@pytest.fixture
def annot_list_koumura(annot_file_koumura):
    scribe = crowsetta.Transcriber(format='koumura')
    annot_list = scribe.from_file(annot_file_koumura)
    return annot_list


@pytest.fixture
def specific_annot_list(annot_list_notmat,
                        annot_list_yarden,
                        annot_list_koumura):
    """factory fixture, returns a function that
    returns a fixture containing a list of Annotation objects,
    given a specified annotation format

    so that unit tests can be parameterized with annotation format names
    """
    FORMAT_ANNOT_LIST_FIXTURE_MAP = {
        'notmat': annot_list_notmat,
        'yarden': annot_list_yarden,
        'koumura': annot_list_koumura,
    }

    def _annot_list_factory(format):
        return FORMAT_ANNOT_LIST_FIXTURE_MAP[format]

    return _annot_list_factory


@pytest.fixture
def specific_labelset(labelset_yarden,
                      labelset_notmat):

    def _specific_labelset(annot_format):
        if annot_format == 'yarden':
            return labelset_yarden
        elif annot_format == 'notmat':
            return labelset_notmat
        else:
            raise ValueError(f'invalid annot_format: {annot_format}')

    return _specific_labelset
