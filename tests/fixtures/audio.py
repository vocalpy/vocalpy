"""fixtures relating to audio files"""
import pytest

from .test_data import DATA_ROOTS_WITH_SUBDIRS, SOURCE_TEST_DATA_ROOT


@pytest.fixture
def default_spect_params():
    return dict(fft_size=512,
                step_size=64,
                freq_cutoffs=(500, 10000),
                thresh=6.25,
                transform_type='log_spect',
                freqbins_key='f',
                timebins_key='t',
                spect_key='s'
                )


AUDIO_DIR_CBIN = SOURCE_TEST_DATA_ROOT / 'audio_cbin_annot_notmat' / 'gy6or6' / '032312'


@pytest.fixture
def audio_dir_cbin(source_test_data_root):
    return AUDIO_DIR_CBIN


AUDIO_LIST_CBIN = sorted(AUDIO_DIR_CBIN.glob('*.cbin'))


@pytest.fixture
def audio_list_cbin(audio_dir_cbin):
    return AUDIO_LIST_CBIN


@pytest.fixture(params=AUDIO_LIST_CBIN)
def a_cbin_path(request):
    return request.param


@pytest.fixture
def audio_list_cbin_all_labels_in_labelset(audio_list_cbin,
                                           annot_list_notmat,
                                           labelset_notmat):
    """list of .cbin audio files where all labels in associated annotation **are** in labelset"""
    labelset_notmat = set(labelset_notmat)
    audio_list_labels_in_labelset = []
    for audio_path in audio_list_cbin:
        audio_fname = audio_path.name
        annot = [annot for annot in annot_list_notmat if annot.audio_path.name == audio_fname]
        assert len(annot) == 1
        annot = annot[0]
        if set(annot.seq.labels).issubset(labelset_notmat):
            audio_list_labels_in_labelset.append(audio_path)

    return audio_list_labels_in_labelset


@pytest.fixture
def audio_list_cbin_labels_not_in_labelset(audio_list_cbin,
                                           annot_list_notmat,
                                           labelset_notmat):
    """list of .cbin audio files where some labels in associated annotation are **not** in labelset"""
    labelset_notmat = set(labelset_notmat)
    audio_list_labels_in_labelset = []
    for audio_path in audio_list_cbin:
        audio_fname = audio_path.name
        annot = [annot for annot in annot_list_notmat if annot.audio_path.name == audio_fname]
        assert len(annot) == 1
        annot = annot[0]
        if not set(annot.seq.labels).issubset(labelset_notmat):
            audio_list_labels_in_labelset.append(audio_path)

    return audio_list_labels_in_labelset


AUDIO_DIR_WAV =  SOURCE_TEST_DATA_ROOT / 'audio_wav_annot_birdsongrec' / 'Bird0' / 'Wave'


# TODO: add .WAV from TIMIT
@pytest.fixture
def audio_dir_wav(source_test_data_root):
    return AUDIO_DIR_WAV


AUDIO_LIST_WAV = sorted(AUDIO_DIR_WAV.glob('*.wav'))


@pytest.fixture
def audio_list_wav(audio_dir_wav):
    return AUDIO_LIST_WAV


@pytest.fixture(params=AUDIO_LIST_WAV)
def a_wav_path(request):
    return request.param


@pytest.fixture
def specific_audio_dir(audio_dir_cbin,
                       audio_dir_wav):
    """factory fixture, returns a function that
    returns a fixture containing a list of Annotation objects,
    given a specified annotation format

    so that unit tests can be parameterized with annotation format names
    """
    FORMAT_AUDIO_DIR_FIXTURE_MAP = {
        'cbin': audio_dir_cbin,
        'wav': audio_dir_wav,
    }

    def _specific_audio_dir(format):
        return FORMAT_AUDIO_DIR_FIXTURE_MAP[format]

    return _specific_audio_dir


@pytest.fixture
def specific_audio_list(audio_list_cbin,
                        audio_list_wav):
    """factory fixture, returns a function that
    returns a fixture containing a list of Annotation objects,
    given a specified annotation format

    so that unit tests can be parameterized with annotation format names
    """
    FORMAT_AUDIO_LIST_FIXTURE_MAP = {
        'cbin': audio_list_cbin,
        'wav': audio_list_wav,
    }

    def _specific_audio_list(format):
        return FORMAT_AUDIO_LIST_FIXTURE_MAP[format]

    return _specific_audio_list


AUDIO_DIR_WAV_WITH_SUBDIRS = DATA_ROOTS_WITH_SUBDIRS / 'wav'

AUDIO_LIST_WAV_WITH_SUBDIRS = sorted(AUDIO_DIR_WAV_WITH_SUBDIRS.glob('**/*wav'))

AUDIO_DIR_CBIN_WITH_SUBDIRS = DATA_ROOTS_WITH_SUBDIRS / 'cbin'

AUDIO_LIST_CBIN_WITH_SUBDIRS = sorted(AUDIO_DIR_CBIN_WITH_SUBDIRS.glob('**/*cbin'))


ALL_AUDIO_PATHS = (
    AUDIO_LIST_WAV +
    AUDIO_LIST_CBIN
)

@pytest.fixture(params=ALL_AUDIO_PATHS)
def an_audio_path(request):
    """Parametrized fixture that returns one audio path
    from all the audio paths in :mod:`tests.fixtures.audio`.

    Used for testing .e.g. :class:`vocalpy.dataset.AudioFile`.
    """
    return request.param
