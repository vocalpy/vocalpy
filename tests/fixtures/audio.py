"""fixtures relating to audio files"""
import pytest

from .test_data import SOURCE_TEST_DATA_ROOT
from .paths import DATA_ROOTS_WITH_SUBDIRS

AUDIO_DIR_CBIN = SOURCE_TEST_DATA_ROOT / 'audio_cbin_annot_notmat' / 'gy6or6' / '032312'


@pytest.fixture
def audio_dir_cbin():
    return AUDIO_DIR_CBIN

AUDIO_LIST_CBIN = sorted(AUDIO_DIR_CBIN.glob('*.cbin'))


@pytest.fixture
def audio_list_cbin():
    return AUDIO_LIST_CBIN


@pytest.fixture(params=AUDIO_LIST_CBIN)
def a_cbin_path(request):
    return request.param


# rec files are not audio, but
# they have the sampling rate for cbin files
# so I'm putting the fixture here
RECFILE_LIST = sorted(AUDIO_DIR_CBIN.glob('*.rec'))


@pytest.fixture(params=RECFILE_LIST)
def a_rec_path(request):
    return request.param


BIRDSONGREC_WAV_DIR =  SOURCE_TEST_DATA_ROOT / 'audio_wav_annot_birdsongrec' / 'Bird0' / 'Wave'


@pytest.fixture
def birdsongrec_wav_dir(source_test_data_root):
    return BIRDSONGREC_WAV_DIR


BIRDSONGREC_WAV_LIST = sorted(BIRDSONGREC_WAV_DIR.glob('*.wav'))


@pytest.fixture
def birdsongrec_wav_list():
    return BIRDSONGREC_WAV_LIST


@pytest.fixture(params=BIRDSONGREC_WAV_LIST)
def a_birdsongrec_wav_path(request):
    return request.param


ZEBRA_FINCH_WAV_DIR = SOURCE_TEST_DATA_ROOT / 'zebra-finch-wav'
ALL_ZEBRA_FINCH_WAVS = sorted(ZEBRA_FINCH_WAV_DIR.glob('*.wav'))


@pytest.fixture(params=ALL_ZEBRA_FINCH_WAVS)
def a_zebra_finch_wav(request):
    """Parametrized fixture that returns

    Used for testing :func:`vocalpy.spectral.sat`
    and :func:`vocalpy.feature.sat`"""
    return request.param


MULTICHANNEL_FLY_DIR = SOURCE_TEST_DATA_ROOT / 'fly-multichannel'
MULTICHANNEL_FLY_WAV = MULTICHANNEL_FLY_DIR / '160420_1746_manual-clip.wav'


@pytest.fixture
def multichannel_fly_wav_path():
    return MULTICHANNEL_FLY_WAV


@pytest.fixture
def multichannel_fly_wav_sound():
    import vocalpy
    return vocalpy.Sound.read(MULTICHANNEL_FLY_WAV)



N_BIRDSONGREC_WAVS = 5
ALL_WAV_PATHS = (
    BIRDSONGREC_WAV_LIST[:N_BIRDSONGREC_WAVS] +
    ALL_ZEBRA_FINCH_WAVS +
    # next line: one-element list so we can concatenate with `+`
    [MULTICHANNEL_FLY_WAV]
)


@pytest.fixture(params=ALL_WAV_PATHS)
def a_wav_path(request):
    return request.param


JOURJINE_ETAL_2023_WAV_DIR = SOURCE_TEST_DATA_ROOT / 'jourjine-et-al-2023/developmentLL'
JOURJINE_ETAL_2023_WAV_LIST = sorted(JOURJINE_ETAL_2023_WAV_DIR.glob('*.wav'))

# only use a few wav files from cbins, to speed up tests
N_CBINS = 5
ALL_AUDIO_PATHS = (
    BIRDSONGREC_WAV_LIST[:N_BIRDSONGREC_WAVS] +
    AUDIO_LIST_CBIN[:N_CBINS] +
    ALL_ZEBRA_FINCH_WAVS +
    [MULTICHANNEL_FLY_WAV]
)


@pytest.fixture(params=ALL_AUDIO_PATHS)
def an_audio_path(request):
    """Parametrized fixture that returns one audio path
    from all the audio paths in :mod:`tests.fixtures.audio`.

    Used for testing .e.g. :class:`vocalpy.dataset.AudioFile`.
    """
    return request.param

# -- these are used in tests/test_paths
# Leave them as constants; the constants get used in a `pytest.mark.parametrize` in those unit tests
AUDIO_DIR_WAV_WITH_SUBDIRS = DATA_ROOTS_WITH_SUBDIRS / 'wav'

AUDIO_LIST_WAV_WITH_SUBDIRS = sorted(AUDIO_DIR_WAV_WITH_SUBDIRS.glob('**/*wav'))

AUDIO_DIR_CBIN_WITH_SUBDIRS = DATA_ROOTS_WITH_SUBDIRS / 'cbin'

AUDIO_LIST_CBIN_WITH_SUBDIRS = sorted(AUDIO_DIR_CBIN_WITH_SUBDIRS.glob('**/*cbin'))
