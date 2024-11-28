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
def all_cbin_paths(request):
    return request.param


# rec files are not audio, but
# they have the sampling rate for cbin files
# so I'm putting the fixture here
RECFILE_LIST = sorted(AUDIO_DIR_CBIN.glob('*.rec'))


@pytest.fixture(params=RECFILE_LIST)
def all_rec_paths(request):
    return request.param


BIRDSONGREC_WAV_DIR =  SOURCE_TEST_DATA_ROOT / 'audio_wav_annot_birdsongrec' / 'Bird0' / 'Wave'


@pytest.fixture
def birdsongrec_wav_dir():
    return BIRDSONGREC_WAV_DIR


BIRDSONGREC_WAV_LIST = sorted(BIRDSONGREC_WAV_DIR.glob('*.wav'))


@pytest.fixture
def birdsongrec_wav_list():
    return BIRDSONGREC_WAV_LIST


@pytest.fixture(params=BIRDSONGREC_WAV_LIST)
def all_birdsongrec_wav_paths(request):
    return request.param


ZEBRA_FINCH_WAV_DIR = SOURCE_TEST_DATA_ROOT / 'zebra-finch-wav'
ALL_ZEBRA_FINCH_WAVS = sorted(ZEBRA_FINCH_WAV_DIR.glob('*.wav'))


@pytest.fixture
def a_zebra_finch_song_sound():
    import vocalpy
    return vocalpy.Sound.read(ALL_ZEBRA_FINCH_WAVS[0])


@pytest.fixture
def a_list_of_zebra_finch_song_sounds():
    import vocalpy
    return [
        vocalpy.Sound.read(path)
        for path in ALL_ZEBRA_FINCH_WAVS
    ]

@pytest.fixture(params=ALL_ZEBRA_FINCH_WAVS)
def all_zebra_finch_wav_paths(request):
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
def all_wav_paths(request):
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
def all_soundfile_paths(request):
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


# ---- additional audio files added for tests
# short clip from single file Nick picked out for workshop; we use this in examples
# but I want to use it in tests too
JOURJINE_ET_AL_GO_WAV_PATH = SOURCE_TEST_DATA_ROOT / 'jourjine-et-al-2023/GO/deermouse-go.wav'

BFSONGREPO_BL26LB16_WAV_PATH = SOURCE_TEST_DATA_ROOT / "bfsongrepo_wav/bl26lb16.wav"

ELIE_THEUNISSEN_2016_WAV_DIR = SOURCE_TEST_DATA_ROOT / 'elie-theunissen-2016'
ELIE_THEUNISSEN_2016_WAV_LIST = sorted(
    ELIE_THEUNISSEN_2016_WAV_DIR.glob("*wav")
)

@pytest.fixture
def single_elie_theunissen_2016_wav_path():
    return ELIE_THEUNISSEN_2016_WAV_LIST[0]


@pytest.fixture
def a_elie_theunissen_2016_sound(single_elie_theunissen_2016_wav_path):
    import vocalpy
    # N.B. we call `to_mono` to speed up feature extraction
    return vocalpy.Sound.read(single_elie_theunissen_2016_wav_path).to_mono()


#@pytest.fixture(params=ELIE_THEUNISSEN_2016_WAV_LIST)
def all_elie_theunissen_2016_wav_paths(request):
    return request.param


@pytest.fixture
def a_list_of_elie_theunissen_2016_sounds(pytestconfig):
    """list of Sound for all wav files from Elie Theunissen 2016 subset in source test data"""
    import vocalpy
    if pytestconfig.getoption("--biosound-fast"):
        return [
            # N.B. we call `to_mono` to speed up feature extraction
            vocalpy.Sound.read(path).to_mono()
            for path in ELIE_THEUNISSEN_2016_WAV_LIST[:2]
        ]
    else:
        return [
            # N.B. we call `to_mono` to speed up feature extraction
            vocalpy.Sound.read(path).to_mono()
            for path in ELIE_THEUNISSEN_2016_WAV_LIST
        ]
