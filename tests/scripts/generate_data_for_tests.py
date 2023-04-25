"""This is the main script that generates all data found in ``./tests/data-for-tests/generated``."""
import pathlib
import shutil
import sys

import vocalpy as voc

HERE = pathlib.Path(__file__).parent

# we append fixtures to path so we don't have to re-declare all the constants from fixtures in this script
sys.path.append(
    str(HERE / '..')
)


from fixtures.audio import AUDIO_LIST_CBIN, AUDIO_LIST_WAV
from fixtures.test_data import GENERATED_TEST_DATA_ROOT

DATA_ROOTS_WITH_SUBDIRS = GENERATED_TEST_DATA_ROOT / 'root_with_subdirs_to_test_paths'
SPECT_NPZ_DIR = GENERATED_TEST_DATA_ROOT / 'spect_npz'


def mkdirs():
    DATA_ROOTS_WITH_SUBDIRS.mkdir()
    SPECT_NPZ_DIR.mkdir()


def generate_npz_spect_files():
    """This generates .npz files with spectrograms in them,
    so we can test that we can load these files
    """
    for wav_path in AUDIO_LIST_WAV:
        audio = voc.Audio.read(wav_path)
        spect, freqs, times = voc.signal.spectrogram(audio.data, audio.samplerate)
        spect = voc.Spectrogram(data=spect, frequencies=freqs, times=times)
        dst_path = SPECT_NPZ_DIR / f'{wav_path.name}.npz'
        spect.write(path=dst_path)


def generate_data_roots_with_subdirs():
    for ext, path_list in (
        ('wav', AUDIO_LIST_WAV),
        ('cbin', AUDIO_LIST_CBIN),
#        ('mat', SPECT_LIST_MAT),
    ):
        # ---- set-up
        data_root = DATA_ROOTS_WITH_SUBDIRS / f'{ext}'
        data_root.mkdir()

        dst1 = data_root / f'{ext}-first-half'
        dst1.mkdir()
        dst2 = data_root / f'{ext}-second-half'
        dst2.mkdir()

        half_ind = len(path_list) // 2
        paths1 = path_list[:half_ind]
        paths2 = path_list[half_ind:]

        for dst, paths in zip(
            (dst1, dst2),
            (paths1, paths2)
        ):
            for path in paths:
                path_dst = dst / path.name
                shutil.copy(path, path_dst)


def main():
    """Makes directories in `./tests/data-for-tests/generated
    (after those have been removed by running ``nox -s test-data-clean-generated``).
    Then runs helper functions that generate test data from source.
    """
    mkdirs()
    generate_npz_spect_files()
    generate_data_roots_with_subdirs()


main()
