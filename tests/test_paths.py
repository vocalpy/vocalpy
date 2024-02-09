import pytest

import vocalpy as voc

from .fixtures.audio import (
    AUDIO_DIR_CBIN,
    AUDIO_DIR_CBIN_WITH_SUBDIRS,
    BIRDSONGREC_WAV_DIR,
    AUDIO_DIR_WAV_WITH_SUBDIRS,
    AUDIO_LIST_CBIN,
    AUDIO_LIST_CBIN_WITH_SUBDIRS,
    BIRDSONGREC_WAV_LIST,
    AUDIO_LIST_WAV_WITH_SUBDIRS,
)
from .fixtures.spect import SPECT_DIR_MAT, SPECT_LIST_MAT


@pytest.mark.parametrize(
    "dir, ext, recurse, expected_paths",
    [
        (BIRDSONGREC_WAV_DIR, "wav", None, BIRDSONGREC_WAV_LIST),
        # directly test passing in recurse arg
        (BIRDSONGREC_WAV_DIR, "wav", False, BIRDSONGREC_WAV_LIST),
        # should be the same result if we do recurse, since there's no sub-dir
        (BIRDSONGREC_WAV_DIR, "wav", True, BIRDSONGREC_WAV_LIST),
        # now test with period in extension
        (BIRDSONGREC_WAV_DIR, ".wav", None, BIRDSONGREC_WAV_LIST),
        (BIRDSONGREC_WAV_DIR, ".wav", False, BIRDSONGREC_WAV_LIST),
        (BIRDSONGREC_WAV_DIR, ".wav", True, BIRDSONGREC_WAV_LIST),
        (AUDIO_DIR_CBIN, "cbin", None, AUDIO_LIST_CBIN),
        (AUDIO_DIR_CBIN, "cbin", False, AUDIO_LIST_CBIN),
        (AUDIO_DIR_CBIN, "cbin", True, AUDIO_LIST_CBIN),
        (AUDIO_DIR_CBIN, ".cbin", None, AUDIO_LIST_CBIN),
        (AUDIO_DIR_CBIN, ".cbin", False, AUDIO_LIST_CBIN),
        (AUDIO_DIR_CBIN, ".cbin", True, AUDIO_LIST_CBIN),
        (AUDIO_DIR_WAV_WITH_SUBDIRS, "wav", True, AUDIO_LIST_WAV_WITH_SUBDIRS),
        (AUDIO_DIR_CBIN_WITH_SUBDIRS, "cbin", True, AUDIO_LIST_CBIN_WITH_SUBDIRS),
        # if we don't recurse, we should get an empty list since there's no audio files in top-level root dir
        (AUDIO_DIR_WAV_WITH_SUBDIRS, "wav", False, []),
        (AUDIO_DIR_CBIN_WITH_SUBDIRS, "cbin", False, []),
        # default for recurse is False so we should get same answer
        (AUDIO_DIR_WAV_WITH_SUBDIRS, "wav", None, []),
        (AUDIO_DIR_CBIN_WITH_SUBDIRS, "cbin", None, []),
        (AUDIO_DIR_WAV_WITH_SUBDIRS, ".wav", True, AUDIO_LIST_WAV_WITH_SUBDIRS),
        (AUDIO_DIR_CBIN_WITH_SUBDIRS, ".cbin", True, AUDIO_LIST_CBIN_WITH_SUBDIRS),
        (AUDIO_DIR_WAV_WITH_SUBDIRS, ".wav", False, []),
        (AUDIO_DIR_CBIN_WITH_SUBDIRS, ".cbin", False, []),
        (AUDIO_DIR_WAV_WITH_SUBDIRS, ".wav", None, []),
        (AUDIO_DIR_CBIN_WITH_SUBDIRS, ".cbin", None, []),
        (SPECT_DIR_MAT, "mat", None, SPECT_LIST_MAT),
        (SPECT_DIR_MAT, "mat", True, SPECT_LIST_MAT),
        (SPECT_DIR_MAT, "mat", False, SPECT_LIST_MAT),
        (SPECT_DIR_MAT, ".mat", None, SPECT_LIST_MAT),
        (SPECT_DIR_MAT, ".mat", True, SPECT_LIST_MAT),
        (SPECT_DIR_MAT, ".mat", False, SPECT_LIST_MAT),
    ],
)
def test_from_dir(dir, ext, recurse, expected_paths):
    if recurse is None:  # test default `recurse` argument
        paths = voc.paths.from_dir(dir, ext)
    else:
        paths = voc.paths.from_dir(dir, ext, recurse)

    assert paths == expected_paths
