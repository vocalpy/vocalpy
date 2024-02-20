import numpy as np
import pytest

import vocalpy as voc

from ..fixtures.audio import JOURJINE_ETAL_2021_WAV_LIST
from ..fixtures.segments import JOURJINE_ETAL_2023_SEG_TXT_LIST


WAV_PATH_SEG_TXT_PATH_TUPLES = zip(
    JOURJINE_ETAL_2021_WAV_LIST,
    JOURJINE_ETAL_2023_SEG_TXT_LIST
)


@pytest.fixture(params=WAV_PATH_SEG_TXT_PATH_TUPLES)
def wav_path_seg_txt_path_tuple(request):
    return request.param


def test_ava(wav_path_seg_txt_path_tuple):
    """Test that :func:`vocalpy.segment.ava.segment` replicates segmenting results
    obtained with the ``ava`` package."""
    wav_path, seg_txt_path = wav_path_seg_txt_path_tuple

    segs = np.loadtxt(seg_txt_path)
    segs = segs.reshape(-1,2)
    onsets_gt, offsets_gt = segs[:,0], segs[:,1]

    sound = voc.Sound.read(wav_path)
    params = {**voc.segment.ava.JOURJINEETAL2023}
    del params['min_isi_dur']
    onsets, offsets = voc.segment.ava.segment(sound, **params)

    assert isinstance(onsets, np.ndarray)
    assert isinstance(offsets, np.ndarray)
    # we set atol=1e-5 because we expect values to be the same up to roughly 5th decimal place
    # https://stackoverflow.com/questions/65909842/what-is-rtol-for-in-numpys-allclose-function
    np.testing.assert_allclose(onsets, onsets_gt, atol=1e-5, rtol=0)
    np.testing.assert_allclose(offsets, offsets_gt, atol=1e-5, rtol=0)
