import numpy as np
import pytest

import vocalpy as voc

from ..fixtures.audio import JOURJINE_ETAL_2021_WAV_LIST
from ..fixtures.segments import AVA_SEGMENT_TXT_LIST


WAV_PATH_SEG_TXT_PATH_TUPLES = zip(
    JOURJINE_ETAL_2021_WAV_LIST,
    AVA_SEGMENT_TXT_LIST
)


@pytest.fixture(params=WAV_PATH_SEG_TXT_PATH_TUPLES)
def wav_path_seg_txt_path_tuple(request):
    return request.param


def test_segment_replicates(wav_path_seg_txt_path_tuple):
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


@pytest.fixture(params=JOURJINE_ETAL_2021_WAV_LIST)
def jourjine_et_al_wav_2023_path(request):
    return request.param


def test_segment_min_isi_dur(jourjine_et_al_wav_2023_path):
    """Test that :func:`vocalpy.segment.ava.segment` parameter `min_isi_dur` works as expected"""
    sound = voc.Sound.read(jourjine_et_al_wav_2023_path)
    params = {**voc.segment.ava.JOURJINEETAL2023}
    onsets_isi, offsets_isi = voc.segment.ava.segment(sound, **params)
    params_wout_isi = {k: v for k, v in params.items() if k != 'min_isi_dur'}
    onsets, offsets = voc.segment.ava.segment(sound, **params_wout_isi)

    isi_durs_with_min_isi = onsets_isi[1:] - offsets_isi[:-1]
    assert not np.any(isi_durs_with_min_isi < params['min_isi_dur'])

    isi_durs_without_min_isi = onsets[1:] - offsets[:-1]
    if np.any(isi_durs_without_min_isi < params['min_isi_dur']):
        assert onsets_isi.shape[0] < onsets.shape[0]
        assert offsets_isi.shape[0] < offsets.shape[0]
        assert onsets_isi.shape[0] == offsets_isi.shape[0]
        assert np.array_equal(
            isi_durs_without_min_isi[isi_durs_without_min_isi > params['min_isi_dur']],
            isi_durs_with_min_isi
        )
    else:
        assert np.array_equal(
            isi_durs_without_min_isi, isi_durs_with_min_isi
        )
