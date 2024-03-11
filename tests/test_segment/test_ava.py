import numpy as np
import pytest

import vocalpy
import vocalpy as voc

from ..fixtures.audio import JOURJINE_ETAL_2023_WAV_LIST
from ..fixtures.segments import AVA_SEGMENT_TXT_LIST


WAV_PATH_SEG_TXT_PATH_TUPLES = zip(
    JOURJINE_ETAL_2023_WAV_LIST,
    AVA_SEGMENT_TXT_LIST
)


@pytest.fixture(params=WAV_PATH_SEG_TXT_PATH_TUPLES)
def wav_path_seg_txt_path_tuple(request):
    return request.param


# we filter this warning because we expect that some of the loadtxt files contain no data;
# we want to test that we don't get segments for those files
@pytest.mark.filterwarnings("ignore: loadtxt")
def test_segment_replicates(wav_path_seg_txt_path_tuple):
    """Test that :func:`vocalpy.segment.ava.segment` replicates segmenting results
    obtained with the ``ava`` package."""
    wav_path, seg_txt_path = wav_path_seg_txt_path_tuple

    if seg_txt_path.name == "LL_21462x24783_ltr2_pup2_ch4_2600_f_359_278_fr0_p5_2020-02-27_11-45-23-clip.txt":
        pytest.xfail(
            "Extra segment in oracle data because of bug that is now fixed, see:"
            "https://github.com/pearsonlab/autoencoded-vocal-analysis/pull/13"
        )

    segs = np.loadtxt(seg_txt_path)
    segs = segs.reshape(-1,2)
    onsets_gt, offsets_gt = segs[:,0], segs[:,1]

    sound = voc.Sound.read(wav_path)
    params = {**voc.segment.JOURJINEETAL2023}
    del params['min_isi_dur']
    segments = voc.segment.ava(sound, **params)

    assert isinstance(segments, vocalpy.Segments)
    # we set atol=1e-5 because we expect values to be the same up to roughly 5th decimal place
    # https://stackoverflow.com/questions/65909842/what-is-rtol-for-in-numpys-allclose-function
    np.testing.assert_allclose(segments.start_times, onsets_gt, atol=1e-5, rtol=0)
    np.testing.assert_allclose(segments.stop_times, offsets_gt, atol=1e-5, rtol=0)


@pytest.fixture(params=JOURJINE_ETAL_2023_WAV_LIST)
def jourjine_et_al_wav_2023_path(request):
    return request.param


def test_segment_min_isi_dur(jourjine_et_al_wav_2023_path):
    """Test that :func:`vocalpy.segment.ava.segment` parameter `min_isi_dur` works as expected"""
    sound = voc.Sound.read(jourjine_et_al_wav_2023_path)
    params = {**voc.segment.JOURJINEETAL2023}
    segments_isi = voc.segment.ava(sound, **params)
    params_wout_isi = {k: v for k, v in params.items() if k != 'min_isi_dur'}
    segments = voc.segment.ava(sound, **params_wout_isi)

    isi_durs_with_min_isi = segments_isi.start_times[1:] - segments_isi.stop_times[:-1]
    assert not np.any(isi_durs_with_min_isi < params['min_isi_dur'])

    isi_durs_without_min_isi = segments.start_times[1:] - segments.stop_times[:-1]
    if np.any(isi_durs_without_min_isi < params['min_isi_dur']):
        assert segments_isi.start_times.shape[0] < segments.start_times.shape[0]
        assert np.allclose(
            isi_durs_without_min_isi[isi_durs_without_min_isi > params['min_isi_dur']],
            isi_durs_with_min_isi
        )
    else:
        assert np.allclose(
            isi_durs_without_min_isi, isi_durs_with_min_isi
        )
