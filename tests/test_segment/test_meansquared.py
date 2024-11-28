import crowsetta
import numpy as np
import pytest
import scipy.io

import vocalpy.segment.meansquared

from ..fixtures.segments import EVSONGANALY_SEGMENT_JSON


def test_meansquared(all_cbin_paths):
    sound = vocalpy.Sound.read(all_cbin_paths)

    notmat = str(all_cbin_paths) + '.not.mat'
    nmd = crowsetta.formats.seq.notmat.load_notmat(notmat)
    min_syl_dur = nmd['min_dur'] / 1000
    min_silent_dur = nmd['min_int'] / 1000
    threshold = nmd['threshold']

    segments = vocalpy.segment.meansquared(sound, threshold, min_syl_dur, min_silent_dur)
    assert isinstance(segments, vocalpy.Segments)


def test_meansquared_raises(multichannel_fly_wav_sound):
    """Test :func:`vocalpy.segment.meansquared` raises an error when a sound has multiple channels"""
    with pytest.raises(ValueError):
        _ = vocalpy.segment.meansquared(multichannel_fly_wav_sound)


@pytest.fixture(params=EVSONGANALY_SEGMENT_JSON)
def evsonganaly_segment_dict(request):
    return request.param


def test_meansquared_replicates_evsonganaly(evsonganaly_segment_dict):
    cbin_path = evsonganaly_segment_dict['cbin_path']
    notmat_path = evsonganaly_segment_dict['notmat_path']
    segment_mat_path = evsonganaly_segment_dict['segment_mat_path']
    sound = vocalpy.Sound.read(cbin_path)
    nmd = crowsetta.formats.seq.notmat.load_notmat(notmat_path)
    min_syl_dur = nmd['min_dur'] / 1000
    min_silent_dur = nmd['min_int'] / 1000
    threshold = nmd['threshold']

    # ---- output
    segments = vocalpy.segment.meansquared(sound, threshold, min_syl_dur, min_silent_dur)

    # ---- assert
    segment_dict = scipy.io.loadmat(segment_mat_path, squeeze_me=True)
    onsets_mat = segment_dict['onsets']
    offsets_mat = segment_dict['offsets']
    # set tolerances for numpy.allclose check.
    # difference np.abs(offsets - offsets_mat) is usually ~0.00003125...
    # We just don't want error to be larger than a millisecond
    # By trial and error, I find that these vals for tolerance result in
    # about that ceiling
    atol = 0.0005
    rtol = 0.00001
    # i.e., 0.0005 + 0.00001 * some_onsets_or_offset_array ~ [0.0005, 0.0005, ...]
    assert np.allclose(segments.start_times, onsets_mat, rtol, atol)
    assert np.allclose(segments.stop_times, offsets_mat, rtol, atol)
