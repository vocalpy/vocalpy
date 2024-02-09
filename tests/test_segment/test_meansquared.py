import crowsetta
import numpy as np
import pytest

import vocalpy.segment


def test_meansquared(a_cbin_path):
    sound = vocalpy.Sound.read(a_cbin_path)

    notmat = str(a_cbin_path) + '.not.mat'
    nmd = crowsetta.formats.seq.notmat.load_notmat(notmat)
    min_syl_dur = nmd['min_dur'] / 1000
    min_silent_dur = nmd['min_int'] / 1000
    threshold = nmd['threshold']

    onsets, offsets = vocalpy.segment.meansquared(sound, threshold, min_syl_dur, min_silent_dur)
    assert isinstance(onsets, np.ndarray)
    assert isinstance(offsets, np.ndarray)
    assert len(onsets) == len(offsets)


def test_meansquared_raises(multichannel_fly_wav_sound):
    with pytest.raises(ValueError):
        _ = vocalpy.segment.meansquared(multichannel_fly_wav_sound)
