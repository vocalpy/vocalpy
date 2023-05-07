import evfuncs
import numpy as np

import vocalpy.signal.segment


def test_audio_amplitude(a_cbin_path):
    dat, fs = evfuncs.load_cbin(a_cbin_path)
    smooth = evfuncs.smooth_data(dat, fs)

    notmat = str(a_cbin_path) + '.not.mat'
    nmd = evfuncs.load_notmat(notmat)
    min_syl_dur = nmd['min_dur'] / 1000
    min_silent_dur = nmd['min_int'] / 1000
    threshold = nmd['threshold']

    onsets, offsets = vocalpy.signal.segment.audio_amplitude(smooth, fs, threshold, min_syl_dur, min_silent_dur)
    assert isinstance(onsets, np.ndarray)
    assert isinstance(offsets, np.ndarray)
    assert len(onsets) == len(offsets)
