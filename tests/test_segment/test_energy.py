import evfuncs
import numpy as np

import vocalpy.segment


def test_audio_amplitude(a_cbin_path):
    audio = vocalpy.Audio.read(a_cbin_path)

    notmat = str(a_cbin_path) + '.not.mat'
    nmd = evfuncs.load_notmat(notmat)
    min_syl_dur = nmd['min_dur'] / 1000
    min_silent_dur = nmd['min_int'] / 1000
    threshold = nmd['threshold']

    onsets, offsets = vocalpy.segment.energy(audio, threshold, min_syl_dur, min_silent_dur)
    assert isinstance(onsets, np.ndarray)
    assert isinstance(offsets, np.ndarray)
    assert len(onsets) == len(offsets)
