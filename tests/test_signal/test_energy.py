import numpy as np

import vocalpy


def test_meansquared(all_soundfile_paths):
    sound = vocalpy.Sound.read(all_soundfile_paths)
    if sound.samplerate <= 10000:  # fly multichannel has low samplerate
        out = vocalpy.signal.energy.meansquared(sound, freq_cutoffs=(100, 4900))
    else:
        out = vocalpy.signal.energy.meansquared(sound)
    assert isinstance(out, np.ndarray)
    assert np.all(out >= 0.)  # because it's squared
