import numpy as np

import vocalpy


def test_smooth(an_audio_path):
    audio = vocalpy.Audio.read(an_audio_path)
    dat, fs = audio.data, audio.samplerate
    smoothed = vocalpy.signal.audio.smooth(dat, fs)
    assert isinstance(smoothed, np.ndarray)
    assert np.all(smoothed >= 0.)  # because it's rectified
