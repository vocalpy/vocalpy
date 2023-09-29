import numpy as np

import vocalpy


def test_smoothed_energy(an_audio_path):
    audio = vocalpy.Audio.read(an_audio_path)
    smoothed = vocalpy.signal.audio.smoothed_energy(audio)
    assert isinstance(smoothed, np.ndarray)
    assert np.all(smoothed >= 0.)  # because it's rectified
