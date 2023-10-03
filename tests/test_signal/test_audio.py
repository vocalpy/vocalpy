import numpy as np

import vocalpy


def test_bandpass_filtfilt(an_audio_path):
    """Smoke test for bandpass_filtfilt"""
    audio = vocalpy.Audio.read(an_audio_path)
    out = vocalpy.signal.audio.bandpass_filtfilt(audio)
    assert isinstance(out, vocalpy.Audio)
    assert not np.array_equal(out.data, audio.data)


def test_meansquared(an_audio_path):
    audio = vocalpy.Audio.read(an_audio_path)
    out = vocalpy.signal.audio.meansquared(audio)
    assert isinstance(out, np.ndarray)
    assert np.all(out >= 0.)  # because it's squared
