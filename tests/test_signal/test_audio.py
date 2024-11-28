import numpy as np

import vocalpy


def test_bandpass_filtfilt(all_soundfile_paths):
    """Smoke test for bandpass_filtfilt"""
    sound = vocalpy.Sound.read(all_soundfile_paths)
    if sound.samplerate <= 10000:  # fly multichannel has low samplerate
        out = vocalpy.signal.audio.bandpass_filtfilt(sound, freq_cutoffs=(100, 4900))
    else:
        out = vocalpy.signal.audio.bandpass_filtfilt(sound)
    assert isinstance(out, vocalpy.Sound)
    assert not np.array_equal(out.data, sound.data)


def test_meansquared(all_soundfile_paths):
    sound = vocalpy.Sound.read(all_soundfile_paths)
    if sound.samplerate <= 10000:  # fly multichannel has low samplerate
        out = vocalpy.signal.audio.meansquared(sound, freq_cutoffs=(100, 4900))
    else:
        out = vocalpy.signal.audio.meansquared(sound)
    assert isinstance(out, np.ndarray)
    assert np.all(out >= 0.)  # because it's squared
