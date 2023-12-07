import numpy as np

import vocalpy


def test_sat(a_zebra_finch_wav):
    """Smoke test that tests :func:`vocalpy.spectral.sat` returns expected outputs"""
    audio = vocalpy.Audio.read(a_zebra_finch_wav)
    out = vocalpy.spectral.sat(audio)
    assert len(out) == 6
    power_spectrogram, cepstrogram, quefrencies, max_freq, dSdt, dSdf = out
    assert isinstance(power_spectrogram, vocalpy.Spectrogram)
    for arr in (cepstrogram, quefrencies, dSdt, dSdf):
        assert isinstance(arr, np.ndarray)
    assert isinstance(max_freq, float)
