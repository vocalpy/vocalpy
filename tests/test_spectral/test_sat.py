import numpy as np
import pytest

import vocalpy

from ..fixtures.audio import ALL_ZEBRA_FINCH_WAVS, MULTICHANNEL_FLY_WAV


@pytest.mark.parametrize(
    'audio_path',
    ALL_ZEBRA_FINCH_WAVS + [MULTICHANNEL_FLY_WAV]
)
def test_sat(audio_path):
    """Test :func:`vocalpy.spectral.sat` returns expected outputs"""
    sound = vocalpy.Sound.read(audio_path)
    out = vocalpy.spectral.sat(sound)
    assert len(out) == 6
    power_spectrogram, cepstrogram, quefrencies, max_freq, dSdt, dSdf = out
    assert isinstance(power_spectrogram, vocalpy.Spectrogram)
    assert power_spectrogram.data.shape[0] == sound.data.shape[0]
    assert power_spectrogram.data.ndim == 3
    assert isinstance(quefrencies, np.ndarray)
    for arr in (cepstrogram, dSdt, dSdf):
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3
        assert arr.shape[0] == sound.data.shape[0]
    assert isinstance(max_freq, float)
