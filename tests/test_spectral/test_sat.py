import numpy as np
import pytest

import vocalpy

from ..fixtures.audio import ALL_ZEBRA_FINCH_WAVS, MULTICHANNEL_FLY_WAV


@pytest.mark.parametrize(
    'audio_path',
    ALL_ZEBRA_FINCH_WAVS + [MULTICHANNEL_FLY_WAV]
)
def test__sat_multitaper(audio_path):
    """Test :func:`vocalpy.spectral.sat` returns expected outputs"""
    sound = vocalpy.Sound.read(audio_path)
    out = vocalpy.spectral.sat._sat_multitaper(sound)
    assert len(out) == 3
    power_spectrogram, spectra1, spectra2 = out
    assert isinstance(power_spectrogram, vocalpy.Spectrogram)
    assert power_spectrogram.data.shape[0] == sound.data.shape[0]
    assert power_spectrogram.data.ndim == 3
    assert isinstance(spectra1, np.ndarray)
    assert isinstance(spectra2, np.ndarray)


@pytest.mark.parametrize(
    'audio_path',
    ALL_ZEBRA_FINCH_WAVS + [MULTICHANNEL_FLY_WAV]
)
def test_sat_multitaper(audio_path):
    """Test :func:`vocalpy.spectral.sat` returns expected outputs"""
    sound = vocalpy.Sound.read(audio_path)
    out = vocalpy.spectral.sat.sat_multitaper(sound)
    assert isinstance(out, vocalpy.Spectrogram)
    assert out.data.shape[0] == sound.data.shape[0]
    assert out.data.ndim == 3
