import numpy as np
import pytest

import vocalpy

from ..fixtures.audio import ALL_ZEBRA_FINCH_WAVS, MULTICHANNEL_FLY_WAV


@pytest.mark.parametrize(
    'audio_path',
    ALL_ZEBRA_FINCH_WAVS + [MULTICHANNEL_FLY_WAV]
)
def test_soundsig_spectro(audio_path):
    """Test :func:`vocalpy.spectral.soundsig_spectro` returns expected outputs"""
    sound = vocalpy.Sound.read(audio_path)
    spect = vocalpy.spectral.soundsig.soundsig_spectro(sound)
    assert isinstance(spect, vocalpy.Spectrogram)
    assert spect.data.shape[0] == sound.data.shape[0]
    assert spect.data.ndim == 3
