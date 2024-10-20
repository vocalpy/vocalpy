import pytest

import vocalpy

from .fixtures.audio import BIRDSONGREC_WAV_LIST


@pytest.mark.parametrize(
    'method',
    [
        None,
        'librosa-db',
    ]
)
def test_spectrogram(method, all_wav_paths):
    sound = vocalpy.Sound.read(all_wav_paths)
    if method is not None:
        spectrogram = vocalpy.spectrogram(sound, method=method)
    else:
        spectrogram = vocalpy.spectrogram(sound)
    assert isinstance(spectrogram, vocalpy.Spectrogram)


def test_input_not_audio_raises():
    """Test :func:`vocalpy.spectrogram` raises ValueError when first arg is not Sound"""
    sound = vocalpy.Sound.read(BIRDSONGREC_WAV_LIST[0])
    with pytest.raises(TypeError):
        vocalpy.spectrogram(sound.data)


def test_method_not_valid_raises():
    """Test :func:`vocalpy.spectrogram` raises ValueError when method arg is not valid"""
    sound = vocalpy.Sound.read(BIRDSONGREC_WAV_LIST[0])
    with pytest.raises(ValueError):
        vocalpy.spectrogram(sound, method='incorrect-method-name')
