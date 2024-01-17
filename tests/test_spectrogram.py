import pytest

import vocalpy

from .fixtures.audio import AUDIO_LIST_WAV


@pytest.mark.parametrize(
    'method',
    [
        None,
        'librosa-db',
    ]
)
def test_spectrogram(method, a_wav_path):
    audio = vocalpy.Audio.read(a_wav_path)
    if method is not None:
        spectrogram = vocalpy.spectrogram(audio, method=method)
    else:
        spectrogram = vocalpy.spectrogram(audio)
    assert isinstance(spectrogram, vocalpy.Spectrogram)


def test_input_not_audio_raises():
    """Test :func:`vocalpy.spectrogram` raises ValueError when first arg is not Audio"""
    audio = vocalpy.Audio.read(AUDIO_LIST_WAV[0])
    with pytest.raises(TypeError):
        vocalpy.spectrogram(audio.data)


def test_method_not_valid_raises():
    """Test :func:`vocalpy.spectrogram` raises ValueError when method arg is not valid"""
    audio = vocalpy.Audio.read(AUDIO_LIST_WAV[0])
    with pytest.raises(ValueError):
        vocalpy.spectrogram(audio, method='incorrect-method-name')
