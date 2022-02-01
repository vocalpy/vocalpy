import pytest

import vocalpy


@pytest.mark.parametrize(
    'audio_format',
    [
        'wav',
        'cbin',
    ]
)
def test_audio(audio_format,
               specific_audio_list):
    list_of_audio_files = specific_audio_list(audio_format)
    for audio_file in list_of_audio_files:
        audio = vocalpy.Audio.from_file(audio_file)
        assert isinstance(audio, vocalpy.Audio)
