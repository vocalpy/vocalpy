import pytest

import vocalpy


@pytest.mark.parametrize(
    'spect_format',
    [
        'npz',
        'mat',
    ]
)
def test_spectrogram_from_file(spect_format,
                               specific_spect_list):
    spect_list = specific_spect_list(spect_format)
    for spect_path in spect_list:
        spect = vocalpy.Spectrogram.from_file(spect_path)
        assert isinstance(spect, vocalpy.Spectrogram)
        for spect_attrs in ('s', 't', 'f', 'audio_path'):
            assert hasattr(spect, spect_attrs)
