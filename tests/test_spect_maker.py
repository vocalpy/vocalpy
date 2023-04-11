import pytest

import vocalpy


@pytest.mark.parametrize(
    'audio_format',
    [
        'wav',
        'cbin',
    ]
)
def test_default_spect_fname_func(specific_audio_dir,
                                  audio_format):
    audio_dir = specific_audio_dir(audio_format)
    audio_paths = sorted(audio_dir.glob(f'*{audio_format}'))

    for audio_path in audio_paths:
        spect_path = vocalpy.spect_maker.default_spect_fname_func(audio_path)
        assert spect_path == audio_path.parent / (audio_path.name + vocalpy.constants.SPECT_FILE_EXT)


@pytest.mark.parametrize(
    'audio_format, output_dir',
    [
        ('wav', None),
        ('cbin', 'tmp_path'),
    ]
)
def test_spect_maker_audio_paths(specific_audio_dir,
                                 audio_format,
                                 output_dir,
                                 tmp_path):
    audio_dir = specific_audio_dir(audio_format)
    if output_dir == 'tmp_path':
        output_dir = tmp_path

    audio_paths = sorted(audio_dir.glob(f'*{audio_format}'))
    spect_maker = vocalpy.SpectMaker
    spects = spect_maker.make(audio_paths, output_dir=output_dir)
    assert all([isinstance(spect, vocalpy.Spectrogram) for spect in spects])
    if output_dir is not None:
        for spect in spects:
            assert spect.spect_path is not None
            spect_path = spect.spect_path
            assert spect_path.exists()
