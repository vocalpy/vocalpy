import evfuncs
import pytest
import soundfile

import vocalpy


def test_field_access():
    assert False


def test_defaults():
    assert False


def test_asdict():
    assert False


def test_equality():
    assert False


def test_inequality():
    assert False


@pytest.mark.parametrize(
    'spect_format, with_format_str',
    [
        ('npz', False),
        ('npz', True),
        ('mat', False),
        ('mat', True),
    ]
)
def test_spectrogram_from_file(spect_format: str,
                               with_format_str: bool,
                               specific_spect_list):
    spect_list = specific_spect_list(spect_format)
    if with_format_str:
        format_str = spect_format
    else:
        format_str = None
    for spect_path in spect_list:
        spect = vocalpy.Spectrogram.from_file(spect_path, format=format_str)
        assert isinstance(spect, vocalpy.Spectrogram)
        for spect_attrs in ('s', 't', 'f', 'audio_path'):
            assert hasattr(spect, spect_attrs)
        # TODO: rewrite to actually load dict and assert that attributes equal what's in dict
        # TODO: use an 'assert helper' for this
        pytest.fail()


def test_from_mat(spect_list_mat: list):
    for spect_path in spect_list_mat:
        spect = vocalpy.Spectrogram.from_mat(spect_path)
        assert isinstance(spect, vocalpy.Spectrogram)
        for spect_attrs in ('s', 't', 'f', 'audio_path'):
            assert hasattr(spect, spect_attrs)


def test_from_npz(spect_list_npz: list):
    for spect_path in spect_list_npz:
        spect = vocalpy.Spectrogram.from_npz(spect_path)
        assert isinstance(spect, vocalpy.Spectrogram)
        for spect_attrs in ('s', 't', 'f', 'audio_path'):
            assert hasattr(spect, spect_attrs)


@pytest.mark.parametrize(
    'audio_format',
    [
        'wav',
        'cbin',
    ]
)
def test_spectrogram_from_arrays(specific_audio_list,
                                 audio_format):
    audio_paths = specific_audio_list(audio_format)

    for audio_path in audio_paths:
        if audio_format == 'wav':
            data, samplerate = soundfile.read(audio_path)
        elif audio_format == 'cbin':
            data, samplerate = evfuncs.load_cbin(audio_path)

        s, t, f = vocalpy.signal.spectrogram(data=data, samplerate=samplerate)

        spect = vocalpy.Spectrogram(s=s, t=t, f=f)
        assert isinstance(spect, vocalpy.Spectrogram)
        for spect_attrs in ('s', 't', 'f', 'audio_path'):
            assert hasattr(spect, spect_attrs)


def test_to_file():
    assert False
