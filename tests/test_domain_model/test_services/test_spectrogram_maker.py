import pytest

import vocalpy

from ...fixtures.audio import AUDIO_LIST_WAV
from ...fixtures.spect import SPECT_LIST_NPZ


@pytest.mark.parametrize(
    "audio_format",
    [
        "wav",
        "cbin",
    ],
)
def test_default_spect_fname_func(specific_audio_dir, audio_format):
    audio_dir = specific_audio_dir(audio_format)
    audio_paths = sorted(audio_dir.glob(f"*{audio_format}"))

    for audio_path in audio_paths:
        spect_fname = vocalpy.domain_model.services.spectrogram_maker.default_spect_fname_func(audio_path)
        assert spect_fname == audio_path.name + vocalpy.constants.SPECT_FILE_EXT


@pytest.mark.parametrize(
    'audio',
    [
        vocalpy.Audio.read(AUDIO_LIST_WAV[0]),
        vocalpy.AudioFile(path=AUDIO_LIST_WAV[0]),
        [vocalpy.Audio.read(path) for path in AUDIO_LIST_WAV[:3]],
        [vocalpy.AudioFile(path=path) for path in AUDIO_LIST_WAV[:3]]
    ]
)
def test_validate_audio(audio):
    assert vocalpy.domain_model.services.spectrogram_maker.validate_audio(audio) is None


@pytest.mark.parametrize(
    'not_audio, expected_exception',
    [
        (vocalpy.Spectrogram.read(SPECT_LIST_NPZ[0]), TypeError),
        (dict(), TypeError),
        ([vocalpy.Spectrogram.read(path) for path in SPECT_LIST_NPZ[:3]], TypeError),
        ([vocalpy.Audio.read(path) for path in AUDIO_LIST_WAV[:3]] + [vocalpy.AudioFile(path=path) for path in AUDIO_LIST_WAV[:3]],
         TypeError
         )
    ]
)
def test_validate_audio_not_audio_raises(not_audio, expected_exception):
    with pytest.raises(expected_exception=expected_exception):
        vocalpy.domain_model.services.spectrogram_maker.validate_audio(not_audio)



class TestSpectrogramMaker:
    @pytest.mark.parametrize(
        'callback, params',
        [
            (None, None),
        ]
    )
    def test_init(self, callback, params):
        spect_maker = vocalpy.SpectrogramMaker(callback=callback, params=params)
        assert isinstance(spect_maker, vocalpy.SpectrogramMaker)
        if callback is None and params is None:
            assert spect_maker.callback is vocalpy.signal.spectrogram.spectrogram
            assert spect_maker.params = vocalpy.SpectrogramParameters(fft_size=512)
        else:
            assert spect_maker.callback is callback
            assert spect_maker.params == params

    @pytest.mark.parametrize(
        "audio",
        [
            vocalpy.Audio.read(AUDIO_LIST_WAV[0]),
            vocalpy.AudioFile(path=AUDIO_LIST_WAV[0]),
            [vocalpy.Audio.read(path) for path in AUDIO_LIST_WAV[:3]],
            [vocalpy.AudioFile(path=path) for path in AUDIO_LIST_WAV[:3]],
        ],
    )
    def test_make(self, audio):
        spect_maker = vocalpy.SpectrogramMaker
        out = spect_maker.make(audio)
        if isinstance(audio, (vocalpy.Audio, vocalpy.AudioFile)):
            assert isinstance(out, vocalpy.Spectrogram)
        elif isinstance(audio, list):
            assert all([isinstance(spect, vocalpy.Spectrogram) for spect in out])

    @pytest.mark.parametrize(
        "audio",
        [
            vocalpy.Audio.read(AUDIO_LIST_WAV[0]),
            vocalpy.AudioFile(path=AUDIO_LIST_WAV[0]),
            [vocalpy.Audio.read(path) for path in AUDIO_LIST_WAV[:3]],
            [vocalpy.AudioFile(path=path) for path in AUDIO_LIST_WAV[:3]],
        ],
    )
    def test_write(self, audio, tmp_path):
        spect_maker = vocalpy.SpectrogramMaker

        out = spect_maker.write(audio, dir_path=tmp_path)

        if isinstance(audio, (vocalpy.Audio, vocalpy.AudioFile)):
            assert isinstance(out, vocalpy.SpectrogramFile)
            path = out.path
            assert path.exists()
            assert out.source_audio_file.path == audio.path

        elif isinstance(audio, list):
            assert all([isinstance(spect_file, vocalpy.SpectrogramFile) for spect_file in out])
            for spect_file in out:
                assert spect_file.path is not None
                spect_path = spect_file.path
                assert spect_path.exists()
                source_audio_path = spect_file.source_audio_file.path
