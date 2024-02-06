import pytest

import vocalpy

from .fixtures.audio import BIRDSONGREC_WAV_LIST
from .fixtures.spect import SPECT_LIST_NPZ


@pytest.mark.parametrize(
    'a_wav_path',
    BIRDSONGREC_WAV_LIST[:3]
)
def test_default_spect_fname_func(a_wav_path):
    spect_fname = vocalpy.spectrogram_maker.default_spect_fname_func(a_wav_path)
    assert spect_fname == a_wav_path.name + vocalpy.constants.SPECT_FILE_EXT


@pytest.mark.parametrize(
    "audio",
    [
        vocalpy.Sound.read(BIRDSONGREC_WAV_LIST[0]),
        vocalpy.AudioFile(path=BIRDSONGREC_WAV_LIST[0]),
        [vocalpy.Sound.read(path) for path in BIRDSONGREC_WAV_LIST[:3]],
        [vocalpy.AudioFile(path=path) for path in BIRDSONGREC_WAV_LIST[:3]],
    ],
)
def test_validate_sound(audio):
    assert vocalpy.spectrogram_maker.validate_sound(audio) is None


@pytest.mark.parametrize(
    "not_audio, expected_exception",
    [
        (vocalpy.Spectrogram.read(SPECT_LIST_NPZ[0]), TypeError),
        (dict(), TypeError),
        ([vocalpy.Spectrogram.read(path) for path in SPECT_LIST_NPZ[:3]], TypeError),
        (
            [vocalpy.Sound.read(path) for path in BIRDSONGREC_WAV_LIST[:3]]
            + [vocalpy.AudioFile(path=path) for path in BIRDSONGREC_WAV_LIST[:3]],
            TypeError,
        ),
    ],
)
def test_validate_sound_not_audio_raises(not_audio, expected_exception):
    with pytest.raises(expected_exception=expected_exception):
        vocalpy.spectrogram_maker.validate_sound(not_audio)


class TestSpectrogramMaker:
    @pytest.mark.parametrize(
        "callback, spect_params",
        [
            (None, None),
        ],
    )
    def test_init(self, callback, spect_params):
        spect_maker = vocalpy.SpectrogramMaker(callback=callback, spect_params=spect_params)
        assert isinstance(spect_maker, vocalpy.SpectrogramMaker)
        if callback is None and spect_params is None:
            assert spect_maker.callback is vocalpy.spectrogram
            assert spect_maker.spect_params == vocalpy.spectrogram_maker.DEFAULT_SPECT_PARAMS
        else:
            assert spect_maker.callback is callback
            assert spect_maker.spect_params == spect_params

    @pytest.mark.parametrize(
        "audio",
        [
            vocalpy.Sound.read(BIRDSONGREC_WAV_LIST[0]),
            vocalpy.AudioFile(path=BIRDSONGREC_WAV_LIST[0]),
            [vocalpy.Sound.read(path) for path in BIRDSONGREC_WAV_LIST[:3]],
            [vocalpy.AudioFile(path=path) for path in BIRDSONGREC_WAV_LIST[:3]],
        ],
    )
    def test_make(self, audio):
        spect_maker = vocalpy.SpectrogramMaker()
        out = spect_maker.make(audio)
        if isinstance(audio, (vocalpy.Sound, vocalpy.AudioFile)):
            assert isinstance(out, vocalpy.Spectrogram)
        elif isinstance(audio, list):
            assert all([isinstance(spect, vocalpy.Spectrogram) for spect in out])

    @pytest.mark.parametrize(
        "audio",
        [
            vocalpy.Sound.read(BIRDSONGREC_WAV_LIST[0]),
            vocalpy.AudioFile(path=BIRDSONGREC_WAV_LIST[0]),
            [vocalpy.Sound.read(path) for path in BIRDSONGREC_WAV_LIST[:3]],
            [vocalpy.AudioFile(path=path) for path in BIRDSONGREC_WAV_LIST[:3]],
        ],
    )
    def test_write(self, audio, tmp_path):
        spect_maker = vocalpy.SpectrogramMaker()

        out = spect_maker.write(audio, dir_path=tmp_path)

        if isinstance(audio, (vocalpy.Sound, vocalpy.AudioFile)):
            assert isinstance(out, vocalpy.SpectrogramFile)
            path = out.path
            assert path.exists()
            if isinstance(audio, vocalpy.Sound):
                assert out.source_audio_file.path == audio.path
            elif isinstance(audio, vocalpy.AudioFile):
                assert out.source_audio_file.path == audio.path

        elif isinstance(audio, list):
            assert isinstance(out, list)
            assert all([isinstance(spect_file, vocalpy.SpectrogramFile) for spect_file in out])
            for spect_file in out:
                path = spect_file.path
                assert path.exists()
                if isinstance(audio, vocalpy.Sound):
                    assert spect_file.source_audio_file.path == audio.path
                elif isinstance(audio, vocalpy.AudioFile):
                    assert spect_file.source_audio_file.path == audio.path
