import inspect

import pytest

import vocalpy

from .fixtures.audio import BIRDSONGREC_WAV_LIST
from .fixtures.spect import SPECT_LIST_NPZ


@pytest.mark.parametrize(
    "sound",
    [
        vocalpy.Sound.read(BIRDSONGREC_WAV_LIST[0]),
        vocalpy.AudioFile(path=BIRDSONGREC_WAV_LIST[0]),
        [vocalpy.Sound.read(path) for path in BIRDSONGREC_WAV_LIST[:3]],
        [vocalpy.AudioFile(path=path) for path in BIRDSONGREC_WAV_LIST[:3]],
    ],
)
def test_validate_sound(sound):
    assert vocalpy.spectrogram_maker.validate_sound(sound) is None


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
        "callback, params, expected_callback, expected_params",
        [
            (None,
             None,
             vocalpy.spectrogram,
             vocalpy.spectrogram_maker.DEFAULT_SPECT_PARAMS
             ),
            (vocalpy.spectrogram,
             None,
             vocalpy.spectrogram,
             {name: param.default
              for name, param in inspect.signature(vocalpy.spectrogram).parameters.items()
              if param.default is not inspect._empty}
             )
        ],
    )
    def test_init(self, callback, params, expected_callback, expected_params):
        spect_maker = vocalpy.SpectrogramMaker(callback=callback, params=params)
        assert isinstance(spect_maker, vocalpy.SpectrogramMaker)
        assert spect_maker.callback is expected_callback
        assert spect_maker.params == expected_params

    @pytest.mark.parametrize(
        "sound",
        [
            vocalpy.Sound.read(BIRDSONGREC_WAV_LIST[0]),
            vocalpy.AudioFile(path=BIRDSONGREC_WAV_LIST[0]),
            [vocalpy.Sound.read(path) for path in BIRDSONGREC_WAV_LIST[:3]],
            [vocalpy.AudioFile(path=path) for path in BIRDSONGREC_WAV_LIST[:3]],
        ],
    )
    def test_make(self, sound, parallel):
        spect_maker = vocalpy.SpectrogramMaker()

        out = spect_maker.make(sound, parallelize=parallel)

        if isinstance(sound, (vocalpy.Sound, vocalpy.AudioFile)):
            assert isinstance(out, vocalpy.Spectrogram)
        elif isinstance(sound, list):
            assert all([isinstance(spect, vocalpy.Spectrogram) for spect in out])
