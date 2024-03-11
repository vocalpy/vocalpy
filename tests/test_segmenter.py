import pytest

import vocalpy

from .fixtures.audio import BIRDSONGREC_WAV_LIST


def assert_segments_is_expected(segments, sound):
    assert isinstance(segments, vocalpy.Segments)
    if isinstance(sound, vocalpy.Sound):
        assert segments.sound is sound
    elif isinstance(sound, vocalpy.AudioFile):
        assert segments.sound == vocalpy.Sound.read(sound.path)


class TestSegmenter:
    @pytest.mark.parametrize(
        "callback, params",
        [
            (None, None),
            (vocalpy.segment.meansquared, {"smooth_win": 2}),
            # TODO: test `ava.segment`
            # TODO: test `ava.segment` with AvaParams
        ],
    )
    def test_init(self, callback, params):
        segmenter = vocalpy.Segmenter(callback=callback, params=params)
        assert isinstance(segmenter, vocalpy.Segmenter)
        if callback is None and params is None:
            assert segmenter.callback is vocalpy.segment.meansquared
            assert segmenter.params == vocalpy.segmenter.DEFAULT_SEGMENT_PARAMS
        else:
            assert segmenter.callback is callback
            assert segmenter.params == params

    @pytest.mark.parametrize(
        "sound",
        [
            vocalpy.Sound.read(BIRDSONGREC_WAV_LIST[0]),
            vocalpy.AudioFile(path=BIRDSONGREC_WAV_LIST[0]),
            [vocalpy.Sound.read(path) for path in BIRDSONGREC_WAV_LIST[:3]],
            [vocalpy.AudioFile(path=path) for path in BIRDSONGREC_WAV_LIST[:3]],
        ],
    )
    def test_segment(self, sound):
        # have to use different segment params from default for these .wav files
        params = {
            "threshold": 5e-05,
            "min_dur": 0.02,
            "min_silent_dur": 0.002,
        }
        segmenter = vocalpy.Segmenter(params=params)
        out = segmenter.segment(sound)
        if isinstance(sound, (vocalpy.Sound, vocalpy.AudioFile)):
            assert_segments_is_expected(out, sound)
        elif isinstance(sound, list):
            assert isinstance(out, list)
            assert len(sound) == len(out)
            for sound_, segments in zip(sound, out):
                assert_segments_is_expected(segments, sound)
