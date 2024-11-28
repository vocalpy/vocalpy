import inspect

import pytest

import vocalpy

from .fixtures.audio import BIRDSONGREC_WAV_LIST


class TestSegmenter:
    @pytest.mark.parametrize(
        "callback, params, expected_callback, expected_params",
        [
            (None,
             None,
             vocalpy.segment.meansquared,
             vocalpy.segmenter.DEFAULT_SEGMENT_PARAMS),
            (vocalpy.segment.meansquared,
             None,
             vocalpy.segment.meansquared,
             {name: param.default
              for name, param in inspect.signature(vocalpy.segment.meansquared).parameters.items()
              if param.default is not inspect._empty}
             ),
            (vocalpy.segment.meansquared,
             {"smooth_win": 2},
             vocalpy.segment.meansquared,
             {"smooth_win": 2}
             ),
            (vocalpy.segment.meansquared,
             vocalpy.segment.MeanSquaredParams(threshold=1500),
             vocalpy.segment.meansquared,
             {**vocalpy.segment.MeanSquaredParams(threshold=1500)}),
            (vocalpy.segment.ava,
             None,
             vocalpy.segment.ava,
             {name: param.default
              for name, param in inspect.signature(vocalpy.segment.ava).parameters.items()
              if param.default is not inspect._empty},
             ),
            (vocalpy.segment.ava,
             vocalpy.segment.AvaParams(thresh_min=3.0),
             vocalpy.segment.ava,
             {**vocalpy.segment.AvaParams(thresh_min=3.0)}
             )
        ],
    )
    def test_init(self, callback, params, expected_callback, expected_params):
        segmenter = vocalpy.Segmenter(callback=callback, params=params)
        assert isinstance(segmenter, vocalpy.Segmenter)
        assert segmenter.callback is expected_callback
        assert segmenter.params == expected_params

    @pytest.mark.parametrize(
        "sound",
        [
            vocalpy.Sound.read(BIRDSONGREC_WAV_LIST[0]),
            vocalpy.AudioFile(path=BIRDSONGREC_WAV_LIST[0]),
            [vocalpy.Sound.read(path) for path in BIRDSONGREC_WAV_LIST[:3]],
            [vocalpy.AudioFile(path=path) for path in BIRDSONGREC_WAV_LIST[:3]],
        ],
    )
    def test_segment(self, sound, parallel):
        # have to use different segment params from default for these .wav files
        params = {
            "threshold": 5e-05,
            "min_dur": 0.02,
            "min_silent_dur": 0.002,
        }
        segmenter = vocalpy.Segmenter(params=params)

        out = segmenter.segment(sound, parallelize=parallel)

        if isinstance(sound, (vocalpy.Sound, vocalpy.AudioFile)):
            assert isinstance(out, vocalpy.Segments)
        elif isinstance(sound, list):
            assert isinstance(out, list)
            assert len(sound) == len(out)
            for segments in out:
                assert isinstance(segments, vocalpy.Segments)
