import pytest

import vocalpy

from .fixtures.audio import AUDIO_LIST_WAV


def assert_is_expected_sequence(sequence, audio, method, segment_params):
    assert isinstance(sequence, vocalpy.Sequence)
    assert sequence.audio.path == audio.path
    assert sequence.method == method
    assert sequence.segment_params == segment_params


class TestSegmenter:
    @pytest.mark.parametrize(
        "callback, method, segment_params",
        [
            (None, None, None),
            (vocalpy.signal.segment.audio_amplitude, None, {"smooth_win": 2}),
            # TODO: test 'method'
        ],
    )
    def test_init(self, callback, method, segment_params):
        segmenter = vocalpy.Segmenter(callback=callback, method=method, segment_params=segment_params)
        assert isinstance(segmenter, vocalpy.Segmenter)
        if callback is None and segment_params is None:
            assert segmenter.callback is vocalpy.signal.segment.audio_amplitude
            assert segmenter.segment_params == vocalpy.segmenter.DEFAULT_SEGMENT_PARAMS
        else:
            assert segmenter.callback is callback
            assert segmenter.segment_params == segment_params

    @pytest.mark.parametrize(
        "audio",
        [
            vocalpy.Audio.read(AUDIO_LIST_WAV[0]),
            vocalpy.AudioFile(path=AUDIO_LIST_WAV[0]),
            [vocalpy.Audio.read(path) for path in AUDIO_LIST_WAV[:3]],
            [vocalpy.AudioFile(path=path) for path in AUDIO_LIST_WAV[:3]],
        ],
    )
    def test_segment(self, audio):
        # have to use different segment params from default for these .wav files
        segment_params = {
            "threshold": 5e-05,
            "min_dur": 0.02,
            "min_silent_dur": 0.002,
        }
        segmenter = vocalpy.Segmenter(segment_params=segment_params)
        out = segmenter.segment(audio)
        if isinstance(audio, (vocalpy.Audio, vocalpy.AudioFile)):
            assert_is_expected_sequence(
                sequence=out, audio=audio, segment_params=segment_params, method=segmenter.callback.__name__
            )
            assert isinstance(out, vocalpy.Sequence)
        elif isinstance(audio, list):
            assert all([isinstance(spect, vocalpy.Sequence) for spect in out])
