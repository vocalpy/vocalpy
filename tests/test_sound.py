import librosa
import numpy as np
import pytest
import soundfile

import vocalpy

from .fixtures.audio import (
    MULTICHANNEL_FLY_WAV,
    BIRDSONGREC_WAV_LIST,
    JOURJINE_ET_AL_GO_WAV_PATH,
    BFSONGREPO_BL26LB16_WAV_PATH,
)


RNG = np.random.default_rng()

def assert_sound_is_instance_with_expected_attrs(
        sound, data, samplerate, channels, audio_format
):
    """Assertions helper we use in TestSound methods"""
    assert isinstance(sound, vocalpy.Sound)

    for attr_name, attr_val in zip(("data", "samplerate", "channels"), (data, samplerate, channels)):
        assert hasattr(sound, attr_name)
        if attr_name == "data":
            if audio_format == "cbin":
                # add channel dim to data, needed for cbin audio
                attr_val = (attr_val.astype(np.float64) / 32768.0)
            if attr_val.ndim == 1:
                attr_val = attr_val[np.newaxis, ...]
            np.testing.assert_allclose(
                getattr(sound, attr_name), attr_val
            )
        else:
            assert getattr(sound, attr_name) == attr_val

    assert sound.samples == sound.data.shape[1]
    assert sound.duration == sound.data.shape[1] / sound.samplerate


class TestSound:
    @pytest.mark.parametrize(
        "data, samplerate, channels",
        [
            (RNG.normal(size=(int(32000 * 2.17))), 32000, 1),
            (RNG.normal(size=(1, int(32000 * 2.17))), 32000, 1),
            (RNG.normal(size=(2, int(32000 * 2.17))), 32000, 2),
            (RNG.normal(size=(2, int(48000 * 2.17))), 48000, 2),
            (RNG.normal(size=(6, int(48000 * 2.17))), 48000, 6),
        ],
    )
    def test_init(self, data, samplerate, channels):
        """Test that we can initialize a :class:`vocalpy.Sound` instance."""
        sound = vocalpy.Sound(data=data, samplerate=samplerate)

        assert_sound_is_instance_with_expected_attrs(
            sound, data, samplerate, channels, audio_format="wav"
        )

    @pytest.mark.parametrize(
        "data, samplerate, expected_exception",
        [
            # `data` is not a numpy array
            (RNG.normal(size=(int(32000 * 2.17))).tolist(), 32000, TypeError),
            # `data` is zero dimensional
            (np.array(2), 32000, ValueError),
            # `data` has more than 2 dimensions
            (RNG.normal(size=(int(32000 * 2.17)))[:, np.newaxis, np.newaxis], 32000, ValueError),
            # `samplerate` is not an int
            # this raises a ValueError because the converter can't cast the string to an int
            (RNG.normal(size=(1, int(32000 * 2.17))), "f", TypeError),
            # `samplerate` is less than zero
            (RNG.normal(size=(1, int(32000 * 2.17))), -32000, ValueError),
        ],
    )
    def test_init_raises(self, data, samplerate, expected_exception):
        """Test that :class:`vocalpy.Sound` raises expected errors"""
        with pytest.raises(expected_exception):
            vocalpy.Sound(data=data, samplerate=samplerate)

    def test_init_warns(self):
        """Test that we get a warning if number channels > number of samples"""
        with pytest.warns():
            vocalpy.Sound(data=RNG.normal(size=(int(32000 * 2.17), 1)), samplerate=32000)

    @pytest.mark.parametrize(
        "data, samplerate",
        [
            (RNG.normal(size=(int(32000 * 2.17))), 32000),
            (RNG.normal(size=(1, int(32000 * 2.17))), 32000),
            (RNG.normal(size=(2, int(32000 * 2.17))), 32000),
            (RNG.normal(size=(2, int(48000 * 2.17))), 48000),
            (RNG.normal(size=(6, int(48000 * 2.17))), 48000),
        ],
    )
    def test___eq__(self, data, samplerate):
        sound = vocalpy.Sound(data=data, samplerate=samplerate)
        other = vocalpy.Sound(data=data.copy(), samplerate=samplerate)
        assert sound == other

    @pytest.mark.parametrize(
        "data, samplerate",
        [
            (RNG.normal(size=(int(32000 * 2.17))), 32000),
            (RNG.normal(size=(1, int(32000 * 2.17))), 32000),
            (RNG.normal(size=(2, int(32000 * 2.17))), 32000),
            (RNG.normal(size=(2, int(48000 * 2.17))), 48000),
            (RNG.normal(size=(6, int(48000 * 2.17))), 48000),
        ],
    )
    def test___ne__(self, data, samplerate):
        sound = vocalpy.Sound(data=data, samplerate=samplerate)
        other = vocalpy.Sound(data=data.copy() + 0.001, samplerate=samplerate)
        assert sound != other

    @staticmethod
    def load_all_soundfile_paths(all_soundfile_paths):
        if all_soundfile_paths.name.endswith("cbin"):
            data, samplerate = vocalpy._vendor.evfuncs.load_cbin(all_soundfile_paths)
            channels = 1
            audio_format = "cbin"
        elif all_soundfile_paths.name.endswith("wav"):
            data, samplerate = soundfile.read(all_soundfile_paths, always_2d=True)
            channels = data.shape[1]
            data = np.transpose(data, (1, 0))
            audio_format = "wav"
        else:
            raise ValueError(
                f"Unrecognized format: {all_soundfile_paths.suffix}"
            )
        return data, samplerate, channels, audio_format

    def test_read(self, all_soundfile_paths):
        """Test that :meth:`vocalpy.Sound.read` works as expected."""
        data, samplerate, channels, audio_format = self.load_all_soundfile_paths(all_soundfile_paths)

        sound = vocalpy.Sound.read(all_soundfile_paths)
        assert_sound_is_instance_with_expected_attrs(
            sound, data, samplerate, channels, audio_format
        )

    def test_write(self, all_soundfile_paths, tmp_path):
        """Test that :meth:`vocalpy.Sound.write` works as expected.

        To do this we instantiate a Sound instance directly,
        write it to a file, read it, and then test that the loaded
        data is what we expect.
        """
        data, samplerate, channels, audio_format = self.load_all_soundfile_paths(all_soundfile_paths)

        if audio_format == "cbin":
            # we have to normalize cbin
            # https://stackoverflow.com/a/42544738/4906855
            data_float_normal = data.astype(np.float64) / 32768.0
            sound = vocalpy.Sound(data=data_float_normal, samplerate=samplerate)
        else:
            sound = vocalpy.Sound(data=data, samplerate=samplerate)
        tmp_wav_path = tmp_path / (all_soundfile_paths.stem + ".wav")
        assert not tmp_wav_path.exists()

        sound.write(tmp_wav_path)

        assert tmp_wav_path.exists()

        sound_loaded = vocalpy.Sound.read(tmp_wav_path)

        assert_sound_is_instance_with_expected_attrs(
            sound_loaded, data, samplerate, channels, audio_format
        )

    def test_write_raises(self, all_cbin_paths, tmp_path):
        sound = vocalpy.Sound.read(all_cbin_paths)
        tmp_cbin_path = tmp_path / all_cbin_paths.name
        with pytest.raises(ValueError):
            sound.write(tmp_cbin_path)

    @pytest.mark.parametrize(
        'all_wav_paths',
        [
            # audio with one channel
            BIRDSONGREC_WAV_LIST[0],
            # audio with more than one channel
            MULTICHANNEL_FLY_WAV,
        ]
    )
    def test___iter__(self, all_wav_paths):
        sound = vocalpy.Sound.read(all_wav_paths)
        sound_channels = [
            sound_ for sound_ in sound
        ]
        assert all(
            [isinstance(sound_, vocalpy.Sound)
             for sound_ in sound_channels]
        )
        for channel, sound_channel in enumerate(sound_channels):
            np.testing.assert_allclose(
                sound_channel.data, sound.data[channel][np.newaxis, ...]
            )
        assert len(sound_channels) == sound.data.shape[0]

    @pytest.mark.parametrize(
        'all_wav_paths, key',
        [
            # audio with one channel
            (BIRDSONGREC_WAV_LIST[0], 0),
            # audio with more than one channel
            (MULTICHANNEL_FLY_WAV, slice(None, 2)),
        ]
    )
    def test___getitem__(self, all_wav_paths, key):
        sound = vocalpy.Sound.read(all_wav_paths)
        sound_channel = sound[key]
        assert isinstance(sound_channel, vocalpy.Sound)
        if isinstance(key, int):
            assert sound_channel.data.shape[0] == 1
            np.testing.assert_allclose(
                sound_channel.data, sound.data[key][np.newaxis, ...]
            )
        elif isinstance(key, slice):
            sliced = sound.data[key]
            assert sound_channel.data.shape[0] == sliced.shape[0]
            np.testing.assert_allclose(
                sound_channel.data, sliced
            )

    @pytest.mark.parametrize(
        'all_wav_paths, key',
        [
            # audio with one channel
            (BIRDSONGREC_WAV_LIST[0], 1),
            # audio with more than one channel
            (MULTICHANNEL_FLY_WAV, 5),
        ]
    )
    def test___getitem__raises(self, all_wav_paths, key):
        sound = vocalpy.Sound.read(all_wav_paths)
        with pytest.raises(IndexError):
            _ = sound[key]

    @pytest.mark.parametrize(
        'segfunc, kwargs, sound',
        [
            (
                vocalpy.segment.meansquared,
                dict(threshold=5000, min_dur=0.02, min_silent_dur=0.004),
                vocalpy.Sound.read(BFSONGREPO_BL26LB16_WAV_PATH)
            ),
            (
                vocalpy.segment.ava, 
                {**vocalpy.segment.JOURJINEETAL2023}, 
                vocalpy.Sound.read(JOURJINE_ET_AL_GO_WAV_PATH)
            ),
        ]
    )
    def test_segment(self, segfunc, kwargs, sound):
        segments = segfunc(sound, **kwargs)
        segsounds = sound.segment(segments)
        assert all([
            isinstance(sound, vocalpy.Sound)
            for sound in segsounds
        ])
        assert len(segsounds) == len(segments)

    @pytest.mark.parametrize(
        'sound, wrong_segments, expected_exception',
        [
            (
                vocalpy.Sound.read(BFSONGREPO_BL26LB16_WAV_PATH),
                # a list is not a Segments instance, should raise a TypeError
                [],
                TypeError,
            )
        ]
    )
    def test_segment_raises(self, sound, wrong_segments, expected_exception):
        with pytest.raises(expected_exception):
            sound.segment(wrong_segments)

    @pytest.mark.parametrize(
        'start, stop, expected_clip_duration',
        [
            (0.5, 0.75, 0.25),
            (None, 0.75, 0.75),
            # if we just use these defaults we should get the same sound back
            (0., None, "sound_duration"),
        ]
    )
    def test_clip(self, start, stop, expected_clip_duration, all_wav_paths):
        sound = vocalpy.Sound.read(all_wav_paths)

        if stop is None:
            clip = sound.clip(start)  # test default stop of None
        elif start is None:
            clip = sound.clip(stop=stop)  # test default start of 0.
        else:
            clip = sound.clip(start, stop)

        # FIXME: test we get data we expect at sample level
        assert isinstance(clip, vocalpy.Sound)
        if expected_clip_duration == "sound_duration":
            expected_clip_duration = sound.duration
        assert np.allclose(clip.duration, expected_clip_duration)

    @pytest.mark.parametrize(
        'start, stop, expected_exception',
        [
            # start can't be int
            (1, None, TypeError),
            # start can be negative
            (-1.0, None, ValueError),
            # start can't be greater than duration of sound
            (10000.0, None, ValueError),
            # stop can't be int
            (0.0, 1, TypeError),
            # stop can't be less than start
            (1.0, 0.5, ValueError),
            # stop can't be negative (still because it's less than start)
            (0.0, -1.0, ValueError),
            # stop can't be greater than duration of sound
            (0., 10000.0, ValueError),
            # stop can't be int -- make sure we check for this when start is None
            (None, 1, TypeError),
            # stop can't be negative -- make sure we check for this when start is None
            (None, -1.0, ValueError),
        ]
    )
    def test_clip_raises(self, start, stop, expected_exception, all_wav_paths):
        sound = vocalpy.Sound.read(all_wav_paths)
        with pytest.raises(expected_exception):
            if start is None:
                sound.clip(stop=stop)
            else:
                sound.clip(start, stop)

    def test_to_mono(self, all_elie_theunissen_2016_wav_paths):
        sound = vocalpy.Sound.read(all_elie_theunissen_2016_wav_paths)
        sound_mono = sound.to_mono()
        assert np.array_equal(
            sound_mono.data,
            librosa.to_mono(sound.data)[np.newaxis, :]
        )
