import numpy as np
import pytest
import soundfile

import vocalpy

from .fixtures.audio import MULTICHANNEL_FLY_WAV, BIRDSONGREC_WAV_LIST


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
    def load_an_audio_path(an_audio_path):
        if an_audio_path.name.endswith("cbin"):
            data, samplerate = vocalpy._vendor.evfuncs.load_cbin(an_audio_path)
            channels = 1
            audio_format = "cbin"
        elif an_audio_path.name.endswith("wav"):
            data, samplerate = soundfile.read(an_audio_path, always_2d=True)
            channels = data.shape[1]
            data = np.transpose(data, (1, 0))
            audio_format = "wav"
        else:
            raise ValueError(
                f"Unrecognized format: {an_audio_path.suffix}"
            )
        return data, samplerate, channels, audio_format

    def test_read(self, an_audio_path):
        """Test that :meth:`vocalpy.Sound.read` works as expected."""
        data, samplerate, channels, audio_format = self.load_an_audio_path(an_audio_path)

        sound = vocalpy.Sound.read(an_audio_path)
        assert_sound_is_instance_with_expected_attrs(
            sound, data, samplerate, channels, audio_format
        )

    def test_write(self, an_audio_path, tmp_path):
        """Test that :meth:`vocalpy.Sound.write` works as expected.

        To do this we instantiate a Sound instance directly,
        write it to a file, read it, and then test that the loaded
        data is what we expect.
        """
        data, samplerate, channels, audio_format = self.load_an_audio_path(an_audio_path)

        if audio_format == "cbin":
            # we have to normalize cbin
            # https://stackoverflow.com/a/42544738/4906855
            data_float_normal = data.astype(np.float64) / 32768.0
            sound = vocalpy.Sound(data=data_float_normal, samplerate=samplerate)
        else:
            sound = vocalpy.Sound(data=data, samplerate=samplerate)
        tmp_wav_path = tmp_path / (an_audio_path.stem + ".wav")
        assert not tmp_wav_path.exists()

        sound.write(tmp_wav_path)

        assert tmp_wav_path.exists()

        sound_loaded = vocalpy.Sound.read(tmp_wav_path)

        assert_sound_is_instance_with_expected_attrs(
            sound_loaded, data, samplerate, channels, audio_format
        )

    def test_write_raises(self, a_cbin_path, tmp_path):
        sound = vocalpy.Sound.read(a_cbin_path)
        tmp_cbin_path = tmp_path / a_cbin_path.name
        with pytest.raises(ValueError):
            sound.write(tmp_cbin_path)

    @pytest.mark.parametrize(
        'a_wav_path',
        [
            # audio with one channel
            BIRDSONGREC_WAV_LIST[0],
            # audio with more than one channel
            MULTICHANNEL_FLY_WAV,
        ]
    )
    def test___iter__(self, a_wav_path):
        sound = vocalpy.Sound.read(a_wav_path)
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
        'a_wav_path, key',
        [
            # audio with one channel
            (BIRDSONGREC_WAV_LIST[0], 0),
            # audio with more than one channel
            (MULTICHANNEL_FLY_WAV, slice(None, 2)),
        ]
    )
    def test___getitem__(self, a_wav_path, key):
        sound = vocalpy.Sound.read(a_wav_path)
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
        'a_wav_path, key',
        [
            # audio with one channel
            (BIRDSONGREC_WAV_LIST[0], 1),
            # audio with more than one channel
            (MULTICHANNEL_FLY_WAV, 5),
        ]
    )
    def test___getitem__raises(self, a_wav_path, key):
        sound = vocalpy.Sound.read(a_wav_path)
        with pytest.raises(IndexError):
            _ = sound[key]
