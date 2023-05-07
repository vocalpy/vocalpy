import numpy as np
import pytest
import soundfile

import vocalpy

RNG = np.random.default_rng()


@pytest.mark.parametrize(
    "data, expected_channels",
    [
        (RNG.normal(size=(int(32000 * 2.17))), 1),
        (RNG.normal(size=(int(32000 * 2.17))), 1),
        (RNG.normal(size=(int(32000 * 2.17), 1)), 1),
        (RNG.normal(size=(int(32000 * 2.17), 1)), 1),
        (RNG.normal(size=(int(32000 * 2.17), 2)), 2),
        (RNG.normal(size=(int(32000 * 2.17), 2)), 2),
        (RNG.normal(size=(int(48000 * 2.17), 2)), 2),
        (RNG.normal(size=(int(48000 * 2.17), 2)), 2),
        (RNG.normal(size=(int(48000 * 2.17), 6)), 6),
        (RNG.normal(size=(int(48000 * 2.17), 6)), 6),
    ],
)
def test_get_channels_from_data(data, expected_channels):
    channels = vocalpy.audio.get_channels_from_data(data)
    assert channels == expected_channels


class TestAudio:
    @pytest.mark.parametrize(
        "data, samplerate, channels",
        [
            (RNG.normal(size=(int(32000 * 2.17))), 32000, 1),
            (RNG.normal(size=(int(32000 * 2.17), 1)), 32000, 1),
            (RNG.normal(size=(int(32000 * 2.17), 2)), 32000, 2),
            (RNG.normal(size=(int(48000 * 2.17), 2)), 48000, 2),
            (RNG.normal(size=(int(48000 * 2.17), 6)), 48000, 6),
        ],
    )
    def test_init(self, data, samplerate, channels):
        """Test that we can initialize a :class:`vocalpy.Audio` instance."""
        audio = vocalpy.Audio(data=data, samplerate=samplerate)

        assert isinstance(audio, vocalpy.Audio)

        for attr_name, attr_val in zip(("data", "samplerate", "channels"), (data, samplerate, channels)):
            assert hasattr(audio, attr_name)
            assert getattr(audio, attr_name) is attr_val

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
            (RNG.normal(size=(int(32000 * 2.17), 1)), "f", ValueError),
            # `samplerate` is less than zero
            (RNG.normal(size=(int(32000 * 2.17), 1)), -32000, ValueError),
        ],
    )
    def test_init_raises(self, data, samplerate, expected_exception):
        """Test that :class:`vocalpy.Spectrogram` raises expected errors"""
        with pytest.raises(expected_exception):
            vocalpy.Audio(data=data, samplerate=samplerate)

    @pytest.mark.parametrize(
        "data, samplerate, channels",
        [
            (RNG.normal(size=(int(32000 * 2.17))), 32000, 1),
            (RNG.normal(size=(int(32000 * 2.17), 1)), 32000, 1),
            (RNG.normal(size=(int(32000 * 2.17), 2)), 32000, 2),
            (RNG.normal(size=(int(48000 * 2.17), 2)), 48000, 2),
            (RNG.normal(size=(int(48000 * 2.17), 6)), 48000, 6),
        ],
    )
    def test_asdict(self, data, samplerate, channels):
        audio = vocalpy.Audio(data=data, samplerate=samplerate)
        assert isinstance(audio, vocalpy.Audio)

        asdict = audio.asdict()
        assert isinstance(asdict, dict)

        for attr_name, attr_val in zip(("data", "samplerate", "channels"), (data, samplerate, channels)):
            assert attr_name in asdict
            assert asdict[attr_name] is attr_val

    @pytest.mark.parametrize(
        "data, samplerate",
        [
            (RNG.normal(size=(int(32000 * 2.17))), 32000),
            (RNG.normal(size=(int(32000 * 2.17), 1)), 32000),
            (RNG.normal(size=(int(32000 * 2.17), 2)), 32000),
            (RNG.normal(size=(int(48000 * 2.17), 2)), 48000),
            (RNG.normal(size=(int(48000 * 2.17), 6)), 48000),
        ],
    )
    def test___eq__(self, data, samplerate):
        audio = vocalpy.Audio(data=data, samplerate=samplerate)
        other = vocalpy.Audio(data=data.copy(), samplerate=samplerate)
        assert audio == other

    @pytest.mark.parametrize(
        "data, samplerate",
        [
            (RNG.normal(size=(int(32000 * 2.17))), 32000),
            (RNG.normal(size=(int(32000 * 2.17), 1)), 32000),
            (RNG.normal(size=(int(32000 * 2.17), 2)), 32000),
            (RNG.normal(size=(int(48000 * 2.17), 2)), 48000),
            (RNG.normal(size=(int(48000 * 2.17), 6)), 48000),
        ],
    )
    def test___ne__(self, data, samplerate):
        audio = vocalpy.Audio(data=data, samplerate=samplerate)
        other = vocalpy.Audio(data=data.copy() + 0.001, samplerate=samplerate)
        assert audio != other

    def test_read(self, a_wav_path, tmp_path):
        """Test that :meth:`vocalpy.Spectrogram.read` works as expected.

        To do this we make an audio file "by hand".
        """
        data, samplerate = soundfile.read(a_wav_path)
        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1]

        # to make sure round-tripping works as we'd expect,
        # we first write what we read with soundfile to a temporary file
        # then use `vocalpy.Audio` to read that temporary file
        # and test that it matches what we read directly from the original file
        tmp_wav_path = tmp_path / a_wav_path.name
        soundfile.write(tmp_wav_path, data, samplerate)

        audio = vocalpy.Audio.read(tmp_wav_path)
        assert isinstance(audio, vocalpy.Audio)
        for attr_name, attr_val in zip(("data", "samplerate", "channels"), (data, samplerate, channels)):
            assert hasattr(audio, attr_name)
            if isinstance(attr_val, np.ndarray):
                np.testing.assert_allclose(getattr(audio, attr_name), attr_val)
            else:
                assert getattr(audio, attr_name) == attr_val

    def test_write(self, a_wav_path, tmp_path):
        """Test that :meth:`vocalpy.Audio.write` works as expected.

        To do this we make a spectrogram file "by hand".
        """
        data, samplerate = soundfile.read(a_wav_path)
        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1]

        # to make sure round-tripping works as we'd expect,
        # we first write what we read with soundfile to a temporary file
        # then use `vocalpy.Audio` to read that temporary file
        # and test that it matches what we read directly from the original file
        audio = vocalpy.Audio(data=data, samplerate=samplerate)
        tmp_wav_path = tmp_path / a_wav_path.name
        assert not tmp_wav_path.exists()

        audio.write(tmp_wav_path)

        assert tmp_wav_path.exists()

        audio_loaded = vocalpy.Audio.read(tmp_wav_path)
        assert isinstance(audio_loaded, vocalpy.Audio)
        for attr_name, attr_val in zip(("data", "samplerate", "channels"), (data, samplerate, channels)):
            assert hasattr(audio_loaded, attr_name)
            if isinstance(attr_val, np.ndarray):
                np.testing.assert_allclose(getattr(audio_loaded, attr_name), attr_val)
            else:
                assert getattr(audio_loaded, attr_name) == attr_val

    def test_lazy(self, a_wav_path):
        """Test that :meth:`vocalpy.Spectrogram.read` works as expected.

        To do this we make an audio file "by hand".
        """
        data, samplerate = soundfile.read(a_wav_path)
        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1]

        audio = vocalpy.Audio(path=a_wav_path)
        assert audio._data is None
        assert audio._samplerate is None
        assert audio._channels is None

        _ = audio.data  # triggers lazy-load
        assert np.array_equal(audio._data, data)
        assert audio._samplerate == samplerate
        assert audio._channels == channels

    def test_open(self, a_wav_path):
        data, samplerate = soundfile.read(a_wav_path)
        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1]

        audio = vocalpy.Audio(path=a_wav_path)
        assert audio._data is None
        assert audio._samplerate is None
        assert audio._channels is None

        with audio.open():
            assert np.array_equal(audio._data, data)
            assert audio._samplerate == samplerate
            assert audio._channels == channels

        # check that attributes go back to none after we __exit__ the context
        assert audio._data is None
        assert audio._samplerate is None
        assert audio._channels is None
