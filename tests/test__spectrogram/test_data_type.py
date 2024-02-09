import numpy as np
import pytest

import vocalpy

rng = np.random.default_rng()
N_F, N_T = 256, 1000
FS = 32000  # sampling frequency
DATA = rng.normal(size=(N_F, N_T))
FREQS = np.linspace(0, 10000, N_F)
TIMES = np.arange(N_T) / FS


class TestSpectrogram:

    @pytest.mark.parametrize(
        'data, frequencies, times',
        [
            (rng.normal(size=(N_F, N_T)), np.linspace(0, 10000, N_F), np.arange(N_T) / FS),
            (rng.normal(size=(1, N_F, N_T)), np.linspace(0, 10000, N_F), np.arange(N_T) / FS),
        ]
    )
    def test_init(self, data, frequencies, times):
        """Test that we can initialize a :class:`vocalpy.Sound` instance."""
        spect = vocalpy.Spectrogram(data=data, frequencies=frequencies, times=times)
        assert isinstance(spect, vocalpy.Spectrogram)
        assert spect.data.ndim == 3

        for attr_name, attr_val in zip(("data", "frequencies", "times"), (data, frequencies, times)):
            assert hasattr(spect, attr_name)
            if attr_name == "data":
                if attr_val.ndim == 2:
                    np.testing.assert_allclose(
                        getattr(spect, attr_name), attr_val[np.newaxis, ...]
                    )
                else:
                    np.testing.assert_allclose(
                        getattr(spect, attr_name), attr_val
                    )
            else:
                assert getattr(spect, attr_name) is attr_val

    @pytest.mark.parametrize(
        "data, times, frequencies, expected_exception",
        [
            # ``data`` is not a Numpy array
            (DATA.tolist(), TIMES, FREQS, TypeError),
            # ``data`` is not 2 dimensional
            (DATA.ravel(), TIMES, FREQS, ValueError),
            # ``data`` has more than 3 dimensions
            (DATA[np.newaxis, np.newaxis, :, :], TIMES, FREQS, ValueError),
            # ``times`` is not a Numpy array
            (DATA, TIMES.tolist(), FREQS, TypeError),
            # ``times`` is not 1 dimensional
            (DATA, TIMES[:, np.newaxis], FREQS, ValueError),
            # ``frequencies`` is not a Numpy array
            (DATA, TIMES, FREQS.tolist(), TypeError),
            # ``frequencies`` is not 1 dimensional
            (DATA, TIMES, FREQS[:, np.newaxis], ValueError),
            # ``data.shape[0]`` and ``freqencies.shape[0]`` don't match
            (DATA, TIMES, FREQS[:-5], ValueError),
            # ``data.shape[1]`` and ``times.shape[0]`` don't match
            (DATA, TIMES[:-5], FREQS, ValueError),
        ],
    )
    def test_init_raises(self, data, times, frequencies, expected_exception):
        """Test that :class:`vocalpy.Spectrogram` raises expected errors"""
        with pytest.raises(expected_exception):
            vocalpy.Spectrogram(data=data, frequencies=frequencies, times=times)

    def test_asdict(self):
        spect = vocalpy.Spectrogram(data=DATA, frequencies=FREQS, times=TIMES)
        assert isinstance(spect, vocalpy.Spectrogram)

        asdict = spect.asdict()
        assert isinstance(asdict, dict)

        for attr_name, attr_val in zip(("data", "frequencies", "times"), (DATA, FREQS, TIMES)):
            assert attr_name in asdict
            if attr_name == "data":
                if attr_val.ndim == 2:
                    np.testing.assert_allclose(
                        asdict[attr_name], attr_val[np.newaxis, ...]
                    )
                else:
                    np.testing.assert_allclose(
                        asdict[attr_name], attr_val
                    )
            else:
                assert asdict[attr_name] is attr_val

    def test___eq__(self):
        spect = vocalpy.Spectrogram(data=DATA, frequencies=FREQS, times=TIMES)
        other = vocalpy.Spectrogram(data=DATA.copy(), frequencies=FREQS.copy(), times=TIMES.copy())
        assert spect == other

    def test___ne__(self):
        spect = vocalpy.Spectrogram(data=DATA, frequencies=FREQS, times=TIMES)
        other = vocalpy.Spectrogram(data=DATA.copy() + 0.001, frequencies=FREQS.copy(), times=TIMES.copy())
        assert spect != other

    @pytest.mark.parametrize(
        'data, frequencies, times',
        [
            (rng.normal(size=(N_F, N_T)), np.linspace(0, 10000, N_F), np.arange(N_T) / FS),
            (rng.normal(size=(1, N_F, N_T)), np.linspace(0, 10000, N_F), np.arange(N_T) / FS),
        ]
    )
    def test_read(self, tmp_path, data, frequencies, times):
        """Test that :meth:`vocalpy.Spectrogram.read` works as expected.

        To do this we make a spectrogram file "by hand".
        """
        if data.ndim == 3:
            spect_dict = {"data": data, "frequencies": frequencies, "times": times}
        elif data.ndim == 2:
            spect_dict = {"data": data[np.newaxis, ...], "frequencies": frequencies, "times": times}
        path = tmp_path / "spect.npz"
        np.savez(path, **spect_dict)

        spect = vocalpy.Spectrogram.read(path)
        assert isinstance(spect, vocalpy.Spectrogram)
        assert spect.data.ndim == 3
        for attr_name, attr_val in zip(("data", "frequencies", "times"), (data, frequencies, times)):
            assert hasattr(spect, attr_name)
            if attr_name == "data" and attr_val.ndim == 2:
                    np.testing.assert_allclose(
                        getattr(spect, attr_name), attr_val[np.newaxis, ...]
                    )
            else:
                np.testing.assert_allclose(
                    getattr(spect, attr_name), attr_val
                )

    @pytest.mark.parametrize(
        'data, frequencies, times',
        [
            (rng.normal(size=(N_F, N_T)), np.linspace(0, 10000, N_F), np.arange(N_T) / FS),
            (rng.normal(size=(1, N_F, N_T)), np.linspace(0, 10000, N_F), np.arange(N_T) / FS),
        ]
    )
    def test_write(self, tmp_path, data, frequencies, times):
        """Test that :meth:`vocalpy.Spectrogram.write` works as expected.

        To do this we make a spectrogram file "by hand".
        """
        spect = vocalpy.Spectrogram(data=data, frequencies=frequencies, times=times)
        assert spect.data.ndim == 3
        path = tmp_path / "spect.npz"

        spect_file = spect.write(path)
        assert path.exists()
        assert isinstance(spect_file, vocalpy.SpectrogramFile)
        assert spect_file.path.name.endswith(".npz")

        spect_loaded = vocalpy.Spectrogram.read(path)
        assert isinstance(spect_loaded, vocalpy.Spectrogram)
        for attr_name, attr_val in zip(("data", "frequencies", "times"), (data, frequencies, times)):
            assert hasattr(spect, attr_name)
            if attr_name == "data" and attr_val.ndim == 2:
                    np.testing.assert_allclose(
                        getattr(spect, attr_name), attr_val[np.newaxis, ...]
                    )
            else:
                np.testing.assert_allclose(
                    getattr(spect, attr_name), attr_val
                )

    @pytest.mark.parametrize(
        'data, frequencies, times',
        [
            (rng.normal(size=(N_F, N_T)), np.linspace(0, 10000, N_F), np.arange(N_T) / FS),
            (rng.normal(size=(1, N_F, N_T)), np.linspace(0, 10000, N_F), np.arange(N_T) / FS),
            (rng.normal(size=(3, N_F, N_T)), np.linspace(0, 10000, N_F), np.arange(N_T) / FS),
        ]
    )
    def test___iter__(self, data, frequencies, times):
        spect = vocalpy.Spectrogram(data=data, frequencies=frequencies, times=times)
        spect_channels = [
            spect_ for spect_ in spect
        ]
        assert all(
            [isinstance(spect_, vocalpy.Spectrogram)
             for spect_ in spect_channels]
        )
        for channel, spect_channel in enumerate(spect_channels):
            np.testing.assert_allclose(
                spect_channel.data, spect.data[channel][np.newaxis, ...]
            )
        assert len(spect_channels) == spect.data.shape[0]

    @pytest.mark.parametrize(
        'data, frequencies, times, key',
        [
            (rng.normal(size=(N_F, N_T)), np.linspace(0, 10000, N_F), np.arange(N_T) / FS, 0),
            (rng.normal(size=(1, N_F, N_T)), np.linspace(0, 10000, N_F), np.arange(N_T) / FS, 0),
            (rng.normal(size=(3, N_F, N_T)), np.linspace(0, 10000, N_F), np.arange(N_T) / FS, slice(None, 2)),
        ]
    )
    def test___getitem__(self, data, frequencies, times, key):
        spect = vocalpy.Spectrogram(data=data, frequencies=frequencies, times=times)
        spect_channel = spect[key]
        assert isinstance(spect_channel, vocalpy.Spectrogram)
        if isinstance(key, int):
            assert spect_channel.data.shape[0] == 1
            np.testing.assert_allclose(
                spect_channel.data, spect.data[key][np.newaxis, ...]
            )
        elif isinstance(key, slice):
            sliced = spect.data[key]
            assert spect_channel.data.shape[0] == sliced.shape[0]
            np.testing.assert_allclose(
                spect_channel.data, sliced
            )

    @pytest.mark.parametrize(
        'data, frequencies, times, key',
        [
            (rng.normal(size=(N_F, N_T)), np.linspace(0, 10000, N_F), np.arange(N_T) / FS, 1),
            (rng.normal(size=(1, N_F, N_T)), np.linspace(0, 10000, N_F), np.arange(N_T) / FS, 1),
            (rng.normal(size=(3, N_F, N_T)), np.linspace(0, 10000, N_F), np.arange(N_T) / FS, 5),
        ]
    )
    def test___getitem__raises(self, data, frequencies, times, key):
        spect = vocalpy.Spectrogram(data=data, frequencies=frequencies, times=times)
        with pytest.raises(IndexError):
            _ = spect[key]
