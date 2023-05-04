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
    def test_init(self):
        """Test that we can initialize a :class:`vocalpy.Audio` instance."""
        spect = vocalpy.Spectrogram(data=DATA, frequencies=FREQS, times=TIMES)
        assert isinstance(spect, vocalpy.Spectrogram)

        for attr_name, attr_val in zip(("data", "frequencies", "times"), (DATA, FREQS, TIMES)):
            assert hasattr(spect, attr_name)
            assert getattr(spect, attr_name) is attr_val

    @pytest.mark.parametrize(
        "data, times, frequencies, expected_exception",
        [
            # ``data`` is not a Numpy array
            (DATA.tolist(), TIMES, FREQS, TypeError),
            # ``data`` is not 2 dimensional
            (DATA.ravel(), TIMES, FREQS, ValueError),
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
            assert asdict[attr_name] is attr_val

    def test___eq__(self):
        spect = vocalpy.Spectrogram(data=DATA, frequencies=FREQS, times=TIMES)
        other = vocalpy.Spectrogram(data=DATA.copy(), frequencies=FREQS.copy(), times=TIMES.copy())
        assert spect == other

    def test___ne__(self):
        spect = vocalpy.Spectrogram(data=DATA, frequencies=FREQS, times=TIMES)
        other = vocalpy.Spectrogram(data=DATA.copy() + 0.001, frequencies=FREQS.copy(), times=TIMES.copy())
        assert spect != other

    def test_read(self, tmp_path):
        """Test that :meth:`vocalpy.Spectrogram.read` works as expected.

        To do this we make a spectrogram file "by hand".
        """
        spect_dict = {"data": DATA, "times": TIMES, "frequencies": FREQS}
        path = tmp_path / "spect.npz"
        np.savez(path, **spect_dict)

        spect = vocalpy.Spectrogram.read(path)
        assert isinstance(spect, vocalpy.Spectrogram)
        for attr_name, attr_val in zip(("data", "frequencies", "times"), (DATA, FREQS, TIMES)):
            assert hasattr(spect, attr_name)
            assert np.array_equal(getattr(spect, attr_name), attr_val)

    def test_write(self, tmp_path):
        """Test that :meth:`vocalpy.Spectrogram.write` works as expected.

        To do this we make a spectrogram file "by hand".
        """
        spect = vocalpy.Spectrogram(data=DATA, frequencies=FREQS, times=TIMES)
        path = tmp_path / "spect.npz"

        spect.write(path)
        assert path.exists()

        spect_loaded = vocalpy.Spectrogram.read(path)
        assert isinstance(spect_loaded, vocalpy.Spectrogram)
        for attr_name, attr_val in zip(("data", "frequencies", "times"), (DATA, FREQS, TIMES)):
            assert hasattr(spect_loaded, attr_name)
            assert np.array_equal(getattr(spect_loaded, attr_name), attr_val)
