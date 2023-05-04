import pathlib

import vocalpy


class TestSpectrogramFile:
    def test_init(self, an_npz_spect_path):
        spect_file = vocalpy.SpectrogramFile(path=an_npz_spect_path)
        assert isinstance(spect_file, vocalpy.SpectrogramFile)
        assert hasattr(spect_file, "path")
        assert isinstance(getattr(spect_file, "path"), pathlib.Path)
        assert getattr(spect_file, "path") == an_npz_spect_path

    def test_init_converter(self, an_npz_spect_path):
        an_npz_spect_path_str = str(an_npz_spect_path)
        spect_file = vocalpy.SpectrogramFile(path=an_npz_spect_path_str)
        assert isinstance(spect_file, vocalpy.SpectrogramFile)
        assert hasattr(spect_file, "path")
        assert isinstance(getattr(spect_file, "path"), pathlib.Path)
        assert getattr(spect_file, "path") == an_npz_spect_path

    def test_with_spect_params(self, an_npz_spect_path, default_spect_params):
        spect_file = vocalpy.SpectrogramFile(path=an_npz_spect_path, spectrogram_parameters=default_spect_params)
        assert isinstance(spect_file, vocalpy.SpectrogramFile)
        for attr_name, attr_type, attr_val in zip(
            ("path", "spectrogram_parameters"),
            (pathlib.Path, vocalpy.SpectrogramParameters),
            (an_npz_spect_path, default_spect_params),
        ):
            assert hasattr(spect_file, attr_name)
            assert isinstance(getattr(spect_file, attr_name), attr_type)
            assert getattr(spect_file, attr_name) == attr_val
