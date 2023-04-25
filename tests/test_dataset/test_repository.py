import pathlib

import pytest

import vocalpy


class TestAudioFile:
    # TODO: parametrize better to test different audio files?
    def test_init(self, an_audio_path):
        """Test that we can initialize an AudioFile instance. A smoke test."""
        audio_file = vocalpy.dataset.AudioFile(path=an_audio_path)
        assert isinstance(audio_file, vocalpy.dataset.AudioFile)
        assert hasattr(audio_file, 'path')
        assert isinstance(
            getattr(audio_file, 'path'), pathlib.Path
        )
        assert getattr(audio_file, 'path') == an_audio_path

    def test_init_converter(self, an_audio_path):
        """Test that initialization converts a string to a path as expected"""
        an_audio_path_str = str(an_audio_path)
        audio_file = vocalpy.dataset.AudioFile(path=an_audio_path_str)
        assert isinstance(audio_file, vocalpy.dataset.AudioFile)
        assert hasattr(audio_file, 'path')
        assert isinstance(
            getattr(audio_file, 'path'), pathlib.Path
        )
        assert getattr(audio_file, 'path') == an_audio_path


class TestSpectrogramFile:
    def test_init(self, an_npz_spect_path):
        spect_file = vocalpy.dataset.SpectrogramFile(path=an_npz_spect_path)
        assert isinstance(spect_file, vocalpy.dataset.SpectrogramFile)
        assert hasattr(spect_file, 'path')
        assert isinstance(
            getattr(spect_file, 'path'), pathlib.Path
        )
        assert getattr(spect_file, 'path') == an_npz_spect_path

    def test_init_converter(self, an_npz_spect_path):
        an_npz_spect_path_str = str(an_npz_spect_path)
        spect_file = vocalpy.dataset.SpectrogramFile(path=an_npz_spect_path_str)
        assert isinstance(spect_file, vocalpy.dataset.SpectrogramFile)
        assert hasattr(spect_file, 'path')
        assert isinstance(
            getattr(spect_file, 'path'), pathlib.Path
        )
        assert getattr(spect_file, 'path') == an_npz_spect_path

    def test_with_spect_params(self, an_npz_spect_path, default_spect_params):
        spect_file = vocalpy.dataset.SpectrogramFile(path=an_npz_spect_path,
                                                     spectrogram_parameters=default_spect_params)
        assert isinstance(spect_file, vocalpy.dataset.SpectrogramFile)
        for attr_name, attr_type, attr_val in zip(
            ('path', 'spectrogram_parameters'),
            (pathlib.Path, vocalpy.dataset.SpectrogramParameters),
            (an_npz_spect_path, default_spect_params),
        ):
            assert hasattr(spect_file, attr_name)
            assert isinstance(
                getattr(spect_file, attr_name), attr_type
            )
            assert getattr(spect_file, attr_name) == attr_val


class TestAnnotationFile:

    def test_init_cbin_notmat_pairs(self, a_cbin_notmat_pair):
        """Test we can initialize an AnnotationFile,
        for the case where one annotation file
        annotates one audio file"""
        audio_path, annot_path = a_cbin_notmat_pair
        audio_file = vocalpy.dataset.AudioFile(path=audio_path)
        annot_file = vocalpy.dataset.AnnotationFile(
            path=annot_path,
            annotates=audio_file
        )
        assert isinstance(annot_file, vocalpy.dataset.AnnotationFile)
        for attr_name, attr_type, attr_val in zip(
                ('path', 'annotates'),
                (pathlib.Path, vocalpy.dataset.AudioFile),
                (annot_path, audio_file)
        ):
            assert hasattr(annot_file, attr_name)
            assert isinstance(getattr(annot_file, attr_name), attr_type)
            assert getattr(annot_file, attr_name) == attr_val

    def test_init_list_of_wav(self, annot_file_koumura, audio_list_wav):
        audio_files = [
            vocalpy.dataset.AudioFile(path=wav_path)
            for wav_path in audio_list_wav
        ]
        annot_file = vocalpy.dataset.AnnotationFile(
            path=annot_file_koumura,
            annotates=audio_files
        )
        assert isinstance(annot_file, vocalpy.dataset.AnnotationFile)
        for attr_name, attr_type, attr_val in zip(
                ('path', 'annotates'),
                (pathlib.Path, list),
                (annot_file_koumura, audio_files)
        ):
            assert hasattr(annot_file, attr_name)
            assert isinstance(getattr(annot_file, attr_name), attr_type)
            assert getattr(annot_file, attr_name) == attr_val

    def test_init_list_of_spect(self, annot_file_yarden, spect_list_mat):
        # TODO: deal with the fact we have .mat files here
        spect_files = [
            vocalpy.dataset.SpectrogramFile(path=mat_spect_path)
            for mat_spect_path in spect_list_mat
        ]
        annot_file = vocalpy.dataset.AnnotationFile(
            path=annot_file_yarden,
            annotates=spect_files
        )
        assert isinstance(annot_file, vocalpy.dataset.AnnotationFile)
        for attr_name, attr_type, attr_val in zip(
                ('path', 'annotates'),
                (pathlib.Path, list),
                (annot_file_yarden, spect_files)
        ):
            assert hasattr(annot_file, attr_name)
            assert isinstance(getattr(annot_file, attr_name), attr_type)
            assert getattr(annot_file, attr_name) == attr_val

    def test_raises(self, annot_file_koumura, audio_list_wav, spect_list_mat):
        audio_files = [
            vocalpy.dataset.AudioFile(path=wav_path)
            for wav_path in audio_list_wav
        ]
        spect_files = [
            vocalpy.dataset.SpectrogramFile(path=mat_spect_path)
            for mat_spect_path in spect_list_mat
        ]
        mixed_list = audio_files + spect_files

        with pytest.raises(TypeError):
            annot_file = vocalpy.dataset.AnnotationFile(
                path=annot_file_koumura,
                annotates=mixed_list
            )


class TestDataset:

    def test_audio_and_spect_none_raises(self):
        """Test that __attrs_post_init__ raises an error when both audio_files and spect_files are None (default)"""
        with pytest.raises(ValueError):
            vocalpy.dataset.Dataset()

    def test_init_with_audio_cbin_annot_notmat(self, audio_cbin_annot_notmat_root):
        audio_paths = vocalpy.paths.from_dir(audio_cbin_annot_notmat_root, 'cbin')
        audio_files = [
            vocalpy.dataset.AudioFile(path=audio_path)
            for audio_path in audio_paths
        ]

        annot_paths = vocalpy.paths.from_dir(audio_cbin_annot_notmat_root, '.not.mat')
        annot_files = [
            vocalpy.dataset.AnnotationFile(path=annot_path)
            for annot_path in annot_paths
        ]

        dataset = vocalpy.Dataset(
            audio_files=audio_files,
            annotation_files=annot_files,
        )

        assert isinstance(dataset, vocalpy.Dataset)
        assert dataset.audio_files == audio_files
        assert dataset.annotation_files == annot_files

    def test_init_with_spect_mat_annot_yarden_root(self, spect_mat_annot_yarden_root):
        audio_paths = vocalpy.paths.from_dir(audio_cbin_annot_notmat_root, 'cbin')
        audio_files = [
            vocalpy.dataset.AudioFile(path=audio_path)
            for audio_path in audio_paths
        ]

        annot_paths = vocalpy.paths.from_dir(audio_cbin_annot_notmat_root, '.not.mat')
        annot_files = [
            vocalpy.dataset.AnnotationFile(path=annot_path)
            for annot_path in annot_paths
        ]

        dataset = vocalpy.Dataset(
            audio_files=audio_files,
            annotation_files=annot_files,
        )

        assert isinstance(dataset, vocalpy.Dataset)
        assert dataset.audio_files == audio_files
        assert dataset.annotation_files == annot_files
