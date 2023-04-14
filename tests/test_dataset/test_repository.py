import pytest

import vocalpy


class TestAudioFile:
    # TODO: parametrize better to test different audio files?
    def test_init(self, a_cbin_path):
        audio_file = vocalpy.dataset.AudioFile(path=a_cbin_path)
        assert isinstance(audio_file, vocalpy.dataset.AudioFile)
        assert hasattr(audio_file, 'path')
        assert getattr(audio_file, 'path') == a_cbin_path


class TestSpectrogramFile:
    def test_init(self, an_npz_spect_path):
        spect_file = vocalpy.dataset.SpectrogramFile(path=an_npz_spect_path)
        assert isinstance(spect_file, vocalpy.dataset.SpectrogramFile)
        assert hasattr(spect_file, 'path')
        assert getattr(spect_file, 'path') == an_npz_spect_path


class TestDataset:

    def test_audio_and_spect_none_raises(self):
        """Test that __attrs_post_init__ raises an error when both audio_files and spect_files are None (default)"""
        with pytest.raises(ValueError):
            vocalpy.dataset.Dataset()

    def test_with_audio_cbin_annot_notmat(self, audio_cbin_annot_notmat_root):
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

