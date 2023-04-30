import pytest

import vocalpy


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
