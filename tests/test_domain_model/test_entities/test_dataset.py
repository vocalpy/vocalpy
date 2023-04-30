import pytest

import vocalpy


class TestDataset:

    def test_audio_and_spect_none_raises(self):
        # TODO: rewrite to test that not all DatasetFiles raises
        """Test that __attrs_post_init__ raises an error when both audio_files and spect_files are None (default)"""
        with pytest.raises(ValueError):
            vocalpy.domain_model.entities.Dataset()

    def test_init_with_audio_cbin_annot_notmat(self, audio_cbin_annot_notmat_root):
        audio_paths = vocalpy.paths.from_dir(audio_cbin_annot_notmat_root, 'cbin')
        audio_files = [
            vocalpy.domain_model.entities.AudioFile(path=audio_path)
            for audio_path in audio_paths
        ]

        annot_paths = vocalpy.paths.from_dir(audio_cbin_annot_notmat_root, '.not.mat')
        annot_files = [
            vocalpy.domain_model.entities.AnnotationFile(path=annot_path)
            for annot_path in annot_paths
        ]

        files = []
        for file in audio_files + annot_files:
            files.append(
                vocalpy.domain_model.entities.DatasetFile(
                    file=file,
                    file_type=vocalpy.domain_model.entities.DatasetFileTypeEnum[file.__class__.__name]
                )
            )

        dataset = vocalpy.Dataset(
            files=files
        )

        assert isinstance(dataset, vocalpy.Dataset)
        assert dataset.files == files

    def test_init_with_spect_mat_annot_yarden_root(self, spect_mat_annot_yarden_root):
        spect_paths = vocalpy.paths.from_dir(spect_mat_annot_yarden_root, '.spect.mat')
        spect_files = [
            vocalpy.domain_model.entities.SpectrogramFile(path=spect_path)
            for spect_path in spect_paths
        ]

        annot_paths = vocalpy.paths.from_dir(spect_mat_annot_yarden_root, '.not.mat')
        annot_files = [
            vocalpy.domain_model.entities.AnnotationFile(path=annot_path)
            for annot_path in annot_paths
        ]

        files = []
        for file in spect_files + annot_files:
            files.append(
                vocalpy.domain_model.entities.DatasetFile(
                    file=file,
                    file_type=vocalpy.domain_model.entities.DatasetFileTypeEnum[file.__class__.__name]
                )
            )

        dataset = vocalpy.Dataset(
            files=files
        )

        assert isinstance(dataset, vocalpy.Dataset)
        assert dataset.files == files
