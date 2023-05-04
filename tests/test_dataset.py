import pytest

import vocalpy


class TestDataset:
    def test_init_with_audio_cbin_annot_notmat(self, audio_cbin_annot_notmat_root):
        audio_paths = vocalpy.paths.from_dir(audio_cbin_annot_notmat_root, "cbin")
        audio_files = [vocalpy.AudioFile(path=audio_path) for audio_path in audio_paths]

        annot_paths = vocalpy.paths.from_dir(audio_cbin_annot_notmat_root, ".not.mat")
        annot_files = [vocalpy.AnnotationFile(path=annot_path) for annot_path in annot_paths]

        files = []
        for file in audio_files + annot_files:
            files.append(
                vocalpy.DatasetFile(
                    file=file,
                )
            )

        dataset = vocalpy.Dataset(files=files)

        assert isinstance(dataset, vocalpy.Dataset)
        assert dataset.files == files

    def test_init_with_spect_mat_annot_yarden_root(self, spect_mat_annot_yarden_root):
        spect_paths = vocalpy.paths.from_dir(spect_mat_annot_yarden_root, ".spect.mat")
        spect_files = [vocalpy.SpectrogramFile(path=spect_path) for spect_path in spect_paths]

        annot_paths = vocalpy.paths.from_dir(spect_mat_annot_yarden_root, ".not.mat")
        annot_files = [vocalpy.AnnotationFile(path=annot_path) for annot_path in annot_paths]

        files = []
        for file in spect_files + annot_files:
            files.append(
                vocalpy.DatasetFile(
                    file=file,
                )
            )

        dataset = vocalpy.Dataset(files=files)

        assert isinstance(dataset, vocalpy.Dataset)
        assert dataset.files == files
