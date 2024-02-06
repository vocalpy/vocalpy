import pathlib

import pytest

import vocalpy


class TestAnnotationFile:
    def test_init_cbin_notmat_pairs(self, a_cbin_notmat_pair):
        """Test we can initialize an AnnotationFile,
        for the case where one annotation file
        annotates one audio file"""
        audio_path, annot_path = a_cbin_notmat_pair
        audio_file = vocalpy.AudioFile(path=audio_path)
        annot_file = vocalpy.AnnotationFile(path=annot_path, annotates=audio_file)
        assert isinstance(annot_file, vocalpy.AnnotationFile)
        for attr_name, attr_type, attr_val in zip(
            ("path", "annotates"), (pathlib.Path, vocalpy.AudioFile), (annot_path, audio_file)
        ):
            assert hasattr(annot_file, attr_name)
            assert isinstance(getattr(annot_file, attr_name), attr_type)
            assert getattr(annot_file, attr_name) == attr_val

    def test_init_list_of_wav(self, annot_file_koumura, birdsongrec_wav_list):
        audio_files = [vocalpy.AudioFile(path=wav_path) for wav_path in birdsongrec_wav_list]
        annot_file = vocalpy.AnnotationFile(path=annot_file_koumura, annotates=audio_files)
        assert isinstance(annot_file, vocalpy.AnnotationFile)
        for attr_name, attr_type, attr_val in zip(
            ("path", "annotates"), (pathlib.Path, list), (annot_file_koumura, audio_files)
        ):
            assert hasattr(annot_file, attr_name)
            assert isinstance(getattr(annot_file, attr_name), attr_type)
            assert getattr(annot_file, attr_name) == attr_val

    def test_init_list_of_spect(self, annot_file_yarden, spect_list_mat):
        # TODO: deal with the fact we have .mat files here
        spect_files = [vocalpy.SpectrogramFile(path=mat_spect_path) for mat_spect_path in spect_list_mat]
        annot_file = vocalpy.AnnotationFile(path=annot_file_yarden, annotates=spect_files)
        assert isinstance(annot_file, vocalpy.AnnotationFile)
        for attr_name, attr_type, attr_val in zip(
            ("path", "annotates"), (pathlib.Path, list), (annot_file_yarden, spect_files)
        ):
            assert hasattr(annot_file, attr_name)
            assert isinstance(getattr(annot_file, attr_name), attr_type)
            assert getattr(annot_file, attr_name) == attr_val

    def test_raises(self, annot_file_koumura, birdsongrec_wav_list, spect_list_mat):
        audio_files = [vocalpy.AudioFile(path=wav_path) for wav_path in birdsongrec_wav_list]
        spect_files = [vocalpy.SpectrogramFile(path=mat_spect_path) for mat_spect_path in spect_list_mat]
        mixed_list = audio_files + spect_files

        with pytest.raises(TypeError):
            vocalpy.AnnotationFile(path=annot_file_koumura, annotates=mixed_list)
