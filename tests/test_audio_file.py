import pathlib

import vocalpy


class TestAudioFile:
    # TODO: parametrize better to test different audio files?
    def test_init(self, all_soundfile_paths):
        """Test that we can initialize an AudioFile instance. A smoke test."""
        audio_file = vocalpy.AudioFile(path=all_soundfile_paths)
        assert isinstance(audio_file, vocalpy.AudioFile)
        assert hasattr(audio_file, "path")
        assert isinstance(getattr(audio_file, "path"), pathlib.Path)
        assert getattr(audio_file, "path") == all_soundfile_paths

    def test_init_converter(self, all_soundfile_paths):
        """Test that initialization converts a string to a path as expected"""
        all_soundfile_paths_str = str(all_soundfile_paths)
        audio_file = vocalpy.AudioFile(path=all_soundfile_paths_str)
        assert isinstance(audio_file, vocalpy.AudioFile)
        assert hasattr(audio_file, "path")
        assert isinstance(getattr(audio_file, "path"), pathlib.Path)
        assert getattr(audio_file, "path") == all_soundfile_paths
