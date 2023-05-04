import pathlib

import vocalpy


class TestAudioFile:
    # TODO: parametrize better to test different audio files?
    def test_init(self, an_audio_path):
        """Test that we can initialize an AudioFile instance. A smoke test."""
        audio_file = vocalpy.AudioFile(path=an_audio_path)
        assert isinstance(audio_file, vocalpy.AudioFile)
        assert hasattr(audio_file, "path")
        assert isinstance(getattr(audio_file, "path"), pathlib.Path)
        assert getattr(audio_file, "path") == an_audio_path

    def test_init_converter(self, an_audio_path):
        """Test that initialization converts a string to a path as expected"""
        an_audio_path_str = str(an_audio_path)
        audio_file = vocalpy.AudioFile(path=an_audio_path_str)
        assert isinstance(audio_file, vocalpy.AudioFile)
        assert hasattr(audio_file, "path")
        assert isinstance(getattr(audio_file, "path"), pathlib.Path)
        assert getattr(audio_file, "path") == an_audio_path
