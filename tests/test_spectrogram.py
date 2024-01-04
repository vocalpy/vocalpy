import vocalpy


def test_spectrogram(a_wav_path):
    audio = vocalpy.Audio.read(a_wav_path)
    spectrogram = vocalpy.spectrogram(audio)
    assert isinstance(spectrogram, vocalpy.Spectrogram)
