"""Convenience function that generates a spectrogram."""
from __future__ import annotations

import librosa
import numpy as np

from .audio import Audio
from ._spectrogram.data_type import Spectrogram


# TODO: add 'SAT' method and 'biosound' method
def spectrogram(audio: Audio, n_fft: int = 512, hop_length: int = 64, method='librosa-db') -> Spectrogram:
    """Generate a spectrogram from audio.

    Parameters
    ----------
    audio :
    n_fft : int
    hop_length : int
    method : str

    Returns
    -------

    """
    if not isinstance(audio, Audio):
        raise TypeError(
            f"audio must be an instance of `vocalpy.Audio` but was: {type(audio)}"
        )
    if method == 'librosa-db':
        S = librosa.stft(audio.data, n_fft=n_fft, hop_length=hop_length)
        S = librosa.amplitude_to_db(np.abs(S))
        t = librosa.frames_to_time(frames=np.arange(S.shape[-1]), sr=audio.samplerate, hop_length=hop_length)
        f = librosa.fft_frequencies(sr=audio.samplerate, n_fft=n_fft)
        return Spectrogram(
            data=S,
            frequencies=f,
            times=t,
        )
    else:
        raise ValueError(
            f"Unknown method: {method}"
        )
