"""Convenience function that generates a spectrogram."""
from __future__ import annotations

import librosa
import numpy as np

from ._spectrogram.data_type import Spectrogram
from .audio import Audio


# TODO: add 'SAT' method and 'biosound' method
def spectrogram(audio: Audio, n_fft: int = 512, hop_length: int = 64, method="librosa-db", **kwargs) -> Spectrogram:
    """Get a spectrogram from audio.

    This is a convenience function that takes an instance of :class:`vocalpy.Audio`
    and returns an instance of :class:`vocalpy.Spectrogram`. The
    :attr:`vocalpy.Spectrogram.data` will be a spectral representation
    computed according to the specified `method`.

    Methods
    =======
    * `'librosa-db`': equivalent to calling ``S = librosa.STFT(audio.data)``
       and then ``S = librosa.amplitude_to_db(np.abs(S))``.

    Parameters
    ----------
    audio : vocalpy.Audio
        Audio used to compute spectrogram.
    n_fft : int
        Length of the frame used for the Fast Fourier Transform,
        in number of audio samples. Default is 512.
    hop_length : int
        Number of audio samples to "hop" for each frame
        whe computing the Fast Fourier Transform.
        Smaller values increase the number of columns in the spectrogram,
        without affecting the frequency resolution of the STFT.
    method : str
        Name  of method.
        Default is `'librosa-db'`.

    Returns
    -------
    spect : vocalpy.Spectrogram
        A :class:`vocalpy.Spectrogram` instance
        computed according to `method`
    """
    if not isinstance(audio, Audio):
        raise TypeError(f"audio must be an instance of `vocalpy.Audio` but was: {type(audio)}")
    if method == "librosa-db":
        S = librosa.stft(audio.data, n_fft=n_fft, hop_length=hop_length)
        S = librosa.amplitude_to_db(np.abs(S))
        t = librosa.frames_to_time(frames=np.arange(S.shape[-1]), sr=audio.samplerate, hop_length=hop_length)
        f = librosa.fft_frequencies(sr=audio.samplerate, n_fft=n_fft)
        return Spectrogram(
            data=S,
            frequencies=f,
            times=t,
            audio_path=audio.path,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
