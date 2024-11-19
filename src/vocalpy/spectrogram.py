"""Convenience function that generates a spectrogram."""

from __future__ import annotations

from typing import Mapping

import librosa
import numpy as np

from . import spectral
from ._spectrogram.data_type import Spectrogram
from .params import Params
from .sound import Sound

METHODS = [
    "librosa-db",
    "sat-multitaper",
    "soundsig-spectro",
]


def spectrogram(
    sound: Sound,
    n_fft: int = 512,
    hop_length: int = 64,
    method="librosa-db",
    params: Mapping | Params | None = None,
) -> Spectrogram:
    """Get a spectrogram from audio.

    This is a convenience function that takes an instance of :class:`vocalpy.Sound`
    and returns an instance of :class:`vocalpy.Spectrogram`. The
    :attr:`vocalpy.Spectrogram.data` will be a spectral representation
    computed according to the specified `method`.

    Methods
    =======
    * `'librosa-db`': dB-scaled spectrogram
      Equivalent to calling ``S = librosa.STFT(sound.data)``
      and then ``S = librosa.amplitude_to_db(np.abs(S))``.
    * ``'sat-multitaper``: multi-taper spectrogram computed the same way
      that the Sound Analysis Toolbox for Matlab (SAT) does.
    * ``'soundsig-spectro``: dB-scaled spectrogram
      computed with Gaussian window; replicates the result of
      the method ``soundsig.BioSound.spectroCalc`` in the
      ``soundsig`` package.

    Parameters
    ----------
    sound : vocalpy.Sound
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
    params : vocalpy.Params, mapping, None
        A dictionary-like mapping from function parameter names
        to argument values.

    Returns
    -------
    spect : vocalpy.Spectrogram
        A :class:`vocalpy.Spectrogram` instance
        computed according to `method`
    """
    if not isinstance(sound, Sound):
        raise TypeError(
            f"audio must be an instance of `vocalpy.Sound` but was: {type(sound)}"
        )

    if method not in METHODS:
        raise ValueError(
            f"Invalid `method`: {method}.\n" f"Valid methods are: {METHODS}\n"
        )

    if method == "librosa-db":
        S = librosa.stft(sound.data, n_fft=n_fft, hop_length=hop_length)
        S = librosa.amplitude_to_db(np.abs(S))
        t = librosa.frames_to_time(
            frames=np.arange(S.shape[-1]),
            sr=sound.samplerate,
            hop_length=hop_length,
        )
        f = librosa.fft_frequencies(sr=sound.samplerate, n_fft=n_fft)
        spect = Spectrogram(data=S, frequencies=f, times=t)
    elif method == "sat-multitaper":
        spect: Spectrogram = spectral.sat_multitaper(sound, n_fft, hop_length)
    elif method == "soundsig-spectro":
        spect: Spectrogram = spectral.soundsig_spectro(
            sound, n_fft, hop_length, **params
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return spect
