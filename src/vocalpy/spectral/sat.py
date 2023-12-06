"""Spectral representations for Sound Analysis Toolbox (SAT)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import librosa
import numpy as np
import scipy.signal.windows

if TYPE_CHECKING:
    from .. import Audio


def sat(audio: Audio, n_fft=400, hop_length=40, freq_range=0.5):
    f = librosa.fft_frequencies(sr=audio.samplerate, n_fft=n_fft)

    # ---- make power spec
    audio_pad = np.pad(audio.data, pad_width=n_fft // 2)
    windows = librosa.util.frame(audio_pad, frame_length=n_fft, hop_length=hop_length, axis=0)
    tapers = scipy.signal.windows.dpss(400, 1.5, Kmax=2)
    windows1 = windows * tapers[0, :]
    windows2 = windows * tapers[1, :]

    spectra1 = np.fft.fft(windows1, n=n_fft)
    spectra2 = np.fft.fft(windows2, n=n_fft)
    power_spectrogram = (np.abs(spectra1) + np.abs(spectra2)) ** 2
    power_spectrogram = power_spectrogram.T[:f.shape[-1], :]

    # make power spectrum into Spectrogram
    t = librosa.frames_to_time(np.arange(windows.shape[0]), sr=audio.samplerate, hop_length=hop_length, n_fft=n_fft)
    from .. import Spectrogram
    power_spectrogram = Spectrogram(data=power_spectrogram, frequencies=f, times=t)

    log_spectra = np.log(spectra1, where=spectra1 > 0)
    cepstrogram = np.fft.ifft(log_spectra, n=n_fft).real
    cepstrogram = cepstrogram.T
    quefrencies = np.array(np.arange(n_fft)) / audio.samplerate

    # freq_range means "use first `freq_range` percent of frequencies"
    max_freq_idx = int(np.floor(f.shape[0] * freq_range))
    max_freq = f[max_freq_idx]

    spectra1 = spectra1[:, :max_freq_idx]
    spectra2 = spectra2[:, :max_freq_idx]
    # time derivative of spectrum
    dSdt = (-spectra1.real * spectra2.real) - (spectra1.imag * spectra2.imag)
    dSdt = dSdt.T
    # frequency derivative of spectrum
    dSdf = (spectra1.imag * spectra2.real) - (spectra1.real * spectra2.imag)
    dSdf = dSdf.T

    return power_spectrogram, cepstrogram, quefrencies, max_freq, dSdt, dSdf
