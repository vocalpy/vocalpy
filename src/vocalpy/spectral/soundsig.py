from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fftpack import fft, fftfreq, next_fast_len

if TYPE_CHECKING:
    from vocalpy import Sound, Spectrogram


class GaussianSpectrumEstimator:
    def __init__(self, nstd=6):
        self.nstd = nstd
        self._gauss_window_cache = {}

    def get_frequencies(self, signal_length, sample_rate):
        signal_length = next_fast_len(signal_length)
        freq = fftfreq(signal_length, d=1.0 / sample_rate)
        nz = freq >= 0.0
        return freq[nz]

    def _get_gauss_window(self, nwinlen):
        if nwinlen in self._gauss_window_cache:
            return self._gauss_window_cache[nwinlen]
        else:
            if nwinlen % 2 == 0:
                nwinlen += 1
            hnwinlen = nwinlen // 2
            gauss_t = np.arange(-hnwinlen, hnwinlen + 1, 1.0)
            gauss_std = float(nwinlen) / float(self.nstd)
            gauss_window = np.exp(-(gauss_t**2) / (2.0 * gauss_std**2)) / (
                gauss_std * np.sqrt(2 * np.pi)
            )
            self._gauss_window_cache[nwinlen] = gauss_window
            return gauss_window

    def estimate(self, signal, samplerate):
        nwinlen = len(signal)
        gauss_window = self._get_gauss_window(nwinlen)

        fft_len = next_fast_len(len(signal))
        # window the signal and take the FFT
        windowed_slice = signal[:fft_len] * gauss_window[:fft_len]
        s_fft = fft(windowed_slice, n=fft_len, overwrite_x=True)
        freq = fftfreq(fft_len, d=1.0 / samplerate)
        nz = freq >= 0.0

        return freq[nz], s_fft[nz]


def soundsig_spectro(
    sound: Sound,
    spec_sample_rate: int = 1000,
    freq_spacing: int = 50,
    min_freq: int = 0,
    max_freq: int = 10000,
    nstd: int = 6,
    scale: bool = True,
    scale_val: int | float = 2**15,
    scale_dtype: npt.DTypeLike = np.int16,
) -> Spectrogram:
    # raw string to avoid flake8 complaining about math
    r"""Compute a dB-scaled spectrogram using a Gaussian window.

    Replicates the result of the method :meth:`soundsig.BioSound.spectroCalc`.

    Parameters
    ----------
    sound : vocalpy.Sound
        Sound loaded from a file. Multi-channel is supported.
    spec_sample_rate : int
        Sampling rate for the output spectrogram, in Hz.
        Sets the overlap for the windows in the STFT.
    freq_spacing : int
        The time-frequency scale for the spectrogram, in Hz.
        Determines the width of the Gaussian window.
    min_freq : int
        The minimum frequency to analyze, in Hz.
        The returned :class:`Spectrogram` will only contain
        frequencies :math:`\gte` ``min_freq``.
    max_freq : int
        The maximum frequency to analyze, in Hz.
        The returned :class:`Spectrogram` will only contain
        frequencies :math:`\lte` ``max_freq``.
    nstd : int
        Number of standard deviations of the Gaussian in one window.
    scale : bool
        If True, scale the ``sound.data``.
        Default is True.
        This is needed to replicate the behavior of ``soundsig``,
        which assumes the audio data is loaded as 16-bit integers.
        Since the default for :class:`vocalpy.Sound` is to load sounds
        with a numpy dtype of float64, this function defaults to
        multiplying the ``sound.data`` by 2**15,
        and then casting to the int16 dtype.
        This replicates the behavior of the ``soundsig`` function,
        given data with dtype float64.
        If you have loaded a sound with a dtype of int16,
        then set this to False.
    scale_val :
        Value to multiply the ``sound.data`` by, to scale the data.
        Default is 2**15.
        Only used if ``scale`` is ``True``.
        This is needed to replicate the behavior of ``soundsig``,
        which assumes the audio data is loaded as 16-bit integers.
    scale_dtype : numpy.dtype
        Numpy Dtype to cast ``sound.data`` to, after scaling.
        Default is ``np.int16``.
        Only used if ``scale`` is ``True``.
        This is needed to replicate the behavior of ``soundsig``,
        which assumes the audio data is loaded as 16-bit integers.

    Returns
    -------
    spect : Spectrogram
        A dB-scaled :class:`Spectrogram` from an STFT
        computed with a Gaussian window.
    """
    if scale:
        from .. import Sound

        sound = Sound(
            data=(sound.data * scale_val).astype(scale_dtype),
            samplerate=sound.samplerate,
            path=sound.path,
        )

    # ---- soundsig.sound.spectrogram
    increment = 1.0 / spec_sample_rate
    window_length = nstd / (2.0 * np.pi * freq_spacing)

    # ---- soundsig.timefreq.gaussian_stft
    spectrum_estimator = GaussianSpectrumEstimator(nstd=nstd)

    # ---- soundsig.timefreq.timefreq
    # compute lengths in # of samples
    nwinlen = int(sound.samplerate * window_length)
    if nwinlen % 2 == 0:
        nwinlen += 1
    hnwinlen = nwinlen // 2
    if sound.data.shape[-1] < nwinlen:
        raise ValueError(
            f"Length of sound.data ({sound.data.shape[-1]}) is less than length of window in samples ({nwinlen})."
        )

    # get the values for the frequency axis by estimating the spectrum of a dummy slice
    full_freq = spectrum_estimator.get_frequencies(nwinlen, sound.samplerate)
    freq_index = (full_freq >= min_freq) & (full_freq <= max_freq)
    freq = full_freq[freq_index]
    nfreq = freq_index.sum()

    nincrement = int(np.round(sound.samplerate * increment))

    spect_channels = []
    for channel_data in sound.data:
        # pad the signal with zeros
        zs = np.zeros([len(channel_data) + 2 * hnwinlen])
        zs[hnwinlen:-hnwinlen] = channel_data
        windows = sliding_window_view(zs, nwinlen, axis=0)[::nincrement]
        nwindows = len(windows)

        # take the FFT of each segment, padding with zeros when necessary to keep window length the same
        tf = np.zeros([nfreq, nwindows], dtype="complex")
        for k, window in enumerate(windows):
            spec_freq, est = spectrum_estimator.estimate(
                window, sound.samplerate
            )
            findex = (spec_freq <= max_freq) & (spec_freq >= min_freq)
            tf[:, k] = est[findex]

        spect_channels.append(tf)

    spec = np.array(spect_channels)
    # Note that the desired spectrogram rate could be slightly modified
    t = np.arange(0, nwindows, 1.0) * float(nincrement) / sound.samplerate

    # ---- soundsig.BioSound.spectCalc
    spec = 20 * np.log10(np.abs(spec))

    from vocalpy import Spectrogram  # avoid circular dependencies

    return Spectrogram(data=spec, times=t, frequencies=freq)
