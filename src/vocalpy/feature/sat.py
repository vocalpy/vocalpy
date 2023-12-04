import librosa
import numpy as np
import numpy.typing as npt
import xarray as xr

from .. import Audio, Spectrogram, spectral


def goodness_of_pitch(cepstrogram: npt.NDArray, quefrencies: npt.NDArray, max_F0: int = 1830) -> npt.NDArray:
    """Calculates the goodness of pitch of each window in a song interval.

    Goodness of pitch is an estimate of the harmonic periodicity of a signal.
    Higher values indicate a more periodic sound (like a harmonic stack), whereas
    lower values indicate less periodic sounds (like noise). Formally, it is the
    peak of the cepstrum of the signal for fundamental frequencies below `max_F0`.

    Returns:
        np.array: array containing the goodness of pitch for each frame in the song interval.
    """
    quefrency_cutoff = 1 / max_F0
    min_quef_idx = np.min(np.argwhere(quefrencies > quefrency_cutoff)) - 1
    max_quef_idx = int(np.floor(len(cepstrogram) / 2))
    return np.max(cepstrogram[min_quef_idx : max_quef_idx, :], axis=0)


def mean_frequency(power_spectrogram: voc.Spectrogram, min_freq: float=380., max_freq: float=11025.) -> npt.NDArray:
    """Calculates the mean frequency of each window in a song interval.

    This is one way to estimate the pitch of a signal.
    It is the center of the distribution of power across frequencies in the signal.
    For another estimate of pitch, see `SongInterval.calc_pitch()`

    Returns:
        np.array: array containing the mean frequency of each frame in the song interval in Hz.
    """
    freq_inds = (power_spectrogram.frequencies > min_freq) & (power_spectrogram.frequencies < max_freq)
    P = power_spectrogram.data[freq_inds, :]
    frequencies = power_spectrogram.frequencies[freq_inds]
    return np.sum(P * frequencies[:, np.newaxis], axis=0) / np.sum(P, axis=0)


#get small number to avoid potential divide by zero errors
EPS = np.finfo(np.double).eps


def frequency_modulation(dSdf: npt.NDArray, dSdt: npt.NDArray) -> npt.NDArray:
    """Calculates the frequency modulation of each window in a song interval.

    Frequency Modulation can be thought of as the slope of frequency traces in a spectrogram. A high
    frequency modulation score is indicative of a sound who's pitch is changing rapidly, or which is
    noisy and has an unstable pitch. A low frequency modulation score indicates that the pitch of a
    sound is stable (like in a flat harmonic stack). This implementation is based on SAP.

    Returns:
        np.array: array containing the frequency modulation of each frame in the song interval.
    """
    return np.arctan(np.max(dSdt, axis=0) / (np.max(dSdt, axis=0) + EPS))


def amplitude_modulation(dSdt: npt.NDArray) -> npt.NDArray:
    """Calculates the amplitude modulation of each window in a song interval.

    Amplitude modulation is a measure of the rate of change of the amplitude of a signal.
    It will be positive at the beginning of a song syllable and negative at the end.

    Returns:
        np.array: array containing the amplitude modulation of each frame in the song interval.
    """
    return np.sum(dSdt, axis=0)


def entropy(power_spectra: npt.NDArray, min_freq: float=380., max_freq: float=11025.) -> npt.NDArray:
    """Calculates the Wiener entropy of each window in a song interval.

    Wiener entropy is a measure of the uniformity of power spread across frequency bands in a frame of audio.
    The output of this function is log-scaled Weiner entropy, which can range in value from 0 to negative
    infinity. A score close to 0 indicates broadly spread power across frequency bands, ie a less structured
    sound like white noise. A large negative score indicates low uniformity across frequency bands, ie a more
    structured sound like a harmonic stack or pure tone.

    Returns:
        np.array: array containing the log-scaled Weiner entropy of each frame in the song interval.
    """
    freq_inds = (power_spectrogram.frequencies > min_freq) & (power_spectrogram.frequencies < max_freq)
    P = power_spectrogram.data[freq_inds, :]
    #calculate entropy for current frame
    sum_log = np.sum(np.log(P), axis=0)
    log_sum = np.log(
        np.sum(P, axis=0) / (
            P.shape[0] - 1
        ))
    return sum_log / (P.shape[0] - 1) - log_sum


def amplitude(power_spectrogram: npt.NDArray, min_freq: float=380., max_freq: float=11025., baseline: int = 70) -> npt.NDArray:
    """Calculates the amplitude of each window in a song interval.

    Amplitude is the volume of a sound in decibels, considering only frequencies above min_frequency.

    Returns:
        np.array: array containing the amplitude of each frame in the song interval in decibels
    """
    freq_inds = (power_spectrogram.frequencies > min_freq) & (power_spectrogram.frequencies < max_freq)
    P = power_spectrogram.data[freq_inds, :]
    return 10 * np.log10(np.sum(P, axis=0)) + baseline


def pitch(y: npt.NDArray, sample_rate: int, min_frequency, fmax_yin: int, hop_length: int = 40) -> npt.NDArray:
    """Estimates the fundamental frequency (or pitch) of each window in a song interval using the yin algorithm.

    For more information on the YIN algorithm for fundamental frequency estimation, please refer to the documentation
    for `librosa.yin()`.

    Returns:
        np.array: array containing the YIN estimated fundamental frequency of each frame in the song interval in Hertz.
    """
    return librosa.yin(audio.data, fmin=min_frequency, fmax=fmax_yin, sr=audio.samplerate, hop_length=hop_length)


def for_similarity():
    pass
