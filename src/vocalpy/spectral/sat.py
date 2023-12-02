import librosa
import numpy as np
import scipy.signal.windows

from .. import Audio, Spectrogram


def sat(audio: Audio, n_fft=400, hop_length=40, freq_range=0.5):
    f = librosa.fft_frequencies(sr=audio.samplerate, n_fft=n_fft)

    # ---- make power spec
    audio_pad = np.pad(audio, pad_width=n_fft // 2)
    windows = librosa.util.frame(audio_pad, frame_length=n_fft, hop_length=hop_length, axis=0)
    tapers = scipy.signal.windows.dpss(400, 1.5, Kmax=2)
    windows1 = windows * tapers[0, :]
    windows2 = windows * tapers[1, :]

    spectra1 = np.fft.fft(windows1, n=n_fft)
    spectra2 = np.fft.fft(windows2, n=n_fft)
    power_spectrum = (np.abs(spectra1) + np.abs(spectra2)) ** 2
    power_spectrum = power_spectrum.T[:f.shape[-1], :]

    # make power spectrum into Spectrogram
    t = librosa.frames_to_time(np.arange(windows.shape[0]), sr=audio.samplerate, hop_length=hop_length, n_fft=n_fft)
    power_spectrogram = Spectrogram(data=power_spectrum, frequencies=f, times=t)

    log_spectrum = np.log(spectra1, where=spectra1 > 0)
    cepstrum = np.fft.ifft(log_spectrum, n=n_fft).real

    # freq_range means "use first `freq_range` percent of frequencies"
    max_freq_idx = int(np.floor(f.shape[0] * freq_range))
    max_freq = f[max_freq_idx]

    spectra1 = spectra1[:, :max_freq_idx]
    spectra2 = spectra2[:, :max_freq_idx]
    # time derivative of spectrum
    dSdt = (-spectra1.real * spectra2.real) - (spectra1.imag * spectra2.imag)
    # frequency derivative of spectrum
    dSdf = (spectra1.imag * spectra2.real) - (spectra1.real * spectra2.imag)

    return power_spectrogram, cepstrum, max_freq, dSdt, dSdf
