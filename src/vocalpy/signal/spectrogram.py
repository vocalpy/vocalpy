"""functions for making spectrograms

filters adapted from SciPy cookbook
https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
spectrogram adapted from code by Kyle Kastner and Tim Sainburg
https://github.com/timsainb/python_spectrograms_and_inversion
"""
import numpy as np
import scipy.signal


__all__ = [
    'butter_bandpass',
    'butter_bandpass_filter',
    'spectrogram'
]


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y


def spectrogram(data,
                samplerate,
                fft_size=512,
                step_size=64,
                thresh=None,
                transform_type=None,
                freq_cutoffs=None):
    """creates a spectrogram

    Parameters
    ----------
    data : numpy.ndarray
        audio signal
    samplerate : int
        sampling frequency in Hz
    fft_size : int
        size of window for Fast Fourier transform, number of time bins.
    step_size : int
        step size for Fast Fourier transform
    transform_type : str
        one of {'log_spect', 'log_spect_plus_one'}.
        'log_spect' transforms the spectrogram to log(spectrogram), and
        'log_spect_plus_one' does the same thing but adds one to each element.
        Default is None. If None, no transform is applied.
    thresh: int
        threshold minimum power for log spectrogram
    freq_cutoffs : tuple
        of two elements, lower and higher frequencies.

    Return
    ------
    s : numpy.ndarray
        spectrogram
    f : numpy.ndarray
        vector of centers of time bins from spectrogram
    t : numpy.ndarray
        vector of centers of frequency bins from spectrogram
    """
    noverlap = fft_size - step_size

    if freq_cutoffs:
        data = butter_bandpass_filter(data,
                                      freq_cutoffs[0],
                                      freq_cutoffs[1],
                                      samplerate)

    # below only take [:3] from return of specgram because we don't need the image
    f, t, s = scipy.signal.spectrogram(data, samplerate, nperseg=fft_size, noverlap=noverlap)

    if transform_type:
        if transform_type == 'log_spect':
            s /= s.max()  # volume normalize to max 1
            s = np.log10(s)  # take log
            if thresh:
                # I know this is weird, maintaining 'legacy' behavior
                s[s < -thresh] = -thresh
        elif transform_type == 'log_spect_plus_one':
            s = np.log10(s + 1)
            if thresh:
                s[s < thresh] = thresh
    else:
        if thresh:
            s[s < thresh] = thresh  # set anything less than the threshold as the threshold

    if freq_cutoffs:
        f_inds = np.nonzero((f >= freq_cutoffs[0]) &
                            (f < freq_cutoffs[1]))[0]  # returns tuple
        s = s[f_inds, :]
        f = f[f_inds]

    return s, f, t
