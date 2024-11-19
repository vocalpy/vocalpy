"""Function to estimate the time-varying fundamental, from ``soundsig`` [1]_.
Adapted under MIT license.

.. [1] https://github.com/theunissenlab/soundsig
"""

from math import ceil

import numpy as np
import numpy.typing as npt
import scipy.linalg
from scipy.fftpack import dct, fft
from scipy.optimize import leastsq
from scipy.signal import filtfilt, firwin

from .constants import DEFAULT_DT
from .detect_peaks import detect_peaks
from .signal import correlation_function, gaussian_window
from .sound import temporal_envelope


def lpc(signal: npt.NDArray, order: int) -> npt.NDArray:
    """Compute the Linear Prediction Coefficients (LPCs).

    Parameters
    ----------
    signal: numpy.ndarray
        The input signal.
    order : int
        The LPC order. The output will have order + 1 items.

    Returns
    -------

    Notes
    -----
    Returns the order + 1 LPC coefficients for the signal.
    :math:`c = lpc(x, k)` will find the :math:`k + 1` coefficients of a :math:`k` order linear filter:

    :math:`xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]`

    Such that the sum of the squared-error :math:`e[i] = xp[i] - x[i]` is minimized.

    """
    order = int(order)

    if signal.ndim > 1:
        raise ValueError("Array of rank > 1 not supported yet")
    if order > signal.size:
        raise ValueError("Input signal must have a length >= lpc order")

    if order > 0:
        p = order + 1
        r = np.zeros(p, signal.dtype)
        # Number of non-zero values in autocorrelation one needs for p LPC coefficients
        nx = np.min([p, signal.size])
        x = np.correlate(signal, signal, "full")
        r[:nx] = x[signal.size - 1 : signal.size + order]  # noqa : E203
        phi = np.dot(scipy.linalg.inv(scipy.linalg.toeplitz(r[:-1])), -r[1:])
        return np.concatenate(([1.0], phi))
    else:
        return np.ones(1, dtype=signal.dtype)


def syn_spect(b: npt.NDArray, x: npt.NDArray) -> npt.NDArray:
    """Generates a model spectrum made out of gaussian peaks

    Parameters
    ----------
    b : numpy.ndarray
        Spectrum, an estimate of a Gaussian harmonic stack.
    x : numpy.ndarray
        Frequencies for spectrum.

    Returns
    -------
    syn_s : numpy.ndarray
        Model spectrum.
    """
    npeaks = np.size(b) - 1  # First element of b is the sampling rate
    sdpk = 60  # Force 80 hz width

    syn_s = np.zeros(len(x))

    for i in range(npeaks):
        a = b[i + 1]  # To enforce positive peaks only
        syn_s = syn_s + a * np.exp(
            -((x - b[0] * (i + 1)) ** 2) / (2 * sdpk**2)
        )

    return syn_s


def residual_syn(
    b: npt.NDArray, x: npt.NDArray, real_s: npt.NDArray
) -> npt.NDArray:
    """Compute residual between real and synthetic spectrum.

    Parameters
    ----------
    b : numpy.ndarray
        Spectrum, an estimate of a Gaussian harmonic stack.
    x : numpy.ndarray
        Frequencies for spectrum.
    real_s : numpy.ndarray

    Returns
    -------
    residual : numpy.ndarray
        Elementwise difference between ``real_s``
        and synthetic spectrum computed with
        :func:`syn_spect`.
    """
    syn_s = syn_spect(b, x)
    return real_s - syn_s


def estimate_f0(
    data: npt.NDArray,
    samplerate: int,
    dt: float = DEFAULT_DT,
    max_fund: int = 1500,
    min_fund: int = 300,
    low_fc: int = 200,
    high_fc: int = 6000,
    min_saliency: float = 0.5,
    min_formant_freq: int = 500,
    max_formant_bw: int = 500,
    window_formant: float = 0.1,
    method: str = "Stack",
) -> dict:
    """Estimate the fundamental frequency of a complex sound.

    Parameters
    ----------
    data : npt.ndarray
        1-d array of sound data. Must have a length of at least 1024.
    samplerate: int
        Sampling rate.
    dt : numpy.ndarray, optional.
        A vector of time values at which the fundamental will be estimated.
    max_fund : int
        Maximum fundamental frequency. Default is 1500.
    min_fund: int
        Minimum fundamental frequency. Default is 300.
    low_fc : int
        Low frequency cut-off for band-passing the signal prior to auto-correlation.
        Default is 200.
    high_fc : int
        High frequency cut-off for band-passing the signal prior to auto-correlation.
        Default is 6000,
    min_saliency : float
        Threshold in the auto-correlation for minimum saliency -
        Returns NaN for pitch values is saliency is below this number.
        Default is 0.5.
    min_formant_freq: int
        Minimum value of first formant. Default is 500.
    max_formant_bw: int
        Maxminum value of formants bandwith. Default is 500.
    window_formant : float
        Duration of time window for formant calculation, in seconds.
        Includes 5 std of normal window. Default is 0.1,
    method : str
        Method to use to estimate.
        One of ``{"AC", "ACA", "Cep", "Stack"}``.
        ``'AC'``: Peak of the auto-correlation function.
        ``'ACA``: Peak of envelope of auto-correlation function.
        ``'Cep'``: First peak in cepstrum.
        ``'Stack'``: Fitting of harmonic stacks (works well for zebra finches).
        Default is ``'Stack'``.

    Returns
    -------
    sal : numpy.ndarray
        The time varying pitch saliency - a number between 0 and 1
        corresponding to relative size of the first auto-correlation peak
    fund : numpy.ndarray
        The time-varying fundamental in Hz at the same resolution as the spectrogram.
    fund2 : numpy.ndarray
        A second peak in the spectrum - not a multiple of the fundamental a sign of a second voice
    form1 : numpy.ndarray
        The first formant, if it exists
    form2 : numpy.ndarray
        The second formant, if it exists
    form3 : numpy.ndarray
        The third formant, if it exists
    sound_len : int
        Length of sal, fund, fund2, form1, form2, form3
    """
    # Band-pass filtering signal prior to auto-correlation
    sound_len = data.shape[-1]
    nfilt = 1024
    if sound_len < 1024:
        print(
            "Warning in fundEstimator: sound too short for bandpass filtering, len(soundIn)=%d"
            % sound_len
        )
    else:
        # high pass filter the signal
        highpass_filt = firwin(
            nfilt - 1, 2.0 * low_fc / samplerate, pass_zero=False
        )
        padlen = min(sound_len - 10, 3 * len(highpass_filt))
        data = filtfilt(highpass_filt, [1.0], data, padlen=padlen)

        # low pass filter the signal
        lowpass_filt = firwin(nfilt, 2.0 * high_fc / samplerate)
        padlen = min(sound_len - 10, 3 * len(lowpass_filt))
        data = filtfilt(lowpass_filt, [1.0], data, padlen=padlen)

    # Initializations and useful variables
    soundLen = len(data)
    sound_dur = soundLen / samplerate

    npts = int(sound_dur / dt)
    t = np.arange(npts) * dt

    nt = t.shape[-1]
    soundRMS = np.zeros(nt)
    fund = np.zeros(nt)
    fund2 = np.zeros(nt)
    sal = np.zeros(nt)
    form1 = np.zeros(nt)
    form2 = np.zeros(nt)
    form3 = np.zeros(nt)

    #  Calculate the size of the window for the auto-correlation
    alpha = 5  # Number of sd in the Gaussian window
    win_len = int(
        np.fix((2.0 * alpha / min_fund) * samplerate)
    )  # Length of Gaussian window based on min_fund
    if win_len % 2 == 0:
        # Make a symmetric window
        win_len += 1

    # Use 200 ms for LPC Window - make this a parameter at some point
    win_len2 = int(np.fix(window_formant * samplerate))
    if win_len2 % 2 == 0:
        # Make a symmetric window
        win_len2 += 1

    _, w = gaussian_window(win_len, alpha)
    _, w2 = gaussian_window(win_len2, alpha)
    maxlags = int(2 * ceil((float(samplerate) / min_fund)))

    # First calculate the rms in each window
    for it in range(nt):
        tval = t[it]  # Center of window in time
        if tval >= sound_dur:
            continue
        tind = int(np.fix(tval * samplerate))  # Center of window in ind
        tstart = tind - (win_len - 1) // 2
        tend = tind + (win_len - 1) // 2

        if tstart < 0:
            winstart = -tstart
            tstart = 0
        else:
            winstart = 0

        if tend >= sound_len:
            windend = win_len - (tend - sound_len + 1) - 1
            tend = sound_len - 1
        else:
            windend = win_len - 1

        sound_win = data[tstart:tend] * w[winstart:windend]
        soundRMS[it] = np.std(sound_win)

    soundRMSMax = max(soundRMS)

    # Calculate the auto-correlation in windowed segments and obtain 4 guess values of the fundamental
    # fund_corr_guess - guess from the auto-correlation function
    # fund_corr_amp_guess - guess form the amplitude of the auto-correlation function
    # fund_cep_guess - guess from the cepstrum
    # fund_stack_guess - guess taken from a fit of the power spectrum with a harmonic stack,
    # using the fund_cep_guess as a starting point
    #  Current version use fund_stack_guess as the best estimate...

    soundlen = 0
    for it in range(nt):
        fund[it] = float("nan")
        sal[it] = float("nan")
        fund2[it] = float("nan")
        form1[it] = float("nan")
        form2[it] = float("nan")
        form3[it] = float("nan")

        if soundRMS[it] < soundRMSMax * 0.1:
            continue

        soundlen += 1
        tval = t[it]  # Center of window in time
        if (
            tval >= sound_dur
        ):  # This should not happen here because the RMS should be zero
            continue
        tind = int(np.fix(tval * samplerate))  # Center of window in ind
        tstart = tind - (win_len - 1) // 2
        tend = tind + (win_len - 1) // 2

        if tstart < 0:
            winstart = -tstart
            tstart = 0
        else:
            winstart = 0

        if tend >= sound_len:
            windend = win_len - (tend - sound_len + 1) - 1
            tend = sound_len - 1
        else:
            windend = win_len - 1

        tstart2 = tind - (win_len2 - 1) // 2
        tend2 = tind + (win_len2 - 1) // 2

        if tstart2 < 0:
            winstart2 = -tstart2
            tstart2 = 0
        else:
            winstart2 = 0

        if tend2 >= sound_len:
            windend2 = win_len2 - (tend2 - sound_len + 1) - 1
            tend2 = sound_len - 1
        else:
            windend2 = win_len2 - 1

        sound_win = data[tstart:tend] * w[winstart:windend]

        sound_win2 = data[tstart2:tend2] * w2[winstart2:windend2]

        # Apply LPC to get time-varying formants and one additional guess for the fundamental frequency
        A = lpc(sound_win2, 8)  # 8 degree polynomial
        rts = np.roots(A)  # Find the roots of A
        rts = rts[np.imag(rts) >= 0]  # Keep only half of them
        angz = np.arctan2(np.imag(rts), np.real(rts))

        # Calculate the frequencies and the bandwidth of the formants
        frqsFormants = angz * (samplerate / (2 * np.pi))
        indices = np.argsort(frqsFormants)
        bw = -0.5 * (samplerate / (2 * np.pi)) * np.log(np.abs(rts))

        # Keep formants above 500 Hz and with bandwidth < 500
        # This was 1000 for bird calls
        formants = []
        for kk in indices:
            if frqsFormants[kk] > min_formant_freq and bw[kk] < max_formant_bw:
                formants.append(frqsFormants[kk])
        formants = np.array(formants)

        if len(formants) > 0:
            form1[it] = formants[0]
        if len(formants) > 1:
            form2[it] = formants[1]
        if len(formants) > 2:
            form3[it] = formants[2]

        # Calculate the auto-correlation
        lags = np.arange(-maxlags, maxlags + 1, 1)
        autoCorr = correlation_function(sound_win, sound_win, lags)
        ind0 = int(np.where(lags == 0)[0][0])  # need to find lag zero index

        # find peaks
        indPeaksCorr = detect_peaks(autoCorr, mph=autoCorr.max() / 10.0)

        # Eliminate center peak and all peaks too close to middle
        indPeaksCorr = np.delete(
            indPeaksCorr,
            np.where((indPeaksCorr - ind0) < samplerate / max_fund)[0],
        )
        pksCorr = autoCorr[indPeaksCorr]

        # Find max peak
        if len(pksCorr) == 0:
            pitchSaliency = (
                0.1  # 0.1 goes with the detection of peaks greater than max/10
            )
        else:
            indIndMax = np.where(pksCorr == max(pksCorr))[0][0]
            indMax = indPeaksCorr[indIndMax]
            fund_corr_guess = samplerate / abs(lags[indMax])
            pitchSaliency = autoCorr[indMax] / autoCorr[ind0]

        sal[it] = pitchSaliency

        if sal[it] < min_saliency:
            continue

        # Calculate the envelope of the auto-correlation after rectification
        envCorr = temporal_envelope(
            autoCorr, samplerate, cutoff_freq=max_fund, resample_rate=None
        )
        locsEnvCorr = detect_peaks(envCorr, mph=envCorr.max() / 10.0)

        # Find the peak closest to zero
        if locsEnvCorr.size > 1:
            lagdiff = np.abs(locsEnvCorr[0] - ind0)
            indIndEnvMax = 0

            for indtest in range(1, locsEnvCorr.size):
                lagtest = np.abs(locsEnvCorr[indtest] - ind0)
                if lagtest < lagdiff:
                    lagdiff = lagtest
                    indIndEnvMax = indtest

            # Take the first peak after the one closest to zero
            if indIndEnvMax + 2 > len(
                locsEnvCorr
            ):  # No such peak - use data for correlation function
                fund_corr_amp_guess = fund_corr_guess
            else:
                indEnvMax = locsEnvCorr[indIndEnvMax + 1]
                if lags[indEnvMax] == 0:  # This should not happen
                    print(
                        "Error: Max Peak in enveloppe auto-correlation found at zero delay"
                    )
                    fund_corr_amp_guess = fund_corr_guess
                else:
                    fund_corr_amp_guess = samplerate / lags[indEnvMax]
        else:
            fund_corr_amp_guess = fund_corr_guess

        # Calculate power spectrum and cepstrum
        Y = fft(sound_win, n=win_len + 1)
        f = (samplerate / 2.0) * (
            np.array(range(int((win_len + 1) / 2 + 1)), dtype=float)
            / float((win_len + 1) // 2)
        )  # noqa : E203
        fhigh = np.where(f >= high_fc)[0][0]

        powSound = 20.0 * np.log10(
            np.abs(Y[0 : (win_len + 1) // 2 + 1])  # noqa : E203
        )  # This is the power spectrum
        powSoundGood = powSound[0:fhigh]
        maxPow = max(powSoundGood)
        powSoundGood = powSoundGood - maxPow  # Set zero as the peak amplitude
        powSoundGood[powSoundGood < -60] = -60

        # Calculate coarse spectral envelope
        p = np.polyfit(f[0:fhigh], powSoundGood, 3)
        powAmp = np.polyval(p, f[0:fhigh])

        # Cepstrum
        CY = dct(powSoundGood - powAmp, norm="ortho")

        tCY = (
            1000.0 * np.array(range(len(CY))) / samplerate
        )  # Units of Cepstrum in ms
        fCY = np.zeros(tCY.size)
        fCY[1:] = (
            1000.0 / tCY[1:]
        )  # Corresponding fundamental frequency in Hz.
        fCY[0] = samplerate * 2.0  # Nyquist limit not infinity
        lowInd = np.where(fCY < low_fc)[0]
        if lowInd.size > 0:
            flowCY = np.where(fCY < low_fc)[0][0]
        else:
            flowCY = fCY.size

        fhighCY = np.where(fCY < high_fc)[0][0]

        # Find peak of Cepstrum
        indPk = np.where(CY[fhighCY:flowCY] == max(CY[fhighCY:flowCY]))[0][-1]
        indPk = fhighCY + indPk
        fmass = 0
        mass = 0
        indTry = indPk
        while CY[indTry] > 0:
            fmass = fmass + fCY[indTry] * CY[indTry]
            mass = mass + CY[indTry]
            indTry = indTry + 1
            if indTry >= len(CY):
                break

        indTry = indPk - 1
        if indTry >= 0:
            while CY[indTry] > 0:
                fmass = fmass + fCY[indTry] * CY[indTry]
                mass = mass + CY[indTry]
                indTry = indTry - 1
                if indTry < 0:
                    break

        fGuess = fmass / mass

        if fGuess == 0 or np.isnan(fGuess) or np.isinf(fGuess):
            # Failure of cepstral method
            fGuess = fund_corr_guess

        fund_cep_guess = fGuess

        # Force fundamendal to be bounded
        if fund_cep_guess > max_fund:
            i = 2
            while fund_cep_guess > max_fund:
                fund_cep_guess = fGuess / i
                i += 1
        elif fund_cep_guess < min_fund:
            i = 2
            while fund_cep_guess < min_fund:
                fund_cep_guess = fGuess * i
                i += 1

        # Fit Gaussian harmonic stack
        maxPow = max(powSoundGood - powAmp)

        vars = np.concatenate(([fund_corr_guess], np.ones(9) * np.log(maxPow)))
        bout = leastsq(
            residual_syn, vars, args=(f[0:fhigh], powSoundGood - powAmp)
        )
        modelPowCep = syn_spect(bout[0], f[0:fhigh])
        errCep = sum((powSoundGood - powAmp - modelPowCep) ** 2)

        vars = np.concatenate(
            ([fund_corr_guess * 2], np.ones(9) * np.log(maxPow))
        )
        bout2 = leastsq(
            residual_syn, vars, args=(f[0:fhigh], powSoundGood - powAmp)
        )
        modelPowCep2 = syn_spect(bout2[0], f[0:fhigh])
        errCep2 = sum((powSoundGood - powAmp - modelPowCep2) ** 2)

        if errCep2 < errCep:
            bout = bout2
            modelPowCep = modelPowCep2

        fund_stack_guess = bout[0][0]
        if (fund_stack_guess > max_fund) or (fund_stack_guess < min_fund):
            fund_stack_guess = float("nan")

        # Store the result depending on the method chosen
        if method == "AC":
            fund[it] = fund_corr_guess
        elif method == "ACA":
            fund[it] = fund_corr_amp_guess
        elif method == "Cep":
            fund[it] = fund_cep_guess
        elif method == "Stack":
            fund[it] = fund_stack_guess

        if not np.isnan(fund_stack_guess):
            powLeft = powSoundGood - powAmp - modelPowCep
            maxPow2 = max(powLeft)
            if (
                maxPow2 > maxPow * 0.5
            ):  # Possible second peak in central area as indicator of second voice.
                f2 = f[np.where(powLeft == maxPow2)[0][0]]
                if f2 > 1000 and f2 < 4000:
                    if pitchSaliency > min_saliency:
                        fund2[it] = f2

    # Fix formants.

    # Find means in regions where there are two formants

    # Decide whether there is formant 3
    n3 = np.sum(~np.isnan(form3))

    if (
        n3 < 0.1 * nt
    ):  # There are only two formants - fix formant 3 by merging...
        meanf1 = np.mean(form1[~np.isnan(form2)])
        meanf2 = np.mean(form2[~np.isnan(form2)])
        for it in range(nt):
            if ~np.isnan(form3[it]):
                df12 = np.abs(form2[it] - meanf1)
                df23 = np.abs(form3[it] - meanf2)
                if df12 < df23:
                    form1[it] = (form1[it] + form2[it]) / 2.0
                    form2[it] = form3[it]
                    form3[it] = np.nan
                else:
                    form2[it] = (form2[it] + form3[it]) / 2.0
                    form3[it] = np.nan
            else:  # if there is only one figure out if its second or first
                if np.isnan(form2[it]):
                    if ~np.isnan(form1[it]):
                        df11 = np.abs(form1[it] - meanf1)
                        df12 = np.abs(form1[it] - meanf2)
                        if df12 < df11:
                            form2[it] = form1[it]
                            form1[it] = np.nan
    else:
        meanf1 = np.mean(form1[~np.isnan(form3)])
        meanf2 = np.mean(form2[~np.isnan(form3)])
        meanf3 = np.mean(form3[~np.isnan(form3)])
        for it in range(nt):
            if np.isnan(form3[it]):
                if np.isnan(form2[it]):  # there is only one formant found
                    if ~np.isnan(form1[it]):
                        df11 = np.abs(form1[it] - meanf1)
                        df12 = np.abs(form1[it] - meanf2)
                        df13 = np.abs(form1[it] - meanf3)
                        if df13 < np.minimum(df11, df12):
                            form3[it] = form1[it]
                            form1[it] = np.nan
                        elif df12 < np.minimum(df11, df13):
                            form2[it] = form1[it]
                            form1[it] = np.nan
                else:  # two formants are found
                    df22 = np.abs(form2[it] - meanf2)
                    df23 = np.abs(form2[it] - meanf3)
                    if df23 < df22:
                        form3[it] = form2[it]
                        df11 = np.abs(form1[it] - meanf1)
                        df12 = np.abs(form1[it] - meanf2)
                        if df12 < df11:
                            form2[it] = form1[it]
                            form1[it] = np.nan
                        else:
                            form2[it] = np.nan

    return sal, fund, fund2, form1, form2, form3, soundlen
