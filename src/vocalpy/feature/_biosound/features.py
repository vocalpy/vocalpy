"""Functions to compute pre-defined acoustic features, as described in [1]_.
Code is adapted from the ``soundsig`` library [2]_, under MIT license.

.. [1] Elie JE and Theunissen FE. "The vocal repertoire of the domesticated zebra finch:
   a data driven approach to decipher the information-bearing acoustic features of communication signals."
   Animal Cognition. 2016. 19(2) 285-315 DOI 10.1007/s10071-015-0933-6
.. [2] https://github.com/theunissenlab/soundsig
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Sequence

import matplotlib.mlab
import numpy as np
import numpy.typing as npt
import xarray as xr

from .constants import DEFAULT_DT
from .fundamental import estimate_f0
from .sound import temporal_envelope

if TYPE_CHECKING:
    from vocalpy import Features, Sound


def temporal_envelope_features(
    data: npt.NDArray,
    samplerate: int,
    cutoff_freq: int = 20,
    amp_sample_rate: int = 1000,
) -> dict:
    """Extract pre-defined acoustic features from temporal envelope of a sound,
    as described in [1]_.

    This is a helper function called by
    :func:`vocalpy.feature.biosound`.
    It replicates the result of calling the ``soundsig.BioSound``
    method ``ampenv``.

    Parameters
    ----------
    data : numpy.ndarray
        The data from a single channel of a :class:`Sound`
    samplerate : int
        The sampling rate of the :class:`Sound`.
    cutoff_freq : int
        The cutoff frequency, in Hz.
        Default is 20, as in [1]_.
    amp_sample_rate: int
        The resampling rate to use when computing the amplitude
        envelope.

    Returns
    -------
    features : dict

    Notes
    -----
    Code is adapted from the ``soundsig`` library [2]_, under MIT license.

    References
    ----------
    .. [1] Elie JE and Theunissen FE. "The vocal repertoire of the domesticated zebra finch:
       a data driven approach to decipher the information-bearing acoustic features of communication signals."
       Animal Cognition. 2016. 19(2) 285-315 DOI 10.1007/s10071-015-0933-6
    .. [2] https://github.com/theunissenlab/soundsig
    """
    amp, tdata = temporal_envelope(
        data,
        samplerate,
        cutoff_freq=cutoff_freq,
        resample_rate=amp_sample_rate,
    )

    ampdata = amp / np.sum(amp)
    meantime = np.sum(tdata * ampdata)
    stdtime = np.sqrt(np.sum(ampdata * ((tdata - meantime) ** 2)))
    skewtime = np.sum(ampdata * (tdata - meantime) ** 3)
    skewtime = skewtime / (stdtime**3)
    kurtosistime = np.sum(ampdata * (tdata - meantime) ** 4)
    kurtosistime = kurtosistime / (stdtime**4)
    indpos = np.where(ampdata > 0)[0]
    entropytime = -np.sum(
        ampdata[indpos] * np.log2(ampdata[indpos])
    ) / np.log2(np.size(indpos))

    return {
        "mean_t": meantime,
        "std_t": stdtime,
        "skew_t": skewtime,
        "kurtosis_t": kurtosistime,
        "entropy_t": entropytime,
        "t_amp": tdata,
        "amp": amp,
        "max_amp": max(amp),
    }


def spectral_envelope_features(
    data: npt.NDArray,
    samplerate: int,
    f_high: int = 10000,
    NFFT=1024,
    noverlap=512,
) -> dict:
    """Extract pre-defined acoustic features from spectral envelope of a sound,
    as described in [1]_.

    This is a helper function called by
    :func:`vocalpy.feature.biosound`.
    It replicates the result of calling the ``soundsig.BioSound``
    method ``spectrum``.

    Parameters
    ----------
    data : numpy.ndarray
        The data from a single channel of a :class:`Sound`
    samplerate : int
        The sampling rate of the :class:`Sound`.
    f_high : int
        Highest frequency to use from spectral envelope.
        Default is 10000, as in [1]_.
    NFFT : int
        Length of FFT window, in number of samples.
        Used to compute power spectrum,
        by calling :func:`matplotlb.mlab.psd`.
        Default is 1024.
    noverlap : int
        Overlap of FFT windows, in number of samples.
        Used to compute power spectrum,
        by calling :func:`matplotlb.mlab.psd`.
        Default is 512.

    Returns
    -------
    features : dict

    Notes
    -----
    Code is adapted from the ``soundsig`` library [2]_, under MIT license.

    References
    ----------
    .. [1] Elie JE and Theunissen FE. "The vocal repertoire of the domesticated zebra finch:
       a data driven approach to decipher the information-bearing acoustic features of communication signals."
       Animal Cognition. 2016. 19(2) 285-315 DOI 10.1007/s10071-015-0933-6
    .. [2] https://github.com/theunissenlab/soundsig
    """
    # Calculates power spectrum and features from power spectrum

    # Need to add argument for window size
    # f_high is the upper bound of the frequency for saving power spectrum
    # nwindow = (1000.0*np.size(soundIn)/samprate)/window_len
    #
    Pxx, Freqs = matplotlib.mlab.psd(
        data, Fs=samplerate, NFFT=NFFT, noverlap=noverlap
    )

    # Find quartile power
    cum_power = np.cumsum(Pxx)
    tot_power = np.sum(Pxx)
    quartile_freq = np.zeros(3, dtype="int")
    quartile_values = [0.25, 0.5, 0.75]
    nfreqs = np.size(cum_power)
    iq = 0
    for ifreq in range(nfreqs):
        if cum_power[ifreq] > quartile_values[iq] * tot_power:
            quartile_freq[iq] = ifreq
            iq = iq + 1
            if iq > 2:
                break

    # Find skewness, kurtosis and entropy for power spectrum below f_high
    ind_fmax = np.where(Freqs > f_high)[0][0]

    # Description of spectral shape
    spectdata = Pxx[0:ind_fmax]
    freqdata = Freqs[0:ind_fmax]
    spectdata = spectdata / np.sum(spectdata)
    meanspect = np.sum(freqdata * spectdata)
    stdspect = np.sqrt(np.sum(spectdata * ((freqdata - meanspect) ** 2)))
    skewspect = np.sum(spectdata * (freqdata - meanspect) ** 3)
    skewspect = skewspect / (stdspect**3)
    kurtosisspect = np.sum(spectdata * (freqdata - meanspect) ** 4)
    kurtosisspect = kurtosisspect / (stdspect**4)
    entropyspect = -np.sum(spectdata * np.log2(spectdata)) / np.log2(ind_fmax)

    return {
        "mean_s": meanspect,
        "std_s": stdspect,
        "skew_s": skewspect,
        "kurtosis_s": kurtosisspect,
        "entropy_s": entropyspect,
        "q1": Freqs[quartile_freq[0]],
        "q2": Freqs[quartile_freq[1]],
        "q3": Freqs[quartile_freq[2]],
        "fpsd": freqdata,
        "psd": spectdata,
    }


def fundamental_features(
    data,
    samplerate,
    dt=DEFAULT_DT,
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
    """Extract features from the time-varying fundamental frequency,
    as described in [1]_.

    This is a helper function called by
    :func:`vocalpy.feature.biosound`.
    It replicates the result of calling the ``soundsig.BioSound``
    method ``fundest``.

    Parameters
    ----------
    data : numpy.ndarray
        The data from a single channel of a :class:`Sound`
    samplerate : int
        The sampling rate of the :class:`Sound`.
    dt : float
        Size of time step in seconds.
        Default is DEFAULT_DT, the time step
        for the spectrogram computed by
        `soundsig.BioSound.spectroCalc`.
    max_fund : int
        Maximum fundamental frequency to analyze, in Hz.
        Default is 1500, as in [1]_.
    min_fund : int
        Minimum fundamental frequency to analyze, in Hz.
        Default is 300, as in [1]_.
    low_fc : int
        Low frequency cut-off for band-passing the signal
        prior to auto-correlation.
        Default is 200, as in [1]_.
    high_fc : int
        High frequency cut-off for band-passing.
        Default is 6000, as in [1]_.
    min_saliency : int
        Threshold in the auto-correlation for minimum saliency.
        Returns NaN for pitch values if saliency is below this number.
        Default is 0.5, as in [1]_.
    min_formant_freq : int
        Minimum value of first formant, in Hz.
        Default is 500, as in [1]_.
    max_formant_bw : int
        Maximum value of formant bandwidth, in Hz..
        Default is 500, as in [1]_.
    window_formant : float
        Time window for formant calculation, in seconds.
        Includes 5 standard deviations of normal window.
        Default is 0.1, as in [1]_.
    method : str
        One of {``'AC'``, ``'ACA'``, ``'Cep'``, or ``'Stack'``}.
        Default is ``'Stack'``, as in [1]_.
        See Notes for detail.

    Returns
    -------
    fund_features : dict
        With the following key-value pairs:
            f0 : np.ndarray
                The time-varying fundamental in Hz,
                at the same resolution as the spectrogram.
            f0_2 : np.ndarray
                A second peak in the spectrum;
                not a multiple of the fundamental, a sign of a second voice.
            F1 : np.ndarray
                The time-varying first formant, if it exists.
            F2 : np.ndarray
                The time-varying second formant, if it exists.
            F3 : np.ndarray
                The time-varying third formant, if it exists.
            fund : np.ndarray
                The average fundamental
            sal : np.ndarray
                The time-varying saliency.
            meansal : np.ndarray
                The average saliency.
            fund2 : np.ndarray
                The average fundamental of the 2nd peak ``f0_2``.
            voice2percent : np.ndarray
                The average percent of presence of a second peak.

    Notes
    -----
    This function implements four methods for estimating the
    time-varying fundamental frequency, specified by the ``method`` argument:
    * ``'AC'``: Peak of the auto-correlation function
    * ``'ACA'``: Peak of the envelope of the auto-correlation function
    * ``'Cep'``: First peak in cepstrum
    * ``'Stack'``: Fitting of harmonic stacks. This is the default, works well for zebra finches.

    Code is adapted from the ``soundsig`` library [2]_, under MIT license.

    References
    ----------
    .. [1] Elie JE and Theunissen FE. "The vocal repertoire of the domesticated zebra finch:
       a data driven approach to decipher the information-bearing acoustic features of communication signals."
       Animal Cognition. 2016. 19(2) 285-315 DOI 10.1007/s10071-015-0933-6
    .. [2] https://github.com/theunissenlab/soundsig
    """
    # Calculate the fundamental, the formants and parameters related to these
    sal, fund, fund2, form1, form2, form3, lenfund = estimate_f0(
        data,
        samplerate,
        dt,
        max_fund,
        min_fund,
        low_fc,
        high_fc,
        min_saliency,
        min_formant_freq,
        max_formant_bw,
        window_formant,
        method=method,
    )
    goodFund = fund[~np.isnan(fund)]
    goodSal = sal[~np.isnan(sal)]
    goodFund2 = fund2[~np.isnan(fund2)]
    if np.size(goodFund) > 0:
        meanfund = np.mean(goodFund)
    else:
        meanfund = np.nan
    meansal = np.mean(goodSal)
    if np.size(goodFund2) > 0:
        pk2 = np.mean(goodFund2)
    else:
        pk2 = np.nan

    if np.size(goodFund) == 0 or np.size(goodFund2) == 0:
        second_v = 0.0
    else:
        second_v = (
            float(np.size(goodFund2)) / float(np.size(goodFund))
        ) * 100.0

    fund_features = {}
    for name, value in zip(
        (
            "f0",
            "f0_2",
            "F1",
            "F2",
            "F3",
            "mean_f0",
            "sal",
            "mean_sal",
            "pk2",
            "second_v",
        ),
        (
            fund,
            fund2,
            form1,
            form2,
            form3,
            meanfund,
            sal,
            meansal,
            pk2,
            second_v,
        ),
    ):
        fund_features[name] = value
    if np.size(goodFund) > 0:
        for name, value in zip(
            ("max_fund", "min_fund", "cv_fund"),
            (np.max(goodFund), np.min(goodFund), np.std(goodFund) / meanfund),
        ):
            fund_features[name] = value

    return fund_features


SoundsigFeatureGroups = Literal["temporal", "spectral", "fundamental"]


SCALAR_FEATURES = {
    "temporal": [
        "mean_t",
        "std_t",
        "skew_t",
        "kurtosis_t",
        "entropy_t",
        "max_amp",
    ],
    "spectral": [
        "mean_s",
        "std_s",
        "skew_s",
        "kurtosis_s",
        "entropy_s",
        "q1",
        "q2",
        "q3",
    ],
    "fundamental": [
        "mean_f0",
        "mean_sal",
        "second_v",
        "pk2",
        "max_fund",
        "min_fund",
        "cv_fund",
    ],
}


def biosound(
    sound: Sound,
    scale: bool = True,
    scale_val: int | float = 2**15,
    scale_dtype: npt.DTypeLike = np.int16,
    ftr_groups: SoundsigFeatureGroups | Sequence[SoundsigFeatureGroups] = (
        "temporal",
        "spectral",
        "fundamental",
    ),
) -> Features:
    """Compute predefined acoustic features (PAFs)
    used to analyze the vocal repertoire of the domesticated zebra finch,
    as described in [1]_.

    Parameters
    ----------
    sound : Sound
        A sound loaded from a file.
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
    features : vocalpy.Features
        A :class:`vocalpy.Features` instance with
        :attr:`~vocalpy.Features.data` attribute that is
        an :class:`xarray.Dataset`,
        where the data variables are the features,
        and the coordinate is the channel.

    Notes
    -----
    Code is adapted from the ``soundsig`` library [2]_, under MIT license.

    References
    ----------
    .. [1] Elie JE and Theunissen FE. "The vocal repertoire of the domesticated zebra finch:
       a data driven approach to decipher the information-bearing acoustic features of communication signals."
       Animal Cognition. 2016. 19(2) 285-315 DOI 10.1007/s10071-015-0933-6
    .. [2] https://github.com/theunissenlab/soundsig
    """
    if isinstance(ftr_groups, (list, tuple)):
        if not all([isinstance(ftr_group, str) for ftr_group in ftr_groups]):
            bad_types = set(
                [
                    type(ftr_group)
                    for ftr_group in ftr_groups
                    if not isinstance(ftr_groups, str)
                ]
            )
            raise TypeError(
                f"`ftr_groups` must be a list or tuple of strings but some items in sequence were not: {bad_types}"
            )
        if not all(
            [
                ftr_group in ("temporal", "spectral", "fundamental")
                for ftr_group in ftr_groups
            ]
        ):
            raise ValueError(
                'All strings in `ftr_groups` must be one of: "temporal", "spectral", "fundamental", '
                f"but got:\n{ftr_groups}"
            )

    elif isinstance(ftr_groups, str):
        if ftr_groups not in ("temporal", "spectral", "fundamental"):
            raise ValueError(
                'Value for `ftr_groups` must be one of: "temporal", "spectral", "fundamental", '
                f"but got:\n{ftr_groups}"
            )
        ftr_groups = (
            ftr_groups,
        )  # so we can write ``if "string" in ftr_groups``

    if scale:
        from ... import Sound

        sound = Sound(
            data=(sound.data * scale_val).astype(scale_dtype),
            samplerate=sound.samplerate,
        )

    features = defaultdict(list)
    for channel_data in sound.data:
        if "temporal" in ftr_groups:
            t_feat = temporal_envelope_features(channel_data, sound.samplerate)
            for ftr_name in SCALAR_FEATURES["temporal"]:
                features[ftr_name].append(t_feat[ftr_name])
        if "spectral" in ftr_groups:
            s_feat = spectral_envelope_features(channel_data, sound.samplerate)
            for ftr_name in SCALAR_FEATURES["spectral"]:
                features[ftr_name].append(s_feat[ftr_name])
        if "fundamental" in ftr_groups:
            f_feat = fundamental_features(channel_data, sound.samplerate)
            for ftr_name in SCALAR_FEATURES["fundamental"]:
                features[ftr_name].append(f_feat[ftr_name])

    for key, val in features.items():
        features[key] = np.array(val)

    channels = np.arange(sound.data.shape[0])
    data = xr.Dataset(
        {
            feature_name: (["channel"], feature_val)
            for feature_name, feature_val in features.items()
        },
        coords={"channel": channels},
    )

    from ... import Features  # avoid circular import

    features = Features(data=data)
    return features
