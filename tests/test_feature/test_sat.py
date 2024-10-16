import numpy as np
import pytest

import vocalpy

from ..fixtures.audio import ALL_ZEBRA_FINCH_WAVS, MULTICHANNEL_FLY_WAV


AUDIO_PATHS = ALL_ZEBRA_FINCH_WAVS + [MULTICHANNEL_FLY_WAV]


@pytest.fixture(params=AUDIO_PATHS)
def sound_to_test_sat_features(request):
    path = request.param
    sound = vocalpy.Sound.read(path)
    return sound


@pytest.mark.parametrize(
    'n_fft, hop_length',
    [
        (400, 40),
    ]
)
def test_goodness_of_pitch(sound_to_test_sat_features, n_fft, hop_length):
    """Test :func:`vocalpy.feature._sat.goodness_of_pitch` returns expected outputs"""
    _, spectra1, spectra2 = vocalpy.spectral.sat._sat_multitaper(sound_to_test_sat_features, n_fft, hop_length)
    cepstrogram, quefrencies = vocalpy.feature._sat._get_cepstral(spectra1, n_fft, sound_to_test_sat_features.samplerate)

    out = vocalpy.feature._sat.goodness_of_pitch(
        cepstrogram, quefrencies,
    )

    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[0] == sound_to_test_sat_features.data.shape[0]
    assert out.shape[-1] == cepstrogram.shape[-1]


@pytest.mark.parametrize(
    'n_fft, hop_length, freq_range',
    [
        (400, 40, 0.5),
    ]
)
def test_mean_frequency(sound_to_test_sat_features, n_fft, hop_length, freq_range):
    """Test :func:`vocalpy.feature._sat.mean_frequency` returns expected outputs"""
    spect = vocalpy.spectral.sat.sat_multitaper(sound_to_test_sat_features, n_fft, hop_length)
    f = spect.frequencies
    max_freq_idx = int(np.floor(f.shape[0] * freq_range))
    max_freq = f[max_freq_idx]

    out = vocalpy.feature._sat.mean_frequency(
        spect, max_freq=max_freq
    )

    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[0] == sound_to_test_sat_features.data.shape[0]
    assert out.shape[-1] == spect.data.shape[-1]
    # sanity check, mean should be less than/equal to max_freq
    assert np.all(
        out[~np.isnan(out)] <= max_freq
    )


@pytest.mark.parametrize(
    'n_fft, hop_length, freq_range',
    [
        (400, 40, 0.5),
    ]
)
def test_frequency_modulation(sound_to_test_sat_features, n_fft, hop_length, freq_range):
    """Test :func:`vocalpy.feature._sat.frequency_modulation` returns expected outputs"""
    power_spectrogram, spectra1, spectra2 = vocalpy.spectral.sat._sat_multitaper(
        sound_to_test_sat_features, n_fft, hop_length
    )
    f = power_spectrogram.frequencies
    max_freq_idx = int(np.floor(f.shape[0] * freq_range))

    dSdt, dSdf = vocalpy.feature._sat._get_spectral_derivatives(spectra1, spectra2, max_freq_idx)
    out = vocalpy.feature._sat.frequency_modulation(
        dSdt, dSdf
    )

    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[-1] == dSdt.shape[-1]


@pytest.mark.parametrize(
    'n_fft, hop_length, freq_range',
    [
        (400, 40, 0.5),
    ]
)
def test_amplitude_modulation(sound_to_test_sat_features, n_fft, hop_length, freq_range):
    """Test :func:`vocalpy.feature._sat.amplitude_modulation` returns expected outputs"""
    spect, spectra1, spectra2 = vocalpy.spectral.sat._sat_multitaper(sound_to_test_sat_features, n_fft, hop_length)
    f = spect.frequencies
    max_freq_idx = int(np.floor(f.shape[0] * freq_range))

    dSdt, _ = vocalpy.feature._sat._get_spectral_derivatives(spectra1, spectra2, max_freq_idx)

    out = vocalpy.feature._sat.amplitude_modulation(
        dSdt
    )

    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[-1] == dSdt.shape[-1]


@pytest.mark.parametrize(
    'n_fft, hop_length, freq_range',
    [
        (400, 40, 0.5),
    ]
)
def test_entropy(sound_to_test_sat_features, n_fft, hop_length, freq_range):
    """Test :func:`vocalpy.feature._sat.entropy` returns expected outputs"""
    power_spectrogram, spectra1, spectra2 = vocalpy.spectral.sat._sat_multitaper(
        sound_to_test_sat_features, n_fft, hop_length
    )
    f = power_spectrogram.frequencies
    max_freq_idx = int(np.floor(f.shape[0] * freq_range))
    max_freq = f[max_freq_idx]

    out = vocalpy.feature._sat.entropy(
        power_spectrogram, max_freq=max_freq
    )

    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[-1] == power_spectrogram.data.shape[-1]


@pytest.mark.parametrize(
    'n_fft, hop_length, freq_range',
    [
        (400, 40, 0.5),
    ]
)
def test_amplitude(sound_to_test_sat_features, n_fft, hop_length, freq_range):
    """Test :func:`vocalpy.feature._sat.amplitude` returns expected outputs"""
    spect = vocalpy.spectral.sat.sat_multitaper(sound_to_test_sat_features, n_fft, hop_length)
    f = spect.frequencies
    max_freq_idx = int(np.floor(f.shape[0] * freq_range))
    max_freq = f[max_freq_idx]

    out = vocalpy.feature._sat.amplitude(
        spect, max_freq=max_freq
    )

    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[-1] == spect.data.shape[-1]


@pytest.mark.parametrize(
    'n_fft, hop_length',
    [
        (400, 40)
    ]
)
def test_pitch(n_fft, hop_length, sound_to_test_sat_features):
    """Test :func:`vocalpy.feature._sat.pitch` returns expected outputs"""

    if sound_to_test_sat_features.samplerate <= 10000:
        out = vocalpy.feature._sat.pitch(
            sound_to_test_sat_features, fmax_yin=5000,
        )
    else:
        out = vocalpy.feature._sat.pitch(
            sound_to_test_sat_features
        )

    # get spectrogram as lazy way to figure out expected number of time bins
    power_spectrogram = vocalpy.spectral.sat.sat_multitaper(sound_to_test_sat_features, n_fft, hop_length)

    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[-1] == power_spectrogram.data.shape[-1]


@pytest.mark.parametrize(
    'n_fft, hop_length',
    [
        (400, 40,)
    ]
)
def test__get_cepstral(sound_to_test_sat_features, n_fft, hop_length):
    _, spectra1, spectra2 = vocalpy.spectral.sat._sat_multitaper(sound_to_test_sat_features, n_fft, hop_length)

    cepstrogram, quefrencies = vocalpy.feature._sat._get_cepstral(
        spectra1, n_fft, sound_to_test_sat_features.samplerate
    )

    assert isinstance(cepstrogram, np.ndarray)
    assert isinstance(quefrencies, np.ndarray)
    assert quefrencies.shape[0] == cepstrogram.shape[1]


@pytest.mark.parametrize(
    'n_fft, hop_length, freq_range',
    [
        (400, 40, 0.5)
    ]
)
def test__get_spectral_derivatives(sound_to_test_sat_features, n_fft, hop_length, freq_range):
    spect, spectra1, spectra2 = vocalpy.spectral.sat._sat_multitaper(sound_to_test_sat_features, n_fft, hop_length)
    f = spect.frequencies
    max_freq_idx = int(np.floor(f.shape[0] * freq_range))

    dSdt, dSdf = vocalpy.feature._sat._get_spectral_derivatives(spectra1, spectra2, max_freq_idx)

    assert isinstance(dSdt, np.ndarray)
    assert isinstance(dSdf, np.ndarray)
    assert dSdt.ndim == 3
    assert dSdf.ndim == 3
    assert dSdt.shape == dSdf.shape


@pytest.mark.parametrize(
    'n_fft, hop_length, freq_range, min_freq, amp_baseline, max_F0, fmax_yin, trough_threshold',
    [
        (400, 40, 0.5, 380.0, 70.0, 1830.0, 8000.0, 0.1)
    ]
)
def test_sat(
        sound_to_test_sat_features, n_fft, hop_length,
        freq_range, min_freq, amp_baseline, max_F0, fmax_yin, trough_threshold
):
    """Test :func:`vocalpy.feature._sat.sat` returns expected outputs"""
    power_spectrogram = vocalpy.spectral.sat.sat_multitaper(sound_to_test_sat_features, n_fft, hop_length)
    if sound_to_test_sat_features.samplerate <= 10000:
        # set to some value at/below Nyquist freq
        fmax_yin = 5000

    out = vocalpy.feature._sat.sat(
        sound_to_test_sat_features,
        n_fft,
        hop_length,
        freq_range,
        min_freq,
        amp_baseline,
        max_F0,
        fmax_yin,
        trough_threshold
    )

    assert isinstance(out, vocalpy.Features)
    assert len(out.data.data_vars) == 6
    assert len(out.data.coords) == 2
    for coord_name in ('channel', 'time'):
        assert coord_name in out.data.coords
    assert out.data.coords['channel'].shape[0] == sound_to_test_sat_features.data.shape[0]
    assert out.data.coords['time'].shape[0] == power_spectrogram.data.shape[-1]
    for ftr_name in (
            'amplitude', 'pitch', 'goodness_of_pitch', 'frequency_modulation', 'amplitude_modulation', 'entropy'
    ):
        assert ftr_name in out.data.data_vars
        assert out.data.data_vars[ftr_name].shape[0] == sound_to_test_sat_features.data.shape[0]
        assert out.data.data_vars[ftr_name].shape[1] == power_spectrogram.data.shape[-1]
