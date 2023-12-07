import numpy as np
import pytest
import xarray as xr

import vocalpy


@pytest.fixture()
def spectral_sat_result(a_zebra_finch_wav):
    audio = vocalpy.Audio.read(a_zebra_finch_wav)
    out = vocalpy.spectral.sat(audio)
    return out


def test_goodness_of_pitch(spectral_sat_result):
    """Smoke test that tests :func:`vocalpy.feature.sat.goodness_of_pitch` returns expected outputs"""
    cepstrogram, quefrencies = spectral_sat_result[1:3]
    out = vocalpy.feature.sat.goodness_of_pitch(
        cepstrogram, quefrencies,
    )
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert out.shape[-1] == cepstrogram.shape[-1]


def test_mean_frequency(spectral_sat_result):
    """Smoke test that tests :func:`vocalpy.feature.sat.mean_frequency` returns expected outputs"""
    power_spectrogram, max_freq = spectral_sat_result[0], spectral_sat_result[3]
    out = vocalpy.feature.sat.mean_frequency(
        power_spectrogram, max_freq=max_freq
    )
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert out.shape[-1] == power_spectrogram.data.shape[-1]
    # sanity check, mean should be less than/equal to max_freq
    assert np.all(
        out[~np.isnan(out)] <= max_freq
    )


def test_frequency_modulation(spectral_sat_result):
    """Smoke test that tests :func:`vocalpy.feature.sat.frequency_modulation` returns expected outputs"""
    dSdt, dSdf = spectral_sat_result[4:]
    out = vocalpy.feature.sat.frequency_modulation(
        dSdt, dSdf
    )
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert out.shape[-1] == dSdt.shape[-1]


def test_amplitude_modulation(spectral_sat_result):
    """Smoke test that tests :func:`vocalpy.feature.sat.amplitude_modulation` returns expected outputs"""
    dSdt = spectral_sat_result[5]
    out = vocalpy.feature.sat.amplitude_modulation(
        dSdt
    )
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert out.shape[-1] == dSdt.shape[-1]


def test_entropy(spectral_sat_result):
    """Smoke test that tests :func:`vocalpy.feature.sat.entropy` returns expected outputs"""
    power_spectrogram, max_freq = spectral_sat_result[0], spectral_sat_result[3]
    out = vocalpy.feature.sat.entropy(
        power_spectrogram, max_freq=max_freq
    )
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert out.shape[-1] == power_spectrogram.data.shape[-1]


def test_amplitude(spectral_sat_result):
    """Smoke test that tests :func:`vocalpy.feature.sat.amplitude` returns expected outputs"""
    power_spectrogram, max_freq = spectral_sat_result[0], spectral_sat_result[3]
    out = vocalpy.feature.sat.amplitude(
        power_spectrogram, max_freq=max_freq
    )
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert out.shape[-1] == power_spectrogram.data.shape[-1]



@pytest.mark.parametrize(
    'frame_length, hop_length',
    [
        (400, 40)
    ]
)
def test_pitch(frame_length, hop_length, a_zebra_finch_wav):
    """Smoke test that tests :func:`vocalpy.feature.sat.pitch` returns expected outputs"""
    audio = vocalpy.Audio.read(a_zebra_finch_wav)
    out = vocalpy.feature.sat.pitch(
        audio
    )
    # get spectrogram as lazy way to figure out expected number of time bins
    spectral_sat_result_ = vocalpy.spectral.sat(audio)
    power_spectrogram = spectral_sat_result_[0]

    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert out.shape[-1] == power_spectrogram.data.shape[-1]


def test_similarity_features(a_zebra_finch_wav):
    """Smoke test that tests :func:`vocalpy.feature.sat.similarity_features` returns expected outputs"""
    audio = vocalpy.Audio.read(a_zebra_finch_wav)
    out = vocalpy.feature.sat.similarity_features(audio)
    assert isinstance(out, xr.Dataset)
    assert len(out.data_vars) == 6
    assert len(out.coords) == 1  # time
