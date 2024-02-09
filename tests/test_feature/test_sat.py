import numpy as np
import pytest
import xarray as xr

import vocalpy

from ..fixtures.audio import ALL_ZEBRA_FINCH_WAVS, MULTICHANNEL_FLY_WAV


AUDIO_PATHS = ALL_ZEBRA_FINCH_WAVS + [MULTICHANNEL_FLY_WAV]


@pytest.fixture(params=AUDIO_PATHS)
def sound_to_test_sat_features(request):
    audio_path = request.param
    sound = vocalpy.Sound.read(audio_path)
    return sound


@pytest.fixture
def sound_and_sat_spectral_representations(sound_to_test_sat_features):
    sat_result = vocalpy.spectral.sat(sound_to_test_sat_features)
    return sound_to_test_sat_features, sat_result


def test_goodness_of_pitch(sound_and_sat_spectral_representations):
    """Test :func:`vocalpy.feature.sat.goodness_of_pitch` returns expected outputs"""
    sound, spectral_sat_result = sound_and_sat_spectral_representations
    cepstrogram, quefrencies = spectral_sat_result[1:3]
    out = vocalpy.feature.sat.goodness_of_pitch(
        cepstrogram, quefrencies,
    )
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[0] == sound.data.shape[0]
    assert out.shape[-1] == cepstrogram.shape[-1]


def test_mean_frequency(sound_and_sat_spectral_representations):
    """Test :func:`vocalpy.feature.sat.mean_frequency` returns expected outputs"""
    sound, spectral_sat_result = sound_and_sat_spectral_representations
    power_spectrogram, max_freq = spectral_sat_result[0], spectral_sat_result[3]
    out = vocalpy.feature.sat.mean_frequency(
        power_spectrogram, max_freq=max_freq
    )
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[0] == sound.data.shape[0]
    assert out.shape[-1] == power_spectrogram.data.shape[-1]
    # sanity check, mean should be less than/equal to max_freq
    assert np.all(
        out[~np.isnan(out)] <= max_freq
    )


def test_frequency_modulation(sound_and_sat_spectral_representations):
    """Test :func:`vocalpy.feature.sat.frequency_modulation` returns expected outputs"""
    sound, spectral_sat_result = sound_and_sat_spectral_representations
    dSdt, dSdf = spectral_sat_result[4:]
    out = vocalpy.feature.sat.frequency_modulation(
        dSdt, dSdf
    )
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[-1] == dSdt.shape[-1]


def test_amplitude_modulation(sound_and_sat_spectral_representations):
    """Test :func:`vocalpy.feature.sat.amplitude_modulation` returns expected outputs"""
    sound, spectral_sat_result = sound_and_sat_spectral_representations
    dSdt = spectral_sat_result[5]
    out = vocalpy.feature.sat.amplitude_modulation(
        dSdt
    )
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[-1] == dSdt.shape[-1]


def test_entropy(sound_and_sat_spectral_representations):
    """Test :func:`vocalpy.feature.sat.entropy` returns expected outputs"""
    sound, spectral_sat_result = sound_and_sat_spectral_representations
    power_spectrogram, max_freq = spectral_sat_result[0], spectral_sat_result[3]
    out = vocalpy.feature.sat.entropy(
        power_spectrogram, max_freq=max_freq
    )
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[-1] == power_spectrogram.data.shape[-1]


def test_amplitude(sound_and_sat_spectral_representations):
    """Test :func:`vocalpy.feature.sat.amplitude` returns expected outputs"""
    sound, spectral_sat_result = sound_and_sat_spectral_representations
    power_spectrogram, max_freq = spectral_sat_result[0], spectral_sat_result[3]
    out = vocalpy.feature.sat.amplitude(
        power_spectrogram, max_freq=max_freq
    )
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[-1] == power_spectrogram.data.shape[-1]



@pytest.mark.parametrize(
    'frame_length, hop_length',
    [
        (400, 40)
    ]
)
def test_pitch(frame_length, hop_length, sound_and_sat_spectral_representations):
    """Test :func:`vocalpy.feature.sat.pitch` returns expected outputs"""
    sound, spectral_sat_result = sound_and_sat_spectral_representations
    if sound.samplerate <= 10000:
        out = vocalpy.feature.sat.pitch(
            sound, fmax_yin=5000,
        )
    else:
        out = vocalpy.feature.sat.pitch(
            sound
        )

    # get spectrogram as lazy way to figure out expected number of time bins
    power_spectrogram = spectral_sat_result[0]

    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[-1] == power_spectrogram.data.shape[-1]


def test_similarity_features(sound_and_sat_spectral_representations):
    """Test :func:`vocalpy.feature.sat.similarity_features` returns expected outputs"""
    sound, spectral_sat_result = sound_and_sat_spectral_representations
    if sound.samplerate <= 10000:
        out = out = vocalpy.feature.sat.similarity_features(sound, fmax_yin=5000)
    else:
        out = out = vocalpy.feature.sat.similarity_features(sound)
    assert isinstance(out, xr.Dataset)
    assert len(out.data_vars) == 6
    assert len(out.coords) == 2
    for coord_name in ('channel', 'time'):
        assert coord_name in out.coords
    assert out.coords['channel'].shape[0] == sound.data.shape[0]
    power_spectrogram = spectral_sat_result[0]
    assert out.coords['time'].shape[0] == power_spectrogram.data.shape[-1]
    for ftr_name in (
            'amplitude', 'pitch', 'goodness_of_pitch', 'frequency_modulation', 'amplitude_modulation', 'entropy'
    ):
        assert ftr_name in out.data_vars
        assert out.data_vars[ftr_name].shape[0] == sound.data.shape[0]
        assert out.data_vars[ftr_name].shape[1] == power_spectrogram.data.shape[-1]
