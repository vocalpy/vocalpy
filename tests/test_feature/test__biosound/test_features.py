from __future__ import annotations

from typing import Literal

import pytest
import numpy as np
import vocalpy
import xarray as xr

from ...fixtures.audio import ELIE_THEUNISSEN_2016_WAV_LIST
from ...fixtures.test_data import GENERATED_TEST_DATA_ROOT

BIOSOUND_FEATURES_ROOT = GENERATED_TEST_DATA_ROOT.joinpath(
    "feature/biosound"
)

SCALE_VAL = 2**15,
SCALE_DTYPE = np.int16,

@pytest.fixture
def a_scaled_mono_elie_theunissen_2016_sound(
    all_elie_theunissen_2016_wav_paths
):
        sound = vocalpy.Sound.read(all_elie_theunissen_2016_wav_paths)
        # we need to scale float values since soundsig loads wav as int16
        sound = vocalpy.Sound(data=(sound.data * SCALE_VAL).astype(SCALE_DTYPE), samplerate=sound.samplerate)
        # we take the first channel to "convert" to mono since this is what soundsig does, see
        # https://github.com/theunissenlab/soundsig/blob/31f63ec6b63589918b327b918e754c8cd519031d/soundsig/sound.py#L60
        sound = sound[0]
        return sound


@pytest.fixture(params=ELIE_THEUNISSEN_2016_WAV_LIST)
def elie_theunissen_2016_sound_and_biosound_features(request):
    def _elie_theunissen_2016_sound_and_biosound_temporal_envelope_features(
            feature_group: Literal["temporal", "spectral", "fundamental"]
    ):
        import vocalpy as voc
        if feature_group not in voc.feature._biosound.features.SCALAR_FEATURES:
            raise ValueError(
                "`feature_group` argument to `elie_theunissen_2016_sound_and_biosound_features` factory fixture "
                f"must be one of the keys in `voc.feature._biosound.features.SCALAR_FEATURES`: {voc.feature._biosound.features.SCALAR_FEATURES}\n"
                f"But `feature_group` argument was: {feature_group}"
            )
        wav_path = request.param
        sound = voc.Sound.read(wav_path)
        # we need to scale float values since soundsig loads wav as int16
        sound = voc.Sound(data=(sound.data * SCALE_VAL).astype(SCALE_DTYPE), samplerate=sound.samplerate)
        # we take the first channel to "convert" to mono since this is what soundsig does, see
        # https://github.com/theunissenlab/soundsig/blob/31f63ec6b63589918b327b918e754c8cd519031d/soundsig/sound.py#L60
        sound = sound[0]

        expected_feature_file_path = BIOSOUND_FEATURES_ROOT.joinpath(
            f"{wav_path.stem}.nc"
        )
        features = xr.load_dataset(expected_feature_file_path)
        features = {
            ftr_name: ftr_val
            for ftr_name, ftr_val in features.items()
            if ftr_name in voc.feature._biosound.features.SCALAR_FEATURES[feature_group]
        }
        return sound, features
    return _elie_theunissen_2016_sound_and_biosound_temporal_envelope_features


def test_temporal_envelope_features(all_elie_theunissen_2016_wav_paths):
    sound = vocalpy.Sound.read(all_elie_theunissen_2016_wav_paths)

    out = vocalpy.feature._biosound.features.temporal_envelope_features(
        data=sound.data, samplerate=sound.samplerate
    )

    assert isinstance(out, dict)
    for ftr_name in vocalpy.feature._biosound.features.SCALAR_FEATURES["temporal"]:
        assert ftr_name in out
        ftr_val = out[ftr_name]
        assert np.isscalar(ftr_val)


def test_temporal_envelope_features_replicates(elie_theunissen_2016_sound_and_biosound_features):
    sound, features = elie_theunissen_2016_sound_and_biosound_features("temporal")

    out = vocalpy.feature._biosound.features.temporal_envelope_features(
        data=sound.data, samplerate=sound.samplerate
    )

    for ftr_name, ftr_val in out.items():
        np.testing.assert_allclose(
            ftr_val, features[ftr_name]
        )


def test_spectral_envelope_features(all_elie_theunissen_2016_wav_paths):
    sound = vocalpy.Sound.read(all_elie_theunissen_2016_wav_paths)
    sound = sound.to_mono()

    out = vocalpy.feature._biosound.features.spectral_envelope_features(
        data=sound.data, samplerate=sound.samplerate
    )

    assert isinstance(out, dict)
    for ftr_name in vocalpy.feature._biosound.features.SCALAR_FEATURES["spectral"]:
        assert ftr_name in out
        ftr_val = out[ftr_name]
        assert np.isscalar(ftr_val)


def test_spectral_envelope_features_replicates(elie_theunissen_2016_sound_and_biosound_features):
    sound, features = elie_theunissen_2016_sound_and_biosound_features("spectral")
    sound = sound.to_mono()

    out = vocalpy.feature._biosound.features.spectral_envelope_features(
        data=sound.data, samplerate=sound.samplerate
    )

    assert isinstance(out, dict)
    for ftr_name in vocalpy.feature._biosound.features.SCALAR_FEATURES["spectral"]:
        assert ftr_name in out
    for ftr_name, ftr_val in out.items():
        np.testing.assert_allclose(
            ftr_val, features[ftr_name]
        )


def test_fundamental_features(all_elie_theunissen_2016_wav_paths):
    sound = vocalpy.Sound.read(all_elie_theunissen_2016_wav_paths)
    sound = sound.to_mono()

    out = vocalpy.feature._biosound.features.fundamental_features(
        data=sound.data, samplerate=sound.samplerate
    )

    assert isinstance(out, dict)
    for ftr_name in vocalpy.feature._biosound.features.SCALAR_FEATURES["fundamental"]:
        assert ftr_name in out
        ftr_val = out[ftr_name]
        assert np.isscalar(ftr_val)


def test_fundamental_features_replicates(elie_theunissen_2016_sound_and_biosound_features):
    sound, features = elie_theunissen_2016_sound_and_biosound_features("fundamental")
    sound = sound.to_mono()

    out = vocalpy.feature._biosound.features.fundamental_features(
        data=sound.data, samplerate=sound.samplerate
    )

    assert isinstance(out, dict)
    for ftr_name in vocalpy.feature._biosound.features.SCALAR_FEATURES["fundamental"]:
        assert ftr_name in out
    for ftr_name, ftr_val in out.items():
        np.testing.assert_allclose(
            ftr_val, features[ftr_name]
        )