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
SCALE_DTYPE = np.int16

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


@pytest.fixture()
def elie_theunissen_2016_sound_and_biosound_features(all_elie_theunissen_2016_wav_paths):
    def _elie_theunissen_2016_sound_and_biosound_features(
            feature_group: Literal["temporal", "spectral", "fundamental", None] = None, scale: bool = True, features_as_dict: bool = True
    ):
        import vocalpy as voc
        if feature_group is not None and feature_group not in voc.feature._biosound.features.SCALAR_FEATURES:
            raise ValueError(
                "`feature_group` argument to `elie_theunissen_2016_sound_and_biosound_features` factory fixture "
                f"must be one of the keys in `voc.feature._biosound.features.SCALAR_FEATURES`: {voc.feature._biosound.features.SCALAR_FEATURES.keys()}\n"
                f"But `feature_group` argument was: {feature_group}"
            )
        sound = voc.Sound.read(all_elie_theunissen_2016_wav_paths)
        if scale:
            # we need to scale float values since soundsig loads wav as int16
            sound = voc.Sound(data=(sound.data * SCALE_VAL).astype(SCALE_DTYPE), samplerate=sound.samplerate)
        # we take the first channel to "convert" to mono since this is what soundsig does, see
        # https://github.com/theunissenlab/soundsig/blob/31f63ec6b63589918b327b918e754c8cd519031d/soundsig/sound.py#L60
        sound = sound[0]

        expected_feature_file_path = BIOSOUND_FEATURES_ROOT.joinpath(
            f"{all_elie_theunissen_2016_wav_paths.stem}.nc"
        )
        features = xr.load_dataset(expected_feature_file_path)
        if feature_group is not None:
            if features_as_dict:
                features = {
                    ftr_name: ftr_val
                    for ftr_name, ftr_val in features.items()
                    if ftr_name in voc.feature._biosound.features.SCALAR_FEATURES[feature_group]
                }
            else:
                features = features[voc.feature._biosound.features.SCALAR_FEATURES[feature_group]]
        return sound, features
    return _elie_theunissen_2016_sound_and_biosound_features


def test_temporal_envelope_features(a_scaled_mono_elie_theunissen_2016_sound):
    sound = a_scaled_mono_elie_theunissen_2016_sound

    out = vocalpy.feature._biosound.features.temporal_envelope_features(
        # we do `sound.data[0. :]` here since these helper functions expect 1-D arrays
        data=sound.data[0, :], samplerate=sound.samplerate
    )

    assert isinstance(out, dict)
    for ftr_name in vocalpy.feature._biosound.features.SCALAR_FEATURES["temporal"]:
        assert ftr_name in out
        ftr_val = out[ftr_name]
        assert np.isscalar(ftr_val)


def test_temporal_envelope_features_replicates(elie_theunissen_2016_sound_and_biosound_features):
    sound, features = elie_theunissen_2016_sound_and_biosound_features("temporal")

    out = vocalpy.feature._biosound.features.temporal_envelope_features(
        # we do `sound.data[0. :]` here since these helper functions expect 1-D arrays
        data=sound.data[0, :], samplerate=sound.samplerate
    )

    for ftr_name, expected_ftr_val in features.items():
        ftr_val = out[ftr_name]
        assert np.allclose(
            ftr_val, expected_ftr_val
        )


def test_spectral_envelope_features(a_scaled_mono_elie_theunissen_2016_sound):
    sound = a_scaled_mono_elie_theunissen_2016_sound

    out = vocalpy.feature._biosound.features.spectral_envelope_features(
        # we do `sound.data[0. :]` here since these helper functions expect 1-D arrays
        data=sound.data[0, :], samplerate=sound.samplerate
    )

    assert isinstance(out, dict)
    for ftr_name in vocalpy.feature._biosound.features.SCALAR_FEATURES["spectral"]:
        assert ftr_name in out
        ftr_val = out[ftr_name]
        assert np.isscalar(ftr_val)


def test_spectral_envelope_features_replicates(elie_theunissen_2016_sound_and_biosound_features):
    sound, features = elie_theunissen_2016_sound_and_biosound_features("spectral")

    out = vocalpy.feature._biosound.features.spectral_envelope_features(
        # we do `sound.data[0. :]` here since these helper functions expect 1-D arrays
        data=sound.data[0, :], samplerate=sound.samplerate
    )

    for ftr_name, expected_ftr_val in features.items():
        ftr_val = out[ftr_name]
        assert np.allclose(
            ftr_val, expected_ftr_val
        )


def test_fundamental_features(a_scaled_mono_elie_theunissen_2016_sound):
    sound = a_scaled_mono_elie_theunissen_2016_sound

    out = vocalpy.feature._biosound.features.fundamental_features(
        # we do `sound.data[0. :]` here since these helper functions expect 1-D arrays
        data=sound.data[0, :], samplerate=sound.samplerate
    )

    assert isinstance(out, dict)
    for ftr_name in vocalpy.feature._biosound.features.SCALAR_FEATURES["fundamental"]:
        assert ftr_name in out
        ftr_val = out[ftr_name]
        assert np.isscalar(ftr_val)


def test_fundamental_features_replicates(elie_theunissen_2016_sound_and_biosound_features):
    sound, features = elie_theunissen_2016_sound_and_biosound_features("fundamental")

    out = vocalpy.feature._biosound.features.fundamental_features(
        # we do `sound.data[0. :]` here since these helper functions expect 1-D arrays
        data=sound.data[0, :], samplerate=sound.samplerate
    )

    for ftr_name, expected_ftr_val in features.items():
        ftr_val = out[ftr_name]  
        np.testing.assert_allclose(
            ftr_val, expected_ftr_val,
            rtol=0.125
        )


def test_biosound(all_elie_theunissen_2016_wav_paths):
    sound = vocalpy.Sound.read(all_elie_theunissen_2016_wav_paths)
    sound = sound[0]  # make "mono" the same way `soundsig` does

    out = vocalpy.feature._biosound.features.biosound(sound)

    assert isinstance(out, vocalpy.Features)
    for ftr_group in vocalpy.feature._biosound.features.SCALAR_FEATURES.keys(): 
        for ftr_name in vocalpy.feature._biosound.features.SCALAR_FEATURES[ftr_group]:
            assert ftr_name in out.data
            ftr_val = out.data[ftr_name].values
            assert ftr_val.shape == (1,)


def test_biosound_replicates(elie_theunissen_2016_sound_and_biosound_features):
    sound, _ = elie_theunissen_2016_sound_and_biosound_features(scale=False)

    out = vocalpy.feature._biosound.features.biosound(sound)

    for ftr_group in vocalpy.feature._biosound.features.SCALAR_FEATURES.keys():
        _, expected_xr_dataset = elie_theunissen_2016_sound_and_biosound_features(ftr_group, features_as_dict=False)
        xr_dataset = out.data[vocalpy.feature._biosound.features.SCALAR_FEATURES[ftr_group]]
        if ftr_group == "fundamental":
            xr.testing.assert_allclose(
                xr_dataset, expected_xr_dataset,
                rtol=0.125
            )
        else:
            xr.testing.assert_allclose(
                xr_dataset, expected_xr_dataset
            )
