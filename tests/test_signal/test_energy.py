import inspect

import numpy as np
import pytest

import vocalpy

from ..fixtures.audio import JOURJINE_ETAL_2023_WAV_LIST


def test_meansquared(all_soundfile_paths):
    sound = vocalpy.Sound.read(all_soundfile_paths)
    if sound.samplerate <= 10000:  # fly multichannel has low samplerate
        out = vocalpy.signal.energy.meansquared(sound, freq_cutoffs=(100, 4900))
    else:
        out = vocalpy.signal.energy.meansquared(sound)
    assert isinstance(out, np.ndarray)
    assert np.all(out >= 0.)  # because it's squared


@pytest.fixture(params=JOURJINE_ETAL_2023_WAV_LIST)
def jourjine_et_al_wav_2023_path(request):
    return request.param

AVA_ENERGY_PARAMETERS = list(
    inspect.signature(vocalpy.signal.energy.ava).parameters.keys()
)


@pytest.mark.parametrize(
    'params, return_spect',
    [
        (vocalpy.segment.JOURJINEETAL2023, False),
        (vocalpy.segment.JOURJINEETAL2023, True),
        (vocalpy.segment.PETERSONETAL2023, False),
        (vocalpy.segment.PETERSONETAL2023, True),
    ]
)
def test_ava(jourjine_et_al_wav_2023_path, params, return_spect):
    """Test :func:`vocalpy.signal.energy.ava`"""
    sound = vocalpy.Sound.read(jourjine_et_al_wav_2023_path)
    kwargs = {**vocalpy.segment.JOURJINEETAL2023}
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in AVA_ENERGY_PARAMETERS
    }
    kwargs["return_spect"] = return_spect

    out = vocalpy.signal.energy.ava(sound, **kwargs)

    if return_spect:
        assert len(out) == 3
        amps, dt, spect = out
    else:
        assert len(out) == 2
        amps, dt = out
    assert isinstance(amps, np.ndarray)
    assert isinstance(dt, float)
    if return_spect:
        assert isinstance(spect, vocalpy.Spectrogram)
