"""A class representing parameters used to compute a spectrogram."""
from __future__ import annotations

import attrs


@attrs.define
class SpectrogramParameters:
    """A class representing parameters used to compute a spectrogram.

    Attributes
    ----------
    fft_size : int
    step_size : int
    sandbox : dict
    """

    fft_size: int
    step_size: int
    sandbox: dict | None = attrs.field(
        validator=attrs.validators.optional(attrs.validators.instance_of(dict)), default=None
    )
