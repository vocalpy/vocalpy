"""Class that represents the parameters used to compute a spectrogram."""
from __future__ import annotations

import attrs


@attrs.define
class SpectrogramParameters:
    """Class that represents the parameters used to compute a spectrogram.

    Attributes
    ----------
    fft_size : int
        Size of window used for Fast Fourier Transform (FFT),
        in number of samples.
    step_size : int
        Size of step taken with window for FFT,
        in number of samples. Also known as "hop size".
    sandbox : dict
        A "sandbox" of any additional parameters.
    """

    fft_size: int
    step_size: int
    sandbox: dict | None = attrs.field(
        validator=attrs.validators.optional(attrs.validators.instance_of(dict)), default=None
    )
