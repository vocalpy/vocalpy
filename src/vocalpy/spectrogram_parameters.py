"""Class that represents the parameters used to compute a spectrogram."""

from __future__ import annotations

import attrs


@attrs.define
class SpectrogramParameters:
    """Class that represents the parameters used to compute a spectrogram.

    Attributes
    ----------
    n_fft : int
        Size of window used for Fast Fourier Transform (FFT),
        in number of samples.
    hop_length : int
        Size of step taken with window for FFT,
        in number of samples.
    sandbox : dict
        A "sandbox" of any additional parameters.
    """

    n_fft: int
    hop_length: int
    sandbox: dict | None = attrs.field(
        validator=attrs.validators.optional(
            attrs.validators.instance_of(dict)
        ),
        default=None,
    )
