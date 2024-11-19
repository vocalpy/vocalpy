"""Class that represents a spectrogram file,
as part of a dataset used to study
animal acoustic communication."""

from __future__ import annotations

import pathlib

import attrs

from .audio_file import AudioFile
from .spectrogram_parameters import SpectrogramParameters


@attrs.define
class SpectrogramFile:
    """Class that represents a spectrogram file,
    as part of a dataset used to study
    animal acoustic communication.

    A spectrogram file contains the array
    containing the computed spectrogram
    as well as two associated arrays,
    one representing the time bins
    and one representing the frequency bins.

    Attributes
    ----------
    path : pathlib.Path
        The path to the spectrogram file.
    spectrogram_parameters : vocalpy.dataset.SpectrogramParameters
        The parameters used to compute the spectrogram.
    source_audio_file : vocalpy.dataset.AudioFile
        The audio file from which the spectrogram was computed.
    """

    path: pathlib.Path = attrs.field(
        converter=pathlib.Path,
        validator=attrs.validators.instance_of(pathlib.Path),
    )
    spectrogram_parameters: SpectrogramParameters | None = attrs.field(
        validator=attrs.validators.optional(
            attrs.validators.instance_of(SpectrogramParameters)
        ),
        default=None,
    )
    source_audio_file: AudioFile | None = attrs.field(
        validator=attrs.validators.optional(
            attrs.validators.instance_of(AudioFile)
        ),
        default=None,
    )
