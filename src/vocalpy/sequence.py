from __future__ import annotations

import pathlib

import attrs
import numpy as np

from .unit import Unit
from .audio import Audio
from .spectrogram import Spectrogram


@attrs.define
class Sequence:
    units: list[Unit] = attrs.field()

    @units.validator
    def is_list_of_unit(self, attribute, value):
        if not isinstance(value, list) or not all([isinstance(item, Unit) for item in value]):
            raise ValueError("`units` must be a list of vocalpy.Unit instances")

    audio_path: pathlib.Path = attrs.field(
        converter=attrs.converters.optional(pathlib.Path),
        validator=attrs.validators.optional(attrs.validators.instance_of(pathlib.Path)),
        default=None,
    )

    spectrogram_path: pathlib.Path = attrs.field(
        converter=attrs.converters.optional(pathlib.Path),
        validator=attrs.validators.optional(attrs.validators.instance_of(pathlib.Path)),
        default=None,
    )

    def __attrs_post_init__(self):
        units = sorted(self.units, key=lambda unit: unit.onset_s or unit.onset_sample)
        onsets = [unit.onset for unit in units]
        if not np.all(onsets[1:] > onsets[:-1]):
            raise ValueError(
                f""
            )
        self._audio = None
        self._spectrogram = None

    @property
    def audio(self):
        if self._audio is None:
            if self.audio_path is None:
                raise ValueError(
                    f'Cannot load audio because the `audio_path` attribute of this Sequence is None. '
                    f'Please create a Sequence with an `audio_path`'
                )
            self._audio = Audio.read(self.audio_path)
        return self._audio

    @property
    def spectrogram(self):
        if self._spectrogram is None:
            if self.spectrogram_path is None:
                raise ValueError(
                    f'Cannot load spectrogram because the `spectrogram_path` attribute of this Sequence is None. '
                    f'Please create a Sequence with an `spectrogram_path`'
                )
            self._spectrogram = Spectrogram.read(self.spectrogram_path)
        return self._audio

    @property
    def onset(self):
        return self.units[0].onset

    @property
    def offset(self):
        return self.units[-1].offset
