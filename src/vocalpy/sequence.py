from __future__ import annotations

import pathlib

import attrs

from .unit import Unit


@attrs.define
class Sequence:
    units: list[Unit] = attrs.field()
    @units.validator
    def is_list_of_unit(self, attribute, value):
        if not isinstance(value, list) or not all([isinstance(item, Unit) for item in value]):
            raise ValueError(
                f"`units` must be a list of vocalpy.Unit instances"
            )

    source_audio_path : pathlib.Path = attrs.field(converter=attrs.converters.optional(pathlib.Path),
                                                   validator=attrs.validators.optional(
                                                       attrs.validators.instance_of(pathlib.Path)
                                                   ),
                                                   default=None)

