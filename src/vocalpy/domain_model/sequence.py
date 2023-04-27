from __future__ import annotations

import attrs

from .unit import Unit


@attrs.define
class Sequence:
    units: list[Unit] = attrs.field()

    @units.validator
    def is_list_of_unit(self, attribute, value):
        if not isinstance(list, value) or not all([isinstance(item, Unit) for item in value]):
            raise ValueError(
                f"`units` must be a list of vocalpy.Unit instances"
            )
