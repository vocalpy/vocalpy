from __future__ import annotations

import attrs
import numpy as np

from .unit import Unit


@attrs.define
class Sequence:
    """A sequence of units,
    as analyzed in acoustic communication research.

    Attributes
    ----------
    units : list
        A :class:`list` of `vocalpy.Unit` instances.
    """

    units: list[Unit] = attrs.field()

    @units.validator
    def is_list_of_unit(self, attribute, value):
        if not isinstance(value, list) or not all(
            [isinstance(item, Unit) for item in value]
        ):
            raise ValueError(
                "`units` must be a list of vocalpy.Unit instances"
            )

    def __attrs_post_init__(self):
        units = sorted(self.units, key=lambda unit: unit.onset)
        onsets = [unit.onset for unit in units]
        if len(onsets) > 1:
            if not np.all(onsets[1:] > onsets[:-1]):
                raise ValueError(
                    f"Onsets of units are not strictly increasing.\nOnsets: {onsets}"
                )
