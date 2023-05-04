"""Class that represents an audio file,
as part of a dataset used to study
animal acoustic communication.

Used by the :class:`vocalpy.Dataset` class.
"""
from __future__ import annotations

import pathlib

import attrs


@attrs.define
class AudioFile:
    """Class that represents an audio file,
    as part of a dataset used to study
    animal acoustic communication.

    Used by the :class:`vocalpy.Dataset` class.

    Attributes
    ----------
    path : pathlib.Path
        The path to the audio file.
    """

    path: pathlib.Path = attrs.field(converter=pathlib.Path, validator=attrs.validators.instance_of(pathlib.Path))
