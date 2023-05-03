import attrs
import numpy as np

from. audio import Audio
from .spectrogram import Spectrogram


def convert_int(val):
    """Converter that converts :class:`numpy.integer` to :class:`int`,
    returns native Python :class:`int` as is, and
    raises an error for any other type.
    """
    if hasattr(val, "dtype") and isinstance(val, np.integer):
        return int(val)
    elif isinstance(val, int):
        return val
    else:
        raise TypeError(f"Invalid type {type(val)} for onset or offset sample: {val}. Must be an integer.")


@attrs.define
class Unit:
    """A unit in a sequence,
     as studied in animal acoustic communication."""
    audio: Audio = attrs.field(
        validator=attrs.validators.optional(
            attrs.validators.instance_of(Audio)
        ),
        default=None
    )
    spectrogram: Spectrogram = attrs.field(
        validator=attrs.validators.optional(
            attrs.validators.instance_of(Spectrogram)
        ),
        default=None
    )
    label: str = attrs.field(
        validator=attrs.validators.optional(
            attrs.validators.instance_of(str)
        ),
        default=None
    )
    onset_s = attrs.field(validator=attrs.validators.optional(
        attrs.validators.instance_of(float)
        ),
        default=None
    )
    offset_s = attrs.field(validator=attrs.validators.optional(
        attrs.validators.instance_of(float)
        ),
        default=None
    )
    onset_sample = attrs.field(
        validator=attrs.validators.optional(
            attrs.validators.instance_of(int)
        ),
        converter=attrs.converters.optional(convert_int),
        default=None,
    )
    offset_sample = attrs.field(
        validator=attrs.validators.optional(
            attrs.validators.instance_of(int)
        ),
        converter=attrs.converters.optional(convert_int),
        default=None,
    )

    def __attrs_post_init__(self):
        if (self.onset_sample is None and self.offset_sample is None) and (
            self.onset_s is None and self.offset_s is None
        ):
            raise ValueError("must provide either onset_sample and offset_sample, or " "onsets_s and offsets_s")

        if self.onset_sample and self.offset_sample is None:
            raise ValueError(f"onset_sample specified as {self.onset_sample} but offset_sample is None")
        if self.onset_sample is None and self.offset_sample:
            raise ValueError(f"offset_sample specified as {self.offset_sample} but onset_sample is None")
        if self.onset_s and self.offset_s is None:
            raise ValueError(f"onset_s specified as {self.onset_s} but offset_s is None")
        if self.onset_s is None and self.offset_s:
            raise ValueError(f"offset_s specified as {self.offset_sample} but onset_s is None")
