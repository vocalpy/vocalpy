import attrs

from .audio import Audio
from .spectrogram import Spectrogram


DEFAULT_LABEL = '-'


@attrs.define
class Unit:
    """A unit in a sequence,
    as studied in animal acoustic communication.

    Attributes
    ----------
    onset : float
        Onset time of unit, in seconds.
    offset : float
        Offset time of unit, in seconds.
    label : str
        A string label applied by an annotator to the unit.
        Default is determined by :data:`vocalpy.unit.DEFAULT_LABEL`.
    audio : vocalpy.Audio, optional
        The audio for this unit.
        Optional, default is None.
    spectrogram : vocalpy.Spectrogram, optional
        The spectrogram for this unit.
        Optional, default is None.
    """
    onset = attrs.field(validator=attrs.validators.optional(attrs.validators.instance_of(float)))
    offset = attrs.field(validator=attrs.validators.optional(attrs.validators.instance_of(float)))

    label: str = attrs.field(
        validator=attrs.validators.optional(attrs.validators.instance_of(str)), default=DEFAULT_LABEL
    )
    audio: Audio = attrs.field(validator=attrs.validators.optional(attrs.validators.instance_of(Audio)), default=None)
    spectrogram: Spectrogram = attrs.field(
        validator=attrs.validators.optional(attrs.validators.instance_of(Spectrogram)), default=None
    )

    def __attrs_post_init__(self):
        if self.offset < self.onset:
            raise ValueError(
                f"Onset should be less than offset, but onset was {self.onset} and offset was {self.offset}."
            )
        if self.onset and self.offset is None:
            raise ValueError(f"onset specified as {self.onset} but offset is None")
        if self.onset_s is None and self.offset_s:
            raise ValueError(f"offset specified as {self.offset} but onset is None")
