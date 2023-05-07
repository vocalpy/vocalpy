from __future__ import annotations

import attrs
import numpy as np

from .audio import Audio
from .spectrogram import Spectrogram
from .unit import Unit


@attrs.define
class Sequence:
    """A sequence of units,
    as analyzed in acoustic communication research.

    Attributes
    ----------
    units : list
        A :class:`list` of `vocalpy.Unit` instances.
        Produced by segmenting audio, e.g. with
        :func:`vocalpy.signal.segment.segment_audio_amplitude`.
    segment_method : str
        The method used to segment audio.
        Either a string name of a method
        or the name of a callback.
        This attribute is added by a
        :class:`vocalpy.Segmenter` when it
        creates a :class:`vocalpy.Sequence`.
    segment_params : dict
        The parameters used with the segmenting method.
        This attribute is added by a
        :class:`vocalpy.Segmenter` when it
        creates a :class:`vocalpy.Sequence`.
    audio : vocalpy.Audio, optional
        The audio from which this sequence was segmented.
        Optional, default is None.
    spectrogram : vocalpy.Spectrogram, optional
        The spectrogram from which this sequence was segmented.
        Optional, default is None.
    """

    units: list[Unit] = attrs.field()

    @units.validator
    def is_list_of_unit(self, attribute, value):
        if not isinstance(value, list) or not all([isinstance(item, Unit) for item in value]):
            raise ValueError("`units` must be a list of vocalpy.Unit instances")

    method: str = attrs.field(converter=str, validator=attrs.validators.instance_of(str))
    segment_params: dict = attrs.field(converter=dict, validator=attrs.validators.instance_of(dict))

    audio: Audio = attrs.field(validator=attrs.validators.optional(attrs.validators.instance_of(Audio)), default=None)
    spectrogram: Spectrogram = attrs.field(
        validator=attrs.validators.optional(attrs.validators.instance_of(Spectrogram)), default=None
    )

    def __attrs_post_init__(self):
        units = sorted(self.units, key=lambda unit: unit.onset)
        onsets = [unit.onset for unit in units]
        if len(onsets) > 1:
            if not np.all(onsets[1:] > onsets[:-1]):
                import pdb

                pdb.set_trace()
                raise ValueError(f"Onsets of units are not strictly increasing")

    @property
    def onset(self):
        return self.units[0].onset

    @property
    def offset(self):
        return self.units[-1].offset
