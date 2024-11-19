"""Class that represents an annotation file,
as part of a dataset used to study
animal acoustic communication."""

from __future__ import annotations

import pathlib

import attrs

from .audio_file import AudioFile
from .spectrogram_file import SpectrogramFile


@attrs.define
class AnnotationFile:
    """Class that represents an annotation file,
    as part of a dataset used to study
    animal acoustic communication.

    Attributes
    ----------
    path : pathlib.Path
        The path to the annotation file.
    annotates : vocalpy.AudioFile or vocalpy.SpectrogramFile
        The file that this annotation file annotates.
    """

    path: pathlib.Path = attrs.field()
    annotates: (
        AudioFile
        | SpectrogramFile
        | list[AudioFile]
        | list[SpectrogramFile]
        | None
    ) = attrs.field()

    @annotates.validator
    def is_file_or_list_of_files(self, attribute, value):
        if not isinstance(value, (AudioFile, SpectrogramFile, list)):
            raise TypeError(
                f"AnnotationFile attribute `annotates` must be either an AudioFile, "
                f"a SpectrogramFile, or a list of AudioFile or SpectrogramFile instances. "
                f"Type of `annotates` was: {type(value)}"
            )

        if isinstance(value, list):
            if not (
                all([isinstance(el, AudioFile) for el in value])
                or all([isinstance(el, SpectrogramFile) for el in value])
            ):
                list_types = set([type(el) for el in value])
                raise TypeError(
                    "If AnnotationFile attribute `annotates` is a list, "
                    "then it must be a list of AudioFile instances, "
                    "or a list of SpectrogramFile instances. "
                    f"Received a list with the following types: {list_types}"
                )
