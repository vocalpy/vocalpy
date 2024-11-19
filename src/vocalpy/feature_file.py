from __future__ import annotations

import pathlib

import attrs

from .annotation_file import AnnotationFile
from .audio_file import AudioFile
from .spectrogram_file import SpectrogramFile


@attrs.define
class FeatureFile:
    """Class that represents a file containing
    features extracted from data,
    e.g., acoustic parameters.

    The features file is part of a dataset used to study
    animal acoustic communication.

    Attributes
    ----------
    path : pathlib.Path
        The path to the annotation file.
    source_file : AudioFile, SpectrogramFile, or AnnotationFile
        The file that features were extracted from.
        A feature file can have either a single `source_file`
        or a :class:`tuple` of `source_files`.
    source_files : tuple of AudioFile, SpectrogramFile, or AnnotationFile
        The files that features were extracted from.
        A feature file can have either a single `source_file`
        or a :class:`tuple` of `source_files`.
    """

    path: pathlib.Path = attrs.field()
    source_file: AudioFile | SpectrogramFile | AnnotationFile = attrs.field(
        validator=attrs.validators.optional(
            attrs.validators.instance_of(
                (AudioFile, SpectrogramFile, AnnotationFile)
            )
        ),
        default=None,
    )
    source_files: tuple = attrs.field(default=None)

    @source_files.validator
    def is_tuple_of_files(self, attribute, value):
        if not all(
            [
                isinstance(el, (AudioFile, SpectrogramFile, AnnotationFile))
                for el in value
            ]
        ):
            raise TypeError(
                "source files must be all AudioFile, SpectrogramFile, or AnnotationFile"
            )

    def __attrs_post_init__(self):
        if self.source_file is not None and self.source_files is not None:
            raise ValueError(
                "A feature file can have either a single `source_file` "
                "or a tuple of `source_files`, but not both."
            )
