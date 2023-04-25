"""A class that represents a file in
a dataset used to study animal acoustic communication.

This abstraction decouples the database schema
from the dataset.
It allows a user to build a dataset
of specific file types--audio files, spectrogram files,
annotation files, etc.--and then use them all
to make a dataset.
The :class:`vocalpy.dataset.Dataset` class
can accept a sequence of such files,
and then use them to create a list of
:class:`vocalpy.dataset.DatasetFile` instances.
It can do this because each :class:`vocalpy.dataset.DatasetFile`
tracks what type of file it represents.
The list created by :class:`vocalpy.dataset.Dataset`
becomes the central table in a relational database
representing the dataset.
"""
from __future__ import annotations

import enum
import pathlib
from typing import TypeAlias

import attrs

from .annotation_file import AnnotationFile
from .audio_file import AudioFile
from .feature_file import FeatureFile
from .spectrogram_file import SpectrogramFile


class DatasetFileTypeEnum(enum.Enum):
    ANNOTATION = AnnotationFile
    AUDIO = AudioFile
    FEATURE = FeatureFile
    SPECTROGRAM = SpectrogramFile


DatasetFileType: TypeAlias = AnnotationFile | AudioFile | FeatureFile | SpectrogramFile


@attrs.define
class DatasetFile:
    file: DatasetFileType = attrs.field(
        validator=attrs.validators.instance_of(
            (AnnotationFile, AudioFile, FeatureFile, SpectrogramFile)
        )
    )
    file_type: DatasetFileTypeEnum
    path: pathlib.Path
