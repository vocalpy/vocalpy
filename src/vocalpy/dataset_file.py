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
from typing import Union

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


DatasetFileType = Union[
    AnnotationFile, AudioFile, FeatureFile, SpectrogramFile
]


@attrs.define
class DatasetFile:
    """A class that represents any file in a dataset.

    Attributes
    ----------
    file : AudioFile, SpectrogramFile, AnnotationFile, or FeatureFile
        An instance of one of the specific file types.
        This is the only required argument when instantiating a
        :class:`vocalpy.DatasetFile`; other attributes are determined
        from this argument.
    file_type: DatasetFileTypeEnum
        The file type, represented as an enum.
        Used when saving the dataset to a database,
        to specify which table the file's metadata should go into.
    path : pathlib.Path
        The path to the file, taken from ``file.path``.

    Examples
    --------
    >>> import vocalpy as voc
    >>> audio_paths = voc.paths.from_dir('./dir', 'wav')
    >>> audio_files = [voc.AudioFile(path=path) for path in audio_paths]
    >>> dataset_files = [voc.DatasetFile(file=audio_file) for sound_file in audio_files]
    """

    file: DatasetFileType = attrs.field(
        validator=attrs.validators.instance_of(
            (AnnotationFile, AudioFile, FeatureFile, SpectrogramFile)
        )
    )

    def __attrs_post_init__(self):
        self.file_type = DatasetFileTypeEnum[self.file.__class__.__name__]
        self.path = self.file.path
