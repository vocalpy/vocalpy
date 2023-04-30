"""This sub-package contains a domain model of an
animal acoustic communication dataset.
The term "domain model" is used in the Domain-Driven Development sense.

The model is meant to allow for
decoupling the dataset itself from the means of persistent storage,
e.g. a database.
"""
from .entities import (
    AnnotationFile, Audio, AudioFile, AudioFile, Dataset,
    DatasetFile, DatasetFileType, DatasetFileTypeEnum,
    FeatureFile, Sequence, Spectrogram,
    SpectrogramFile, SpectrogramParameters, Unit
)
from .services import (
    Segmenter, SpectrogramMaker,
)

__all__ = [
    'AnnotationFile',
    'Audio',
    'AudioFile',
    'Dataset',
    'DatasetFile',
    'DatasetFileType',
    'DatasetFileTypeEnum',
    'FeatureFile',
    'Segmenter',
    'Sequence',
    'Spectrogram',
    'SpectrogramFile',
    'SpectrogramMaker',
    'SpectrogramParameters',
    'Unit',
]
