"""This sub-package contains a domain model of an
animal acoustic communication dataset.
The term "domain model" is used in the Domain-Driven Development sense.

The model is meant to allow for
decoupling the dataset itself from the means of persistent storage,
e.g. a database.
"""
from .audio import Audio
from .annotation_file import AnnotationFile
from .audio_file import AudioFile
from .dataset import Dataset
from .dataset_file import DatasetFile, DatasetFileType, DatasetFileTypeEnum
from .feature_file import FeatureFile
from .sequence import Sequence
from .spectrogram import Spectrogram
from .spectrogram_file import SpectrogramFile
from .spectrogram_parameters import SpectrogramParameters
from .unit import Unit


__all__ = [
    'AnnotationFile',
    'Audio',
    'AudioFile',
    'Dataset',
    'DatasetFile',
    'DatasetFileType',
    'DatasetFileTypeEnum',
    'FeatureFile',
    'Sequence',
    'Spectrogram',
    'SpectrogramFile',
    'SpectrogramParameters',
    'Unit',
]
