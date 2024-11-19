"""A core package for acoustic communication research in Python."""

from . import (
    _vendor,
    constants,
    examples,
    feature,
    metrics,
    paths,
    plot,
    segment,
    signal,
    spectral,
    validators,
)
from .__about__ import (
    __author__,
    __commit__,
    __copyright__,
    __email__,
    __license__,
    __summary__,
    __title__,
    __uri__,
    __version__,
)
from ._spectrogram.data_type import Spectrogram
from .annotation import Annotation
from .annotation_file import AnnotationFile
from .audio_file import AudioFile
from .dataset_file import DatasetFile, DatasetFileType, DatasetFileTypeEnum
from .examples import example
from .feature_extractor import FeatureExtractor
from .feature_file import FeatureFile
from .features import Features
from .params import Params
from .segmenter import Segmenter
from .segments import Segments
from .sequence import Sequence
from .sound import Sound
from .spectrogram import spectrogram
from .spectrogram_file import SpectrogramFile
from .spectrogram_maker import SpectrogramMaker
from .spectrogram_parameters import SpectrogramParameters
from .unit import Unit

__all__ = [
    "__author__",
    "__commit__",
    "__copyright__",
    "__email__",
    "__license__",
    "__summary__",
    "__title__",
    "__uri__",
    "__version__",
    "Annotation",
    "AnnotationFile",
    "Sound",
    "AudioFile",
    "constants",
    "DatasetFile",
    "DatasetFileType",
    "DatasetFileTypeEnum",
    "FeatureExtractor",
    "FeatureFile",
    "Features",
    "example",
    "examples",
    "feature",
    "paths",
    "plot",
    "Params",
    "Segment",
    "Segmenter",
    "Segments",
    "Sequence",
    "metrics",
    "segment",
    "signal",
    "spectral",
    "spectrogram",
    "Spectrogram",
    "SpectrogramFile",
    "SpectrogramMaker",
    "SpectrogramParameters",
    "Unit",
    "validators",
    "_vendor",
]
