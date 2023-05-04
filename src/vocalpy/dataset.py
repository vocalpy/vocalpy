from __future__ import annotations

import pathlib

import attrs
from sqlalchemy import create_engine

from . import paths
from .annotation_file import AnnotationFile
from .audio_file import AudioFile
from .dataset_file import DatasetFile, DatasetFileType, DatasetFileTypeEnum
from .repository import SqlAlchemyRepository


@attrs.define
class Dataset:
    """Class that represents a dataset
    used to study animal acoustic communication.

    Attributes
    ----------
    files : list
        The files in a dataset.
        A :class:`list` of any of the following:
        :class:`AudioFile`, :class:`SpectrogramFile`,
        :class:`AnnotationFile`, :class:`FeatureFile`.
    """

    files: list[DatasetFile] = attrs.field()

    @files.validator
    def validate_files(self, attribute, value):
        if not isinstance(value, list):
            raise TypeError(f"Dataset `files` must be a list but type was: {type(value)}")

        if not all([isinstance(element, DatasetFile) for element in value]):
            types_in_files = set([type(element) for element in value])
            raise TypeError(
                f"All elements in the list `files` should be of type DatasetFile "
                f"but found the following types: {types_in_files}"
            )

    @classmethod
    def from_files(cls, files: list[DatasetFileType]):
        dataset_files = []

        for file in files:
            dataset_file = DatasetFile(file=file, file_type=DatasetFileTypeEnum(type(file)), path=file.path)
            dataset_files.append(dataset_file)

        return cls(files=dataset_files)

    def to_sqlite(self, sqlite_path: str | pathlib.Path, repository=SqlAlchemyRepository):
        if not str(sqlite_path).endswith(".db"):
            sqlite_path = pathlib.Path(f"{sqlite_path}.db")
        engine = create_engine(f"sqlite:///{sqlite_path}")
        Session = sessionmaker()
        Session.configure(bind=engine)
        session = Session()

        # note we replace class with instance
        repository = repository(session=session)

        for file in self.files:
            repository.add(file, session)

        # repository.commit()

        @classmethod
        def from_dir(
            cls,
            dir: str | pathlib.Path,
            audio_ext: str = "wav",
            annot_ext: str = "csv",
            spect_ext: str = "npz",
            recurse: bool = False,
        ):
            dir = pathlib.Path(dir)
            if not dir.is_dir(dir):
                raise NotADirectoryError(f"Argument `dir` not recognized as a directory: {dir}")

            # ---- look for audio files
            audio_paths = paths.from_dir(dir, audio_ext, recurse)
            if audio_paths:
                audio_files = [AudioFile(path=audio_path) for audio_path in audio_paths]
            else:
                audio_files = None

            # ---- look for spectrogram files, possibly pair with audio files
            spect_paths = paths.from_dir(dir, spect_ext, recurse)
            if spect_paths:
                if audio_files:
                    audio_spect_pairs = pair_files(audio_files, spect_paths)
                else:
                    # if we didn't find any audio files, default to None for each spectrogram path
                    audio_spect_pairs = [(None, spect_path) for spect_path in spect_paths]
                spect_files = [
                    SpectrogramFile(path=spect_path, source_audio_file=audio_file)
                    for audio_file, spect_path in audio_spect_pairs
                ]
            else:
                spect_files = None

            # ---- look for annotation files, preferentially pair with audio files, else pair with spectrogram files
            annot_paths = paths.from_dir(dir, annot_ext, recurse)
            if annot_paths:
                # by default we try to pair annotations with audio files first
                if audio_files:
                    audio_annot_pairs = pair_files(audio_files, annot_paths)
                    annot_files = [
                        AnnotationFile(path=annot_path, annotates=audio_file)
                        for audio_file, annot_path in audio_annot_pairs
                    ]
                elif spect_files:
                    spect_annot_pairs = pair_files(spect_files, annot_paths)
                    annot_files = [
                        AnnotationFile(path=annot_path, annotates=audio_file)
                        for audio_file, annot_path in spect_annot_pairs
                    ]
            else:
                annot_files = None

            files = []
            for files_list in (annot_files, audio_files, spect_files):
                if files_list is not None:
                    files.extend(files_list)

            return cls(files=files)

    @classmethod
    def from_sqlite(cls, sqlite_path):
        pass
