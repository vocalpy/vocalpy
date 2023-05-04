import enum

from sqlalchemy import Column, Enum, ForeignKey, Integer, String, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class AudioFile(Base):
    __tablename__ = "audio_files"

    id = Column(Integer, primary_key=True)
    dataset_file_id = Column(Integer, ForeignKey("dataset_files.id"))

    dataset_file = relationship("DatasetFile", back_populates="audio_file")
    spectrogram_files = relationship("SpectrogramFile", back_populates="source_audio_file")

    def __repr__(self):
        return f"AudioFile(id={self.id!r}, dataset_file_id={self.dataset_file_id!r})"


annotates_table = Table(
    "annotates",
    Base.metadata,
    Column("annotation_file_id", ForeignKey("annotation_files.id")),
    Column("dataset_file_id", ForeignKey("dataset_files.id")),
)


class AnnotationFile(Base):
    __tablename__ = "annotation_files"

    id = Column(Integer, primary_key=True)
    dataset_file_id = Column(Integer, ForeignKey("dataset_files.id"))
    annotates = relationship("DatasetFile", secondary=annotates_table)

    def __repr__(self):
        return f"AnnotationFile(id={self.id!r}, dataset_file_id={self.dataset_file_id!r}"


class SpectrogramFile(Base):
    __tablename__ = "spectrogram_files"

    id = Column(Integer, primary_key=True)
    source_audio_file_id = Column("audio_file_id", Integer, ForeignKey("audio_files.id"))
    spectrogram_parameters_id = Column("spectrogram_parameters_id", Integer, ForeignKey("spectrogram_parameters.id"))
    dataset_file_id = Column(Integer, ForeignKey("dataset_files.id"))

    source_audio_file = relationship("AudioFile", back_populates="")
    spectrogram_parameters = relationship("SpectrogramParameters", back_populates="spectrogram_files")
    dataset_file = relationship("DatasetFile", back_populates="spectrogram_file")

    def __repr__(self):
        return (
            f"SpectrogramFile(id={self.id!r}, source_audio_file={self.source_audio_file!r}, "
            f"spectrogram_parameters={self.spectrogram_parameters!r})"
        )


class SpectrogramParameters(Base):
    __tablename__ = "spectrogram_parameters"

    id = Column(Integer, primary_key=True)
    fft_size = Column(Integer)
    step_size = Column(Integer)

    spectrogram_files = relationship("SpectrogramFile", back_populates="spectrogram_parameters")

    def __repr__(self):
        return f"SpectrogramParameters(id={self.id!r}, fft_size={self.fft_size!r}, step_size={self.step_size!r}"


class DatasetFileTypeEnum(enum.Enum):
    AudioFile = 1
    SpectrogramFile = 2
    AnnotationFile = 3
    FeatureFile = 4


class DatasetFile(Base):
    __tablename__ = "dataset_files"

    id = Column(Integer, primary_key=True)
    path = Column(String)
    file_type = Column(Enum(DatasetFileTypeEnum))

    audio_file = relationship("AudioFile", back_populates="dataset_file", uselist=False)
    spectrogram_file = relationship("SpectrogramFile", back_populates="dataset_file", uselist=False)

    def __repr__(self):
        return f"DatasetFile(id={self.id!r}, path={self.path!r}, file_type={self.file_type!r}"
