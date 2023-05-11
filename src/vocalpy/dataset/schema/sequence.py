from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, relationship


class SequenceDatasetBase(DeclarativeBase):
    pass


class Audio(SequenceDatasetBase):
    __tablename__ = "audios"

    id = Column(Integer, primary_key=True)
    path = Column(String)

    sequences = relationship("Sequence", back_populates="audio")

    def __repr__(self):
        return f"Audio(id={self.id!r}, path={self.path!r})"


class SegmentParams(SequenceDatasetBase):
    __tablename__ = "segment_params"
    id = Column(Integer, primary_key=True)
    path = Column(String)

    sequences = relationship("Sequence", back_populates="segment_params")

    def __repr__(self):
        return f"SegmentParams(id={self.id!r}, path={self.path!r})"


class Sequence(SequenceDatasetBase):
    __tablename__ = "sequences"

    id = Column(Integer, primary_key=True)
    audio_id = Column(Integer, ForeignKey("audios.id"))
    onset = Column(Float)
    offset = Column(Float)
    method = Column(String)  # should this be a table?
    segment_params_id = Column(Integer, ForeignKey("segment_params.id"))

    audio = relationship("Audio", back_populates="sequences")
    segment_params = relationship("SegmentParams", back_populates="sequences")

    units = relationship("Unit", back_populates="sequence")

    def __repr__(self):
        return (
            f"Sequence(id={self.id!r}, audio_id={self.audio_id!r}, onset={self.onset!r}, "
            f"offset={self.offset!r}, method={self.method!r}, "
            f"segment_params_id={self.segment_params_id!r}, audio={self.audio!r})"
        )


class Unit(SequenceDatasetBase):
    __tablename__ = "units"

    id = Column(Integer, primary_key=True)
    sequence_id = Column(Integer, ForeignKey("sequences.id"))
    onset = Column(Float)
    offset = Column(Float)
    label = Column(String)

    sequence = relationship("Sequence", back_populates="units")

    def __repr__(self):
        return f"Unit(id={self.id!r}, sequence_id={self.sequence_id!r}, onset={self.onset!r}, offset={self.offset!r})"
