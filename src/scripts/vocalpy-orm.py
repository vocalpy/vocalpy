#!/usr/bin/env python
# coding: utf-8

# In[1]:


import enum

from sqlalchemy import (
    Column,
    create_engine,
    Enum,
    ForeignKey,
    Integer,
    String,
    Table,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

import vocalpy as voc


# In[2]:


Base = declarative_base()


# In[3]:


class AudioFile(Base):
    __tablename__ = 'audio_files'
    
    id = Column(Integer, primary_key=True)
    dataset_file_id = Column(Integer, ForeignKey('dataset_files.id'))
    
    dataset_file = relationship("DatasetFile", back_populates="audio_file")
    spectrogram_files = relationship("SpectrogramFile", back_populates="source_audio_file")
    
    # def __repr__(self):


# In[ ]:


annotates_table = Table(
    "annotates",
    Base.metadata,
    Column("annotation_file_id", ForeignKey("annnotation_files.id")),
    Column("dataset_file_id", ForeignKey("dataset_files.id")),
)


# In[4]:


class AnnotationFile(Base):
    __tablename__ = 'annotation_files'
    
    id = Column(Integer, primary_key=True)
    dataset_file_id = Column(Integer, ForeignKey('dataset_files.id'))
    annotates = relationship("DatasetFile", secondary=annotates_table)


# In[ ]:


class SpectrogramFile(Base):
    __tablename__ = 'spectrogram_files'
    
    id = Column(Integer, primary_key=True)
    source_audio_file_id = Column("audio_file_id", Integer, ForeignKey("audio_files.id"))
    spectrogram_parameters_id = Column("spectrogram_parameters_id", Integer, ForeignKey("spectrogram_parameters.id"))
    dataset_file_id = Column(Integer, ForeignKey('dataset_files.id'))

    source_audio_file = relationship("AudioFile", back_populates="")
    spectrogram_parameters = relationship("SpectrogramParameters", back_populates="spectrogram_files")
    dataset_file = relationship("DatasetFile", back_populates="spectrogram_file")


# In[5]:


class SpectrogramParameters(Base):
    __tablename__ = 'spectrogram_parameters'
    
    id = Column(Integer, primary_key=True)
    fft_size = Column(Integer)
    step_size = Column(Integer)
    
    spectrogram_files = relationship("SpectrogramFile", back_populates="spectrogram_parameters")


# In[6]:


class DatasetFileTypeEnum(enum.Enum):
    AudioFile = 1
    SpectrogramFile = 2
    AnnotationFile = 3
    FeatureFile = 4


# In[7]:


class DatasetFile(Base):
    __tablename__ = 'dataset_files'
    
    id = Column(Integer, primary_key=True)
    path = Column(String)
    file_type = Column(Enum(DatasetFileTypeEnum))

    audio_file = relationship("AudioFile", 
                              back_populates="dataset_file", 
                              uselist=False)
    spectrogram_file = relationship("SpectrogramFile", 
                                    back_populates="dataset_file", 
                                    uselist=False)


# In[8]:


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# In[9]:


def get_engine():
     return create_engine('sqlite:///:memory:', echo=True)

def get_session(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

def an_audio_path():
    return './tests/data/source/bird1.cbin'


# In[10]:


engine = get_engine()
session = get_session(engine)

# next line makes tables?
Base.metadata.create_all(engine)

path = an_audio_path()

test_dataset_file = DatasetFile(path=path, file_type='AudioFile')
session.add(test_dataset_file)

test_audio_file = AudioFile(dataset_file=test_dataset_file)
session.add(test_audio_file)

other_path = './tests/data/source/bird1.npz'
test_dataset_file2 = DatasetFile(path=other_path, file_type='SpectrogramFile')
session.add(test_dataset_file2)

test_spect_file = SpectrogramFile(dataset_file=test_dataset_file2)
session.add(test_spect_file)


# In[11]:


queried_audio_file = session.query(AudioFile).first()


# In[17]:


queried_audio_file.dataset_file


# In[ ]:


def test_dataset_file():
    engine = get_engine()
    session = get_session(engine)

    # next line makes tables?
    Base.metadata.create_all(engine)
    
    path = an_audio_path()
    
    test_file = DatasetFile(path=path, file_type='AudioFile')
    session.add(test_file)

    queried_file = session.query(DatasetFile).filter_by(path=path).first()
    assert queried_file.path == path
    assert queried_file.file_type == 'AudioFile'
    assert queried_file.id == 1

    Base.metadata.drop_all(engine)


# In[ ]:


test_dataset_file()


# In[ ]:


def test_audio_file():
    engine = get_engine()
    session = get_session(engine)

    # next line makes tables?
    Base.metadata.create_all(engine)
    
    path = an_audio_path()
    
    test_dataset_file = DatasetFile(path=path, file_type='AudioFile')
    session.add(test_dataset_file)

    test_audio_file = AudioFile(dataset_file=test_dataset_file)
    session.add(test_audio_file)
    
    queried_file = session.query(
        AudioFile
    ).filter_by(
        id=1
    ).first()
    assert queried_file.path == path
    assert queried_file.file_type == 'AudioFile'
    assert queried_file.id == 1

    Base.metadata.drop_all(engine)


# In[ ]:


test_audio_file()


# In[ ]:




