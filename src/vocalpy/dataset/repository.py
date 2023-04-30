"""Module that implements the repository pattern.

"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain_model import DatasetFile


class AbstractRepository(abc.ABC):
    @abc.abstractmethod
    def add(self, file: DatasetFile):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, reference) -> DatasetFile:
        raise NotImplementedError


class SqlAlchemyRepository(AbstractRepository):
    def __init__(self, session):
        self.session = session

    def add(self, file: DatasetFile):
        self.session.add(file)

    def get(self, reference):
        from ..domain_model import DatasetFile
        return self.session.query(DatasetFile).filter_by(reference=reference).one()

    def list(self):
        from ..domain_model import DatasetFile
        return self.session.query(DatasetFile).all()
