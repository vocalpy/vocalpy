"""Repository that uses SQLAlchemy."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import AbstractRepository

if TYPE_CHECKING:
    from ..dataset_file import DatasetFile


class SqlAlchemyRepository(AbstractRepository):
    def __init__(self, session):
        self.session = session

    def add(self, file: DatasetFile):
        self.session.add(file)

    def get(self, reference):
        from ..dataset_file import DatasetFile

        return self.session.query(DatasetFile).filter_by(reference=reference).one()

    def list(self):
        from ..dataset_file import DatasetFile

        return self.session.query(DatasetFile).all()
