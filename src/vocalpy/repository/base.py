"""Base repository for persisting a dataset to a database."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dataset_file import DatasetFile


class AbstractRepository(abc.ABC):
    @abc.abstractmethod
    def add(self, file: DatasetFile):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, reference) -> DatasetFile:
        raise NotImplementedError
