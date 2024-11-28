"""Class that represents annotation data."""

from __future__ import annotations

import pathlib

import attrs
import crowsetta


@attrs.define
class Annotation:
    """Class that represents annotation data.

    Attributes
    ----------
    data : crowsetta.Annotation
    path : pathlib.Path
    """

    data: crowsetta.Annotation = attrs.field(
        validator=attrs.validators.instance_of(crowsetta.Annotation)
    )
    path: pathlib.Path

    @classmethod
    def read(
        cls, path: str | pathlib.Path, format: str, **kwargs
    ) -> Annotation:
        """Read an annotation from a file

        Parameters
        ----------
        path : str, pathlib.Path
            Path to file containing annotations.
        format : str
            Annotation file format.
            Must be a valid format recognized by :mod:`crowsetta`.
        kwargs : keyword arguments
            Optional keyword arguments.
            If specified, these are
            passed to :meth:`crowsetta.Transcriber.from_file`.

        Returns
        -------
        annot : Annotation
            An :class:`~vocalpy.Annotation` instance
            that contains the annotations read
            from ``path``.
        """
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")

        try:
            scribe = crowsetta.Transcriber(format=format)
        except ValueError as e:
            raise ValueError(
                f"Unable to load format: {format}. Please run `crowsetta.formats.as_list()` "
                "to confirm that it is a valid format name."
            ) from e

        annot_in_format = scribe.from_file(path, **kwargs)
        annot = annot_in_format.to_annot()
        return cls(data=annot, path=path)
