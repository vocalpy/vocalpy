"""Functions for working with example data."""

from __future__ import annotations

import importlib.resources
import json
import os
import pathlib
import shutil
from enum import Enum
from typing import TYPE_CHECKING, Union

import pooch
import requests.exceptions
from attr import define

from .example_data import ExampleData

if TYPE_CHECKING:
    import vocalpy

    ExampleType = Union[
        pathlib.Path,
        vocalpy.Sound,
        vocalpy.Spectrogram,
        vocalpy.Annotation,
        ExampleData,
    ]


# ---- all `makefunc`s that return ExampleData for larger example datasets go here
def bfsongrepo_makefunc(
    path: pathlib.Path | list[pathlib.Path],
    return_path: bool = False,
    annot_format: str = "simple-seq",
) -> ExampleData:
    import vocalpy  # avoid circular import

    wav_paths = [path for path in path if path.suffix == ".wav"]
    csv_paths = [path for path in path if path.suffix == ".csv"]
    if return_path:
        return ExampleData(sound=wav_paths, annotation=csv_paths)
    else:
        return ExampleData(
            sound=[vocalpy.Sound.read(wav_path) for wav_path in wav_paths],
            annotation=[
                vocalpy.Annotation.read(csv_path, format=annot_format)
                for csv_path in csv_paths
            ],
        )


def jourjine_et_al_2023_makefunc(
    path: pathlib.Path | list[pathlib.Path], return_path: bool = False
) -> ExampleData:
    """Make ``'jourjine-et-al-2023'`` example data"""
    import vocalpy  # avoid circular import

    wav_path = [path for path in path if path.suffix == ".wav"]
    wav_path = wav_path[0]
    csv_path = [path for path in path if path.suffix == ".csv"]
    csv_path = csv_path[0]

    if return_path:
        return ExampleData(sound=wav_path, segments=csv_path)
    else:
        sound = vocalpy.Sound.read(wav_path)
        return ExampleData(
            sound=sound,
            segments=vocalpy.Segments.from_csv(
                csv_path,
                samplerate=sound.samplerate,
                columns_map={
                    "start_seconds": "start_s",
                    "stop_seconds": "stop_s",
                },
            ),
        )


def zblib_makefunc(
    path: pathlib.Path | list[pathlib.Path], return_path: bool = False
) -> ExampleData:
    import vocalpy  # avoid circular import

    wav_paths = [path for path in path if path.suffix == ".wav"]
    if return_path:
        return ExampleData(sound=wav_paths)
    else:
        return ExampleData(
            sound=[vocalpy.Sound.read(wav_path) for wav_path in wav_paths]
        )


MAKEFUNCS = [
    bfsongrepo_makefunc,
    jourjine_et_al_2023_makefunc,
    zblib_makefunc,
]

MAKEFUNCS_MAP = {makefunc.__name__: makefunc for makefunc in MAKEFUNCS}

ExampleTypes = Enum("Exampletypes", "Sound Spectrogram Annotation ExampleData")


@define
class Example:
    """Class that represents example data.

    Attributes
    ----------
    name : str
        Human-readable name of example data
    description : str
        Description of example data,
        including any relevant citations.
    type : ExampleTypes
        Type of data.
        A :class:`Enum` member that is used
        in the :meth:`Example.load` method
        to determine how to load the data.
    requires_download: bool
        If ``True``, this example data requires a download.
        The :meth:`Example.load` method will call :mod:`pooch`.
    filename : str
        For examples that are a single file,
        this is the name of the file.
        For examples that are multiple files,
        this is the name of the archive
        downloaded from Zenodo with :mod:`pooch`.
    path : pathlib.Path, optional
        For examples that are a single file,
        this is the path to the file.
    makefunc : callable, optional
        For examples that are multiple files,
        this is a function that returns an
        :class:`ExampleData` instance
        with attributes containing the
        multiple files.
    makefunc_kwargs : dict, optional
        A :class:`dict` of keyword arguments
        to pass into :attr:`Exanple.makefunc`.

    Notes
    -----
    This dataclass is used to load metadata from
    `vocalpy/examples/example-metadata.json`.
    """

    name: str
    description: str
    type: ExampleTypes
    requires_download: bool
    filename: str | None
    path: pathlib.Path | None = None
    makefunc: callable | None = None
    makefunc_kwargs: dict | None = None

    @classmethod
    def from_metadata(
        cls,
        description_filename: str,
        example_type: str,
        name: str | None = None,
        filename: str | None = None,
        requires_download: bool = False,
        makefunc_name: str | None = None,
        makefunc_kwargs: dict | None = None,
    ):
        """Create a :class:`Example` instance from metadata.

        Parameters
        ----------
        description_filename: str
            Name of text file that contains description of
            examnple data, including any relevant citations.
        example_type: str
            String name of example type,
            that should match one member of the :class:`Enum`
            ``ExampleTypes``.
            The :meth:`Example.load` method uses this
            to determine how to load the example.
        name: str, optional
            A human-readable name for the example.
            If None, defaults to filename.
        filename : str
            For examples that are a single file,
            the name of the file.
        makefunc_name : string, optional
            For examples that are multiple files,
            the name of the function that
            returns an instance of :class:`ExampleData`
            with attributes that contain data loaded
            from the files.
        makefunc_kwargs : dict, optional
            A :class:`dict` of keyword arguments to
            pass into the ``makefunc``.
            Optional, default is None.

        Returns
        -------
        example : Example
            Instance of :class:`Example` dataclass
        """
        if filename is None and name is None:
            raise ValueError(
                "`name` and `filename` for example can't both be None"
            )

        if name is None:
            name = filename

        description_path = importlib.resources.files(
            "vocalpy.examples"
        ).joinpath(description_filename)
        description = description_path.read_text()

        type_ = ExampleTypes[example_type]

        if filename:
            path = importlib.resources.files("vocalpy.examples").joinpath(
                filename
            )

        if makefunc_name is not None:
            makefunc = MAKEFUNCS_MAP[makefunc_name]
        else:
            makefunc = None

        return cls(
            name,
            description,
            type_,
            requires_download,
            filename,
            path,
            makefunc,
            makefunc_kwargs,
        )

    def __attrs_post_init__(self):
        if self.name is None:
            raise ValueError("`name` can't be None")

        if not any(
            [self.type is example_type for example_type in ExampleTypes]
        ):
            raise ValueError(
                f"example type '{self.type}' is not one of the ExampleTypes: {ExampleTypes}"
            )

    def load(self, return_path: bool):
        import vocalpy

        if self.requires_download:
            try:
                POOCH.load_registry_from_doi()
            except requests.exceptions.ConnectionError as e:
                raise ConnectionError(
                    "Unable to connect to registry to download example dataset. "
                    "This may be due to an issue with an internet connection."
                ) from e
            if self.filename.endswith(".tar.gz"):
                path = POOCH.fetch(self.filename, processor=pooch.Untar())
            elif self.filename.endswith(".zip"):
                path = POOCH.fetch(self.filename, processor=pooch.Unzip())
            else:
                path = POOCH.fetch(self.filename)
            if isinstance(path, list):
                path = [pathlib.Path(path_) for path_ in path]
            else:
                path = pathlib.Path(path)
        else:
            path = self.path

        if isinstance(path, list):
            # enforce consisting sorting across platforms
            path = sorted(path)

        if return_path:
            if self.type == ExampleTypes.ExampleData:
                return self.makefunc(path, return_path=return_path)
            else:
                return path
        else:
            if self.type == ExampleTypes.Sound:
                return vocalpy.Sound.read(path)
            elif self.type == ExampleTypes.Spectrogram:
                return vocalpy.Spectrogram.read(path)
            elif self.type == ExampleTypes.Annotation:
                return vocalpy.Annotation.read(path, format=self.annot_format)
            elif self.type == ExampleTypes.ExampleData:
                return self.makefunc(path, return_path=return_path)


EXAMPLE_METADATA_JSON_PATH = pathlib.Path(
    importlib.resources.files("vocalpy.examples").joinpath(
        "example-metadata.json"
    )
)
with EXAMPLE_METADATA_JSON_PATH.open("r") as fp:
    ALL_EXAMPLE_METADATA = json.load(fp)

EXAMPLES = [
    Example.from_metadata(**example_metadata)
    for example_metadata in ALL_EXAMPLE_METADATA
]

REGISTRY = {example_.name: example_ for example_ in EXAMPLES}

# ---- pooch set-up ------------------------------
VOCALPY_DATA_DIR = "VOCALPY_DATA_DIR"

ZENODO_DATASET_BASE_URL = "doi:10.5281/zenodo.10685639"

POOCH = pooch.create(
    path=pooch.os_cache("vocalpy"),
    base_url=ZENODO_DATASET_BASE_URL,
    registry=None,
    env=VOCALPY_DATA_DIR,
)


def get_cache_dir() -> pathlib.Path:
    """Returns path to directory where example data is cached."""
    return os.environ.get(VOCALPY_DATA_DIR, POOCH.abspath)


def clear_cache() -> None:
    """Clears cache, by removing cache dir"""
    cache_dir = get_cache_dir()
    shutil.rmtree(cache_dir)


def example(name: str, return_path: bool = False) -> ExampleType:
    """Get an example from :mod:`vocalpy.examples`.

    To see all available example data, call :func:`vocalpy.examples.show`.

    Parameters
    ----------
    name : str
        Name of the example.
        To see names of examples and the associated metadata,
        call :func:`vocalpy.examples.show`.
    return_path : bool
        If True, return the path to the example data.
        Default is False.

    Returns
    -------
    example : pathlib.Path, vocalpy.Annotation, vocalpy.Sound, vocalpy.Spectrogram, list
        By default, the path or a list of paths to the example data is returned.
        If ``return_type`` is specified,
        then an instance or list of instances of the corresponding VocalPy data type
        will be returned, e.g.,
        a :class:`vocalpy.Sound` instance with the example data
        read into it.

    Examples
    --------

    >>> sound = voc.example('bells.wav')
    >>> spect = voc.spectrogram(sound)
    >>> voc.plot.spect(spect)

    If you want the path(s) to where the data
    is on your local machine,
    set `return_path` to `True`.
    This is useful to demonstrate :mod:`vocalpy` 
    functionality that loads files, or to work with the data
    in the files in some other way.

    >>> sound_path = voc.example('bells.wav', return_path=True)
    >>> sound = voc.Sound.read(sound_path)

    Notes
    -----
    By default, local files will be cached in the directory given by
    :func:`vocalpy.example.get_cache_dir`.  You can override this by setting
    an environment variable ``VOCALPY_DATA_DIR`` prior to importing VocalPy:

    >>> import os
    >>> os.environ['VOCALPY_DATA_DIR'] = '/path/to/store/data'
    >>> import vocalpy as voc

    See Also
    --------
    vocalpy.examples.show
    """
    if name not in REGISTRY:
        raise ValueError(
            f"No example data found with name: {name}. "
            "To see the names of all example data, call `vocalpy.examples.show()`"
        )

    example_: Example = REGISTRY[name]
    return example_.load(return_path=return_path)


def show() -> None:
    """Show example data in :mod:`vocalpy`.

    Prints examples in the form ``name``, ``metadata``
    where ``name`` is the string you would pass to
    :func:`vocalpy.example` to retrieve the example,
    and ``metadata`` is a string containing metadata:
    the source of the data,
    and any relevant citation.

    See Also
    --------
    vocalpy.example

    Notes
    -----
    To retrieve an example, call :func:`vocalpy.example`.
    """
    print("VocalPy example data")
    print("=" * 72)
    for example in EXAMPLES:
        print(
            f"name: {example.name}\n"
            "description:\n"
            f"{example.description}\n"
        )
