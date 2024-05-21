"""Functions for working with example data."""
from __future__ import annotations

import copy
import importlib.resources
import os
import pathlib
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import pooch
import requests.exceptions

if TYPE_CHECKING:
    import vocalpy

    ExampleType = Union[
        pathlib.Path,
        list[pathlib.Path],
        vocalpy.Sound,
        list[vocalpy.Sound],
        vocalpy.Spectrogram | list[vocalpy.Spectrogram],
        vocalpy.Annotation,
        list[vocalpy.Annotation],
    ]


@dataclass
class ExampleMeta:
    """Dataclass that represents example data,
    including metadata needed to handle them,
    and metadata specific to the dataset
    that is displayed by VocalPy.

    Attributes
    ----------
    name : str
        Name of example. Displayed by :func:`vocalpy.examples.list`.
        By default, this should be the filename.
        If the filename is not easy enough to type,
        this can be a simple human-readable name,
        and a separate ``fname`` attribute can specify the filename.
    metadata : str
        Metadata for example, that links to source for data.
        Displayed by :func:`vocalpy.examples.list`.
    type: list[str]
        List of types of data.
        Default is ``["sound"]``.
    fname: str, optional
        Filename. By default, the name should be the filename,
        but if a filename is too verbose (e.g. a .tar.gz file),
        this attribute allows ``name`` to be something human-readable.
        Default is None.
    ext: dict
        A ``dict`` mapping ``type`` to file extensions.
        Default is ``{"sound": ".wav"}``.
    requires_download: bool
        If True, this example requires download, and we use
        :module:`pooch` to "fetch" it.
    annot_format: str, optional
        For examples that have annotations,
        the annotation format.
        String that is recognized by :module:`crowsetta`
        as a valid annotation format.
    """

    name: str
    metadata: str
    type: list[str] = field(default_factory=lambda: copy.copy(["sound"]))
    fname: str | None = None
    # next line: lambda returns a new dict that maps 'sound' to 'wav' extension
    ext: dict = field(default_factory=lambda: copy.copy({"sound": ".wav"}))
    requires_download: bool = False
    annot_format: str | None = None


EXAMPLE_METADATA = [
    ExampleMeta(
        name="bells.wav",
        metadata="Zebra finch song from Sound Analysis Pro website: http://soundanalysispro.com/",
    ),
    ExampleMeta(
        name="flashcam.wav",
        metadata="Zebra finch song from Sound Analysis Pro website: http://soundanalysispro.com/",
    ),
    ExampleMeta(
        name="samba.wav",
        metadata="Zebra finch song from Sound Analysis Pro website: http://soundanalysispro.com/",
    ),
    ExampleMeta(
        name="simple.wav",
        metadata="Zebra finch song from Sound Analysis Pro website: http://soundanalysispro.com/",
    ),
    ExampleMeta(
        name="BM003.wav",
        metadata="""Mouse ultrasonic vocalization from:
Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021).
Data from: Low-dimensional learned feature spaces quantify individual and group differences
in vocal repertoires. Duke Research Data Repository. https://doi.org/10.7924/r4gq6zn8w.
Adapted under Creative Commons License 1.0: https://creativecommons.org/publicdomain/zero/1.0/.
File name in dataset: BM003_day9_air_20s_sparse_chunk007_0297.wav""",
    ),
    ExampleMeta(
        name="bfsongrepo",
        metadata="""Sample of song from Bengalese Finch Song Repository.
Nicholson, David; Queen, Jonah E.; J. Sober, Samuel (2017). Bengalese Finch song repository. figshare.
Dataset. https://doi.org/10.6084/m9.figshare.4805749.v9
https://nickledave.github.io/bfsongrepo
Files are approximately 20 songs from bird with ID "gy6or6", from the day "032312"
""",
        type=["sound", "annotation"],
        fname="bfsongrepo.tar.gz",
        requires_download=True,
        ext={"sound": ".wav", "annotation": ".csv"},
        annot_format="simple-seq",
    ),
    ExampleMeta(
        name="jourjine-et-al-2023",
        metadata="""Sample of deer mouse vocalizations from:
Jourjine, Nicholas et al. (2023). Data from:
Two pup vocalization types are genetically and functionally separable in deer mice [Dataset].
Dryad. https://doi.org/10.5061/dryad.g79cnp5ts
Audio files are 20-second clips from approximately 10 files in the developmentLL data,
(https://datadryad.org/stash/downloads/file_stream/2143657), generated with the script:
tests/scripts/generate_ava_segment_test_data/generate_test_audio_for_ava_segment_from_jourjine_etal_2023.py
""",
        fname="jourjine-et-al-2023.tar.gz",
        requires_download=True,
    ),
]


REGISTRY = {example.name: example for example in EXAMPLE_METADATA}

VOCALPY_DATA_DIR = "VOCALPY_DATA_DIR"

ZENODO_DATASET_BASE_URL = "doi:10.5281/zenodo.10688472"

POOCH = pooch.create(
    path=pooch.os_cache("vocalpy"), base_url=ZENODO_DATASET_BASE_URL, registry=None, env=VOCALPY_DATA_DIR
)


def get_cache_dir() -> pathlib.Path:
    """Returns path to directory where example data is cached.

    By
    """
    return os.environ.get(VOCALPY_DATA_DIR, POOCH.abspath)


def clear_cache() -> None:
    """Clears cache, by removing cache dir"""
    cache_dir = get_cache_dir()
    shutil.rmtree(cache_dir)


VALID_RETURN_TYPE = ("path", "sound", "annotation", "spectrogram")


def example(name: str, return_type: str | None = None) -> ExampleType:
    """Get an example from :mod:`vocalpy.examples`.

    To see all available example data, call :func:`vocalpy.examples.show`.

    By default, local files will be cached in the directory given by
    :func:`vocalpy.example.get_cache_dir`.  You can override this by setting
    an environment variable ``VOCALPY_DATA_DIR`` prior to importing VocalPy:

    >>> import os
    >>> os.environ['VOCALPY_DATA_DIR'] = '/path/to/store/data'
    >>> import vocalpy as voc

    Parameters
    ----------
    name : str
        Name of the example.
        To see names of examples and the associated metadata,
        call :func:`vocalpy.examples.list`.
    return_type : str
        The type to return.
        One of ('path', 'sound', 'annotation', 'spectrogram').
        The default is to return the path to the data.

    Returns
    -------
    example : pathlib.Path, vocalpy.Annotation, vocalpy.Sound, vocalpy.Spectrogram, list
        By default, the path or a list of paths to the example data is returned.
        If ``return_type`` is specified,
        then an instance or list of instances of the corresponding VocalPy data type
        will be returned, e.g.,
        a :class:`vocalpy.Sound` instance with the example data
        read into it.

    See Also
    --------
    vocalpy.examples.show

    Examples
    --------
    >>> sound = voc.example('bells.wav', return_type='sound')
    >>> spect = voc.spectrogram(sound)
    >>> voc.plot.spect(spect)
    """
    if name not in REGISTRY:
        raise ValueError(f"No example found with name: {name}. " f"For a list of examples, call ")
    if return_type is not None:
        if not isinstance(return_type, str) or return_type not in VALID_RETURN_TYPE:
            raise ValueError(
                f"``return_type`` must be a string, one of: {VALID_RETURN_TYPE}. "
                "The default is 'path', that returns a pathlib.Path or list of pathlib.Path instances."
            )
    else:
        return_type = "path"
    import vocalpy  # avoid circular import

    example_: ExampleMeta = REGISTRY[name]
    if example_.requires_download:
        try:
            POOCH.load_registry_from_doi()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                "Unable to connect to registry to download example dataset. "
                "This may be due to an issue with an internet connection."
            ) from e
        if example_.fname.endswith(".tar.gz"):
            path = POOCH.fetch(example_.fname, processor=pooch.Untar())
        else:
            path = POOCH.fetch(example_.fname)
        if isinstance(path, list):
            path = [pathlib.Path(path_) for path_ in path]
        else:
            path = pathlib.Path(path)
    else:
        path = pathlib.Path(importlib.resources.files("vocalpy.examples").joinpath(name))
    if isinstance(path, list):
        # enforce consisting sorting across platforms
        path = sorted(path)
    if return_type == "path":
        return path
    else:
        if return_type not in example_.type:
            raise ValueError(
                f"The ``return_type`` was {return_type}, "
                f"but example '{name}' only has the following return types: {example_.type}"
            )
        if return_type == "sound":
            if isinstance(path, pathlib.Path):
                return vocalpy.Sound.read(path)
            elif isinstance(path, list):
                return [vocalpy.Sound.read(path_) for path_ in path if path_.name.endswith(example_.ext["sound"])]
        elif return_type == "spectrogram":
            if isinstance(path, pathlib.Path):
                return vocalpy.Spectrogram.read(path)
            elif isinstance(path, list):
                return [
                    vocalpy.Spectrogram.read(path_)
                    for path_ in path
                    if path_.name.endswith(example_.ext["spectrogram"])
                ]
        elif return_type == "annotation":
            if isinstance(path, pathlib.Path):
                return vocalpy.Annotation.read(path)
            elif isinstance(path, list):
                return [
                    vocalpy.Annotation.read(path_, format=example_.annot_format)
                    for path_ in path
                    if path_.name.endswith(example_.ext["annotation"])
                ]


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
    print("Examples built into VocalPy")
    print("=" * 72)
    for example_meta in EXAMPLE_METADATA:
        print(f"name: {example_meta.name}\n" "metadata:\n" f"{example_meta.metadata}\n")
