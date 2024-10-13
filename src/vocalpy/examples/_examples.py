"""Functions for working with example data."""

from __future__ import annotations

import importlib.resources
import os
import pathlib
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pooch
import requests.exceptions

from .example_data import ExampleData

# ---- define this at the top since we use it for ExampleMeta
ExampleTypes = Enum("Exampletypes", "Sound Spectrogram Annotation ExampleData")


# ---- define this before `makefuncs` that return ExampleData,
# since it is the return type
@dataclass
class ExampleMeta:
    """Dataclass that represents example data,
    including metadata needed to handle them,
    and metadata specific to the dataset
    that is displayed by VocalPy.

    Attributes
    ----------
    name : str
        Name of example. Displayed by :func:`vocalpy.examples.show`.
        By default, this should be the filename.
        If the filename is not easy enough to type,
        this can be a simple human-readable name,
        and a separate ``fname`` attribute can specify the filename.
    metadata : str
        Metadata for example, that links to source for data.
        Displayed by :func:`vocalpy.examples.show`.
    type: ExampleType
        Type of data. Default is :class:`vocalpy.Sound`.
    fname: str, optional
        Filename. By default, the name should be the filename,
        but if a filename is too verbose (e.g. a .tar.gz file),
        this attribute allows ``name`` to be something human-readable.
        Default is None.
    requires_download: bool
        If True, this example requires download, and we use
        :module:`pooch` to "fetch" it.
    annot_format: str, optional
        For examples that have annotations,
        the annotation format.
        String that is recognized by :module:`crowsetta`
        as a valid annotation format.
    makefunc: callable
        For examples that return :class:`ExampleData`,
        a callable function that returns an :class`ExampleData`
        instance with the example data.
    """

    name: str
    metadata: str
    type: Literal[ExampleTypes.Sound, ExampleTypes.Spectrogram, ExampleTypes.Annotation, ExampleTypes.ExampleData] = (
        ExampleTypes.Sound
    )
    fname: str | None = None
    # next line: lambda returns a new dict that maps 'sound' to 'wav' extension
    requires_download: bool = False
    annot_format: str | None = None
    makefunc: callable | None = None

    def __post_init__(self):
        if not any([self.type is example_type for example_type in ExampleTypes]):
            raise ValueError(f"example type '{self.type}' is not one of the ExampleTypes: {ExampleTypes}")


# ---- all `makefunc`s that return ExampleData for larger example datasets go here
def bfsongrepo_makefunc(
    path: pathlib.Path | list[pathlib.Path], metadata: ExampleMeta, return_path: bool = False
) -> ExampleData:
    import vocalpy  # avoid circular import

    wav_paths = [path for path in path if path.suffix == ".wav"]
    csv_paths = [path for path in path if path.suffix == ".csv"]
    if return_path:
        return ExampleData(sound=wav_paths, annotation=csv_paths)
    else:
        return ExampleData(
            sound=[vocalpy.Sound.read(wav_path) for wav_path in wav_paths],
            annotation=[vocalpy.Annotation.read(csv_path, format=metadata.annot_format) for csv_path in csv_paths],
        )


def jourjine_et_al_2023_makefunc(
    path: pathlib.Path | list[pathlib.Path], metadata: ExampleMeta, return_path: bool = False
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
        return ExampleData(
            sound=vocalpy.Sound.read(wav_path),
            segments=vocalpy.Segments.from_csv(
                csv_path,
                columns_map={"start_seconds": "start_s", "stop_seconds": "stop_s"}
            ),
        )


# ---- now that we've declared all the `makefunc`s we can actually describe all the example data
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
        name="bl26lb16.wav",
        metadata="""Bengalese finch song from:
Bengalese Finch Song Repository.
Nicholson, David; Queen, Jonah E.; J. Sober, Samuel (2017). Bengalese Finch song repository. figshare.
Dataset. https://doi.org/10.6084/m9.figshare.4805749.v9
https://nickledave.github.io/bfsongrepo

In the source dataset, this audio file is named
``bl2616lb16/041912/bl26lb16_190412_0834.20350.wav``.
        """,
    ),
    ExampleMeta(
        name="deermouse-go.wav",
        metadata="""Deer mouse pup calls from:
Jourjine, Nicholas et al. (2023). Data from:
Two pup vocalization types are genetically and functionally separable in deer mice [Dataset].
Dryad. https://doi.org/10.5061/dryad.g79cnp5ts

This is a short clip from the start of the file named
``GO_24860x23748_ltr2_pup3_ch4_4800_m_337_295_fr1_p9_2021-10-02_12-35-01.wav``
in the source dataset.
""",
    ),
    ExampleMeta(
        name="fruitfly-song-multichannel.wav",
        metadata="""courtship song (pulse and sine) of male Drosophila melanogaster. Data from:
Clemens, Jan, 2021, "Drosophila melanogaster - train and test data (multi channel)", https://doi.org/10.25625/8KAKHJ, GRO.data, V1
Contains recordings (on 9 microphone channels) and manual annotations of the courtship song (pulse and sine) of male Drosophila melanogaster.

The recordings were previously unpublished and were first used in:
Clemens J, Coen P, Roemschied FA, Pereira TD, Mazumder D, Aldarondo DE, Pacheco DA, Murthy M. 2018.
Discovery of a New Song Mode in Drosophila Reveals Hidden Structure in the Sensory and Neural Drivers of Behavior.
Current biology 28:2400–2412.e6.

This is a short clip from the file named
``160420_1746_manual.wav`` in the source dataset,
that was clipped using the tsv annotations created with the DAS gui.
To minimize file size, only three of the channels from the original nine are kept.
""",
    ),
    ExampleMeta(
        name="bfsongrepo",
        metadata="""Sample of song from Bengalese Finch Song Repository.
Nicholson, David; Queen, Jonah E.; J. Sober, Samuel (2017). Bengalese Finch song repository. figshare.
Dataset. https://doi.org/10.6084/m9.figshare.4805749.v9
https://nickledave.github.io/bfsongrepo
Files are approximately 20 songs from bird with ID "gy6or6", from the day "032312"
""",
        type=ExampleTypes.ExampleData,
        fname="bfsongrepo.tar.gz",
        requires_download=True,
        makefunc=bfsongrepo_makefunc,
        annot_format="simple-seq",
    ),
    ExampleMeta(
        name="jourjine-et-al-2023",
        metadata="""Sample of deer mouse vocalizations from:
Jourjine, Nicholas et al. (2023). Data from:
Two pup vocalization types are genetically and functionally separable in deer mice [Dataset].
Dryad. https://doi.org/10.5061/dryad.g79cnp5ts
""",
        type=ExampleTypes.ExampleData,
        fname="jourjine-et-al-2023.tar.gz",
        requires_download=True,
        makefunc=jourjine_et_al_2023_makefunc,
        annot_format="simple-seq",
    ),
]


REGISTRY = {example.name: example for example in EXAMPLE_METADATA}

VOCALPY_DATA_DIR = "VOCALPY_DATA_DIR"

ZENODO_DATASET_BASE_URL = "doi:10.5281/zenodo.10685639"

POOCH = pooch.create(
    path=pooch.os_cache("vocalpy"), base_url=ZENODO_DATASET_BASE_URL, registry=None, env=VOCALPY_DATA_DIR
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
    This is useful for :mod:`vocalpy: functionality
    that loads files, or to work with the data
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
    if return_path:
        if example_.type == ExampleTypes.ExampleData:
            return example_.makefunc(path, metadata=example_, return_path=return_path)
        else:
            return path
    else:
        if example_.type == ExampleTypes.Sound:
            return vocalpy.Sound.read(path)
        elif example_.type == ExampleTypes.Spectrogram:
            return vocalpy.Spectrogram.read(path)
        elif example_.type == ExampleTypes.Annotation:
            return vocalpy.Annotation.read(path, format=example_.annot_format)
        elif example_.type == ExampleTypes.ExampleData:
            return example_.makefunc(path, metadata=example_, return_path=return_path)


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
