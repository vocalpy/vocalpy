"""Make example data.

This script makes the data in this Zenodo dataset:
https://zenodo.org/records/10685640

We host example data there that is too big to go as a single file into
``./src/vocalpy/examples``.
Example data should go here if it's (1) more than one file or (2) a file larger than 500K.

Don't run this directly.
Instead run ``nox -s make-example-data``,
that first cleans the directory `./src/scripts/example_data`,
and then runs this script.

The script does the following:
1. For each `ExampleData`
2. create a sub-directory inside `./src/scripts/example_data`
3. and download the example data into it, removing files if necessary
4. then make a .tar.gz archive from files in the directory with the specified extensions

After this process completes,
you need to upload the .tar.gz files to the Zenodo dataset
to make a new version of the dataset.

# Example data prepared by this script (name here should match name in vocalpy example metadata)

## bfsongrepo

1. We download one day of data from the dataset, using code adapted from
https://github.com/NickleDave/bfsongrepo/blob/main/src/scripts/download_dataset.py
2. We take the first ten .wav and .csv files from the directory 022212
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import shutil
import sys
import tarfile
import time
import urllib.request
import warnings


def reporthook(count: int, block_size: int, total_size: int) -> None:
    """hook for urlretrieve that gives us a simple progress report
    https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(
        "\r...%d%%, %d MB, %d KB/s, %d seconds passed"
        % (percent, progress_size / (1024 * 1024), speed, duration)
    )
    sys.stdout.flush()


BFSONGREPO_DATA_TO_DOWNLOAD = {
    "gy6or6": {
        "sober.repo1.gy6or6.032212.wav.csv.tar.gz": {
            "MD5": "9cff3aefc45607617e61f7447dbc8800",
            "download": "https://figshare.com/ndownloader/files/41668980",
        },
    }
}


def download_bfsongrepo_data(
    download_urls_by_bird_ID: dict, dst: pathlib.Path
) -> None:
    """Download part of bfsgonrepo dataset, given a dict of download urls"""
    tar_dir = dst / "bfsongrepo-tars"
    tar_dir.mkdir()
    # top-level keys are bird ID: bl26lb16, gr41rd51, ...
    for bird_id, tars_dict in download_urls_by_bird_ID.items():
        print(f"Downloading .tar files for bird: {bird_id}")
        # bird ID -> dict
        # where keys are .tar.gz filenames mapping to download url + MD5 hash
        for tar_name, url_md5_dict in tars_dict.items():
            print(f"Downloading tar: {tar_name}")
            download_url = url_md5_dict["download"]
            filename = tar_dir / tar_name
            urllib.request.urlretrieve(download_url, filename, reporthook)
            print("\n")


def extract_bfsongrepo_tars(bfsongrepo_dir: pathlib.Path) -> None:
    tar_dir = (
        bfsongrepo_dir / "bfsongrepo-tars"
    )  # made by download_dataset function
    tars = sorted(tar_dir.glob("*.tar.gz"))
    for tar_path in tars:
        print(f"\nunpacking: {tar_path}")

        shutil.unpack_archive(
            filename=tar_path, extract_dir=bfsongrepo_dir, format="gztar"
        )


def download_and_extract_bfsongrepo_tar(dst: str | pathlib.Path) -> None:
    """Downloads and extracts bfsongrepo tar"""
    dst = pathlib.Path(dst).expanduser().resolve()
    if not dst.is_dir():
        raise NotADirectoryError(
            f"Value for 'dst' argument not recognized as a directory: {dst}"
        )
    bfsongrepo_dir = dst / "bfsongrepo"
    if bfsongrepo_dir.exists():
        warnings.warn(
            f"Directory already exists: {bfsongrepo_dir}\n"
            "Will download and write over any existing files. Press Ctrl-C to stop.",
            stacklevel=2,
        )

    try:
        bfsongrepo_dir.mkdir(exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"Unable to create directory in 'dst': {dst}\n"
            "Please try running with 'sudo' on Unix systems or as Administrator on Windows systems.\n"
            "If that fails, please download files for tutorial manually from the 'download' links in tutorial page."
        ) from e

    print(
        f"Downloading Bengalese Finch Song Repository data to: {bfsongrepo_dir}"
    )

    download_bfsongrepo_data(BFSONGREPO_DATA_TO_DOWNLOAD, bfsongrepo_dir)
    extract_bfsongrepo_tars(bfsongrepo_dir)


def tar_source_data_subdir(
    dataset_root,
    tar_dst,
    archive_name,
    ext=None,
    dry_run=False,
    skip_exists=False,
):
    if ext is None:
        ext = ["wav"]
    dataset_root = pathlib.Path(dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise NotADirectoryError(f"Dataset root not found: {dataset_root}")
    tar_dst = pathlib.Path(tar_dst).expanduser().resolve()
    if not tar_dst.exists():
        raise NotADirectoryError(f".tar destination root not found: {tar_dst}")

    archive_path = tar_dst / f"{archive_name}.tar.gz"
    print(f"will create archive: {archive_path}")

    if not dry_run:
        if skip_exists:
            if archive_path.exists():
                print("Archive exists already, skipping.")
                return

        print("Adding files to archive.")
        with tarfile.open(archive_path, "w:gz") as tf:
            for ext_ in ext:
                paths = sorted(dataset_root.glob(f"*{ext_}"))
                for path in paths:
                    print(f"Adding: {path.name}")
                    arcname = str(path).replace(str(dataset_root) + "/", "")
                    tf.add(name=path, arcname=arcname)


HERE = pathlib.Path(__file__).parent
EXAMPLE_DATA_DST = HERE / "example_data"
BFSONGREPO_ROOT = EXAMPLE_DATA_DST / "bfsongrepo"
N_BFSONGREPO_FILES = 10


def make_bfsongrepo_data_dir():
    # ---- download bfsongrepo data
    download_and_extract_bfsongrepo_tar(dst=EXAMPLE_DATA_DST)

    # ---- get only files we want from bfsongrepo
    dir_we_want = BFSONGREPO_ROOT / "gy6or6" / "032212"
    wav_paths = sorted(dir_we_want.glob("*.wav"))
    csv_paths = sorted(dir_we_want.glob("*.csv"))
    wav_paths = wav_paths[:N_BFSONGREPO_FILES]
    csv_paths = csv_paths[:N_BFSONGREPO_FILES]
    for path in wav_paths + csv_paths:
        shutil.move(path, BFSONGREPO_ROOT)

    # get rid of the rest of the files now that we've moved the ones we want
    shutil.rmtree(BFSONGREPO_ROOT / "gy6or6")


JOURJINE_ET_AL_2023_SUBSET_URLS = {
    "GO_24860x23748_ltr2_pup3_ch4_4800_m_337_295_fr1_p9_2021-10-02_12-35-01.wav": "https://www.dropbox.com/scl/fi/ysfo9scult6rtiilof07z/GO_24860x23748_ltr2_pup3_ch4_4800_m_337_295_fr1_p9_2021-10-02_12-35-01.wav?rlkey=wnyz3mczz94sn38r534q5e26e&st=tvh2nyu0&dl=1",  # noqa: E501
    "GO_24860x23748_ltr2_pup3_ch4_4800_m_337_295_fr1_p9_2021-10-02_12-35-01.csv": "https://drive.google.com/uc?export=download&id=1T_PqiaWZrIIxmu7mtq_BiUfgPEj9Z2kj",  # noqa: E501
}

JOURJINE_ET_AL_2023_SUBSET_DST = EXAMPLE_DATA_DST / "jourjine-et-al-2023"


def make_jourjine_et_al_2023(
    dst=JOURJINE_ET_AL_2023_SUBSET_DST,
):
    print(
        f"Making directory for jourjine-et-al-2023 data:\n{JOURJINE_ET_AL_2023_SUBSET_DST}"
    )
    JOURJINE_ET_AL_2023_SUBSET_DST.mkdir(exist_ok=True)
    for filename, url in JOURJINE_ET_AL_2023_SUBSET_URLS.items():
        print(f"Downloading file:\n{filename}\nFrom url:\n{url}")
        response = urllib.request.urlopen(url)
        with (dst / filename).open("wb") as fp:
            fp.write(response.read())


REPO_ROOT = HERE / ".." / ".."
SOURCE_TEST_DATA_ROOT = REPO_ROOT / "tests/data-for-tests/source"


@dataclasses.dataclass
class ExampleData:
    """Dataclass that represents example dataset that this script will make

    Attributes
    ----------
    name : str
        String name used to refer to example dataset.
        Becomes the name of the .tar.gz archive.
    dir_ : str or pathlib.Path
        Path to directory to put into .tar.gz archive.
        This is the archive that gets downloaded by
        `pooch` when a user calls `vocalpy.example`
        with the name of the dataset.
    ext : list
        List of string, the extensions of the files
        that should go into the .tar.gz file.
    makefunc : callable
        A callable that makes the the directory
        specified by `dir_`, if it doesn't exist already.
    """

    name: str
    dir_: str | pathlib.Path
    ext: list[str]
    makefunc: callable | None


EXAMPLE_DATA = [
    ExampleData(
        name="bfsongrepo",
        dir_=BFSONGREPO_ROOT,
        ext=[".wav", ".csv"],
        makefunc=make_bfsongrepo_data_dir,
    ),
    ExampleData(
        name="jourjine-et-al-2023",
        dir_=JOURJINE_ET_AL_2023_SUBSET_DST,
        ext=[".wav", ".csv"],
        makefunc=make_jourjine_et_al_2023,
    ),
]


EXAMPLE_DATA_NAME_MAP = {
    example_data.name: example_data for example_data in EXAMPLE_DATA
}
EXAMPLE_DATA_NAMES = list(EXAMPLE_DATA_NAME_MAP.keys())


def main(example_names: list[str]) -> None:
    # -- validate args
    for example_name in example_names:
        if example_name not in EXAMPLE_DATA_NAMES:
            raise ValueError(
                f"Invalid name for example dataset: {example_name}.\n"
                f"Valid names are: {EXAMPLE_DATA_NAMES}"
            )

    # -- make .tar.gz files
    for example_name in example_names:
        example_data = EXAMPLE_DATA_NAME_MAP[example_name]
        if example_data.makefunc:
            example_data.makefunc()
        tar_source_data_subdir(
            dataset_root=example_data.dir_,
            tar_dst=EXAMPLE_DATA_DST,
            archive_name=example_data.name,
            ext=example_data.ext,
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example-names",
        nargs="+",
        default=EXAMPLE_DATA_NAMES,
        choices=EXAMPLE_DATA_NAMES,
    )
    args = parser.parse_args()
    return args


args = get_args()
main(args.example_names)
