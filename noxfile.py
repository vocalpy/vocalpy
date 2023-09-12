import os
import pathlib
import shutil
import sys
import tarfile
import urllib.request

import nox


# ---- this constant is used by more than one session: dev, download-test-data
DIR = pathlib.Path(__file__).parent.resolve()

VENV_DIR = pathlib.Path('./.venv').resolve()


# ---- the top half of this noxfile are more standard sessions: dev, lint, tests, docs, build --------------------------

@nox.session(python="3.10")
def dev(session: nox.Session) -> None:
    """
    Sets up a python development environment for the project.

    This session will:
    - Create a python virtualenv for the session
    - Install the `virtualenv` cli tool into this environment
    - Use `virtualenv` to create a global project virtual environment
    - Invoke the python interpreter from the global project environment to install
      the project and all it's development dependencies.
    """

    session.install("virtualenv")
    # VENV_DIR here is a pathlib.Path location of the project virtualenv
    # e.g. .venv
    session.run("virtualenv", os.fsdecode(VENV_DIR), silent=True)

    if sys.platform.startswith("linux") or sys.platform == "darwin":
        python = os.fsdecode(VENV_DIR.joinpath("bin/python"))
    elif sys.platform.startswith("win"):
        python = os.fsdecode(VENV_DIR.joinpath("Scripts/python.exe"))

    # Use the venv's interpreter to install the project along with
    # all it's dev dependencies, this ensures it's installed in the right way
    session.run(python, "-m", "pip", "install", "-e", ".[dev]", external=True)


TEST_PYTHONS = [
    "3.9",
    "3.10",
    "3.11"
]


@nox.session(python=TEST_PYTHONS)
def test(session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install(".[test]")
    session.run("pytest", "-n", "auto", *session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session
def docs(session: nox.Session) -> None:
    """
    Build the docs.
    """
    session.install(".[doc]")

    if session.posargs:
        if "autobuild" in session.posargs:
            print("Building docs at http://127.0.0.1:8000 with sphinx-autobuild -- use Ctrl-C to quit")
            session.run("sphinx-autobuild", "docs", "docs/_build/html")
        else:
            print("Unsupported argument to docs")
    else:
        session.run("sphinx-build", "-nW", "--keep-going", "-b", "html", "docs/", "docs/_build/html")


@nox.session
def coverage(session: nox.Session) -> None:
    """Run tests and measure coverage"""
    session.run(
        "pytest", "-n", "auto", "--cov=./", "--cov-report=xml", *session.posargs
    )


@nox.session
def build(session: nox.Session) -> None:
    """
    Build an SDist and wheel with ``flit``.
    """
    dist_p = DIR.joinpath("dist")
    if dist_p.exists():
        shutil.rmtree(dist_p)

    session.install("flit")
    session.run("flit", "build")


# ---- the bottom half of the noxfile, the rest of the sessions have to do with data for tests -------------------------
# either generating, downloading, or archiving

DATA_FOR_TESTS_DIR = pathlib.Path("./tests/data-for-tests/")
SOURCE_TEST_DATA_DIR = DATA_FOR_TESTS_DIR / "source"
SOURCE_TEST_DATA_DIRS = [
    dir_ for dir_
    in sorted(pathlib.Path(SOURCE_TEST_DATA_DIR).glob('*/'))
    if dir_.is_dir()
]


# ---- used by sessions that "clean up" data for tests
def clean_dir(dir_path):
    """
    "clean" a directory by removing all files
    (that are not hidden)
    without removing the directory itself
    """
    dir_path = pathlib.Path(dir_path)
    dir_contents = dir_path.glob('*')
    for content in dir_contents:
        if content.is_dir():
            shutil.rmtree(content)
        else:
            if content.name.startswith('.'):
                # e.g., .gitkeep file we don't want to delete
                continue
            content.unlink()


@nox.session(name='test-data-clean-source')
def test_data_clean_source(session) -> None:
    """
    Clean (remove) 'source' test data, used by TEST_DATA_GENERATE_SCRIPT.
    """
    clean_dir(SOURCE_TEST_DATA_DIR)


def copy_url(url: str, path: str) -> None:
    """Copy data from a url to a local file."""
    urllib.request.urlretrieve(url, path)


def make_tarfile(name: str, to_add: list):
    with tarfile.open(name, "w:gz") as tf:
        for add_name in to_add:
            tf.add(name=add_name)


SOURCE_TEST_DATA_URL = "https://osf.io/xq6hf/download"
SOURCE_TEST_DATA_TAR = SOURCE_TEST_DATA_DIR / "source-test-data.tar.gz"


@nox.session(name='test-data-tar-source')
def test_data_tar_source(session) -> None:
    """
    Make a .tar.gz file of just the 'generated' test data used to run tests on CI.
    """
    session.log(f"Making tarfile with source data: {SOURCE_TEST_DATA_TAR}")
    make_tarfile(SOURCE_TEST_DATA_TAR, SOURCE_TEST_DATA_DIRS)


def is_test_data_subdir_empty(test_data_subdir):
    listdir = [path.name for path in sorted(test_data_subdir.iterdir())]
    return listdir == [".gitkeep"] or len(listdir) < 1


@nox.session(name='test-data-download-source')
def test_data_download_source(session) -> None:
    """
    Download and extract a .tar.gz file of 'source' test data, used by TEST_DATA_GENERATE_SCRIPT.
    """
    session.log(f'Downloading: {SOURCE_TEST_DATA_URL}')
    copy_url(url=SOURCE_TEST_DATA_URL, path=SOURCE_TEST_DATA_TAR)
    session.log(f'Extracting downloaded tar: {SOURCE_TEST_DATA_TAR}')
    with tarfile.open(SOURCE_TEST_DATA_TAR, "r:gz") as tf:
        tf.extractall(path='.')



TEST_DATA_GENERATE_SCRIPT = './tests/scripts/generate_data_for_tests.py'


@nox.session(name='test-data-generate', python="3.10")
def test_data_generate(session) -> None:
    """
    Produced 'generated' test data, by running TEST_DATA_GENERATE_SCRIPT on 'source' test data.
    """
    session.install(".[test]")
    session.run("python", TEST_DATA_GENERATE_SCRIPT)


# TODO: fix this url!
GENERATED_TEST_DATA_DIR = DATA_FOR_TESTS_DIR / "generated"
GENERATED_TEST_DATA_SUBDIRS = [
    dir_ for dir_
    in sorted(pathlib.Path(GENERATED_TEST_DATA_DIR).glob('*/'))
    if dir_.is_dir()
]
GENERATED_TEST_DATA_TAR = GENERATED_TEST_DATA_DIR / 'generated_test_data.tar.gz'


@nox.session(name='test-data-clean-generated')
def test_data_clean_generated(session) -> None:
    """
    Clean (remove) 'generated' test data.
    """
    clean_dir(GENERATED_TEST_DATA_DIR)


@nox.session(name='test-data-tar-generated')
def test_data_tar_generated(session) -> None:
    """
    Make a .tar.gz file of all 'generated' test data.
    """
    session.log(f"Making tarfile with all generated data: {GENERATED_TEST_DATA_TAR}")
    make_tarfile(GENERATED_TEST_DATA_TAR, GENERATED_TEST_DATA_SUBDIRS)


GENERATED_TEST_DATA_URL = 'https://osf.io/3fzye/download'


@nox.session(name='test-data-download-generated')
def test_data_download_generated(session) -> None:
    """
    Download and extract a .tar.gz file of all 'generated' test data
    """
    session.log(f'Downloading: {GENERATED_TEST_DATA_URL}')
    copy_url(url=GENERATED_TEST_DATA_URL, path=GENERATED_TEST_DATA_TAR)
    session.log(f'Extracting downloaded tar: {GENERATED_TEST_DATA_TAR}')
    with tarfile.open(GENERATED_TEST_DATA_TAR, "r:gz") as tf:
        tf.extractall(path='.')
