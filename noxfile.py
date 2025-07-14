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


# ---- the top half of this noxfile are more standard sessions: lint, tests, docs, build --------------------------
TEST_PYTHONS = [
    "3.11",
    "3.12",
    "3.13",
]


@nox.session(python=TEST_PYTHONS)
def test(session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install(".[test]")
    session.run("pytest", *session.posargs)


@nox.session(python=TEST_PYTHONS[1])
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("isort", "black", "flake8")
    # run isort first since black disagrees with it
    session.run("isort", "./src")
    session.run("black", "./src", "--line-length=79")
    session.run("flake8", "./src", "--max-line-length", "120")


@nox.session(python=TEST_PYTHONS[1])
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
        "pytest", "--cov=./", "--cov-report=xml", *session.posargs
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


# ---- used by sessions that "clean up" data, for example data and for tests
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


@nox.session(name='make-example-data')
def make_example_data(session: nox.Session) -> None:
    """
    Make example data.
    Runs scripts in
    """
    clean_dir("./src/scripts/example_data/")
    if session.posargs:
        session.run(
            "python", 
            "./src/scripts/make_example_data.py",
            "--example-names",
            *session.posargs
        )
    else:
        session.run("python", "./src/scripts/make_example_data.py")

# ---- sessions that have to do with data for tests --------------------------------------------------------------------
# either generating, downloading, or archiving

DATA_FOR_TESTS_DIR = pathlib.Path("./tests/data-for-tests/")
# THIS PATH NEEDS TO BE RELATIVE TO PROJECT ROOT OR WE BREAK TESTS ON CI THAT USE TAR'ED TEST DATA
# i.e., keep as is, don't use the constant in tests.fixtures that involve paths relative to fixtures dir
SOURCE_TEST_DATA_DIR = DATA_FOR_TESTS_DIR / "source"
SOURCE_TEST_DATA_DIRS = [
    dir_ for dir_
    in sorted(pathlib.Path(SOURCE_TEST_DATA_DIR).glob('*/'))
    if dir_.is_dir()
]



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
    Make a .tar.gz file of just the 'source' test data used to run tests.
    """
    session.log(f"Making tarfile with source data: {SOURCE_TEST_DATA_TAR}")
    make_tarfile(SOURCE_TEST_DATA_TAR, SOURCE_TEST_DATA_DIRS)


def is_test_data_subdir_empty(test_data_subdir):
    listdir = [path.name for path in sorted(test_data_subdir.iterdir())]
    return listdir == [".gitkeep"] or len(listdir) < 1


def _test_data_download_source(session) -> None:
    """
    Download and extract a .tar.gz file of 'source' test data, used by TEST_DATA_GENERATE_SCRIPT.

    Helper function used by :func:`test_data_download_source` as well as 
    :func:`dev`.
    """
    session.log(f'Downloading: {SOURCE_TEST_DATA_URL}')
    copy_url(url=SOURCE_TEST_DATA_URL, path=SOURCE_TEST_DATA_TAR)
    session.log(f'Extracting downloaded tar: {SOURCE_TEST_DATA_TAR}')
    with tarfile.open(SOURCE_TEST_DATA_TAR, "r:gz") as tf:
        tf.extractall(path='.')


@nox.session(name='test-data-download-source')
def test_data_download_source(session) -> None:
    """
    Download and extract a .tar.gz file of 'source' test data, used by TEST_DATA_GENERATE_SCRIPT.
    """
    _test_data_download_source(session)


TEST_DATA_GENERATE_AVA_SEGMENTS_SCRIPT = './tests/scripts/generate_ava_segment_test_data/generate_ava_segment_text_files_from_jourjine_et_al_2023.py'


@nox.session(name='test-data-generate-ava-segments', python="3.10")
def test_data_generate_ava_segments(session) -> None:
    """Produce generated test data for ava segments"""
    session.install("joblib==1.3.2")
    session.install("numpy==1.26.3")
    session.install("scipy==1.12.0")
    session.run("python", TEST_DATA_GENERATE_AVA_SEGMENTS_SCRIPT)


TEST_DATA_GENERATE_SCRIPT = './tests/scripts/generate_data_for_tests.py'


@nox.session(name='test-data-generate', python="3.10")
def test_data_generate(session) -> None:
    """
    Produce 'generated' test data, by running TEST_DATA_GENERATE_SCRIPT on 'source' test data.
    """
    session.install(".[test]")
    session.run("python", TEST_DATA_GENERATE_SCRIPT)


# THIS PATH NEEDS TO BE RELATIVE TO PROJECT ROOT OR WE BREAK TESTS ON CI THAT USE TAR'ED TEST DATA
# i.e., keep as is, don't use the constant in tests.fixtures that involve paths relative to fixtures dir
GENERATED_TEST_DATA_DIR = DATA_FOR_TESTS_DIR / "generated"
GENERATED_TEST_DATA_SUBDIRS = [
    dir_ for dir_
    in GENERATED_TEST_DATA_DIR.iterdir()
    if dir_.is_dir()
]
GENERATED_TEST_DATA_TAR = GENERATED_TEST_DATA_DIR / 'generated_test_data.tar.gz'


@nox.session(name='test-data-clean-generated')
def test_data_clean_generated(session) -> None:
    """
    Clean (remove) 'generated' test data.
    """
    for dir_path in GENERATED_TEST_DATA_SUBDIRS:
        clean_dir(dir_path)
    
    if GENERATED_TEST_DATA_TAR.exists():
        GENERATED_TEST_DATA_TAR.unlink()


@nox.session(name='test-data-tar-generated')
def test_data_tar_generated(session) -> None:
    """
    Make a .tar.gz file of all 'generated' test data.
    """
    session.log(f"Making tarfile with all generated data: {GENERATED_TEST_DATA_TAR}")
    make_tarfile(GENERATED_TEST_DATA_TAR, GENERATED_TEST_DATA_SUBDIRS)


GENERATED_TEST_DATA_URL = 'https://osf.io/3fzye/download'


def _test_data_download_generated(session) -> None:
    """
    Download and extract a .tar.gz file of all 'generated' test data

    Helper function used by :func:`test_data_download_generated` 
    as well as :func:`dev`.
    """
    session.log(f'Downloading: {GENERATED_TEST_DATA_URL}')
    copy_url(url=GENERATED_TEST_DATA_URL, path=GENERATED_TEST_DATA_TAR)
    session.log(f'Extracting downloaded tar: {GENERATED_TEST_DATA_TAR}')
    with tarfile.open(GENERATED_TEST_DATA_TAR, "r:gz") as tf:
        tf.extractall(path='.')


@nox.session(name='test-data-download-generated')
def test_data_download_generated(session) -> None:
    """
    Download and extract a .tar.gz file of all 'generated' test data
    """
    _test_data_download_generated(session)


@nox.session(python=TEST_PYTHONS[1])
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

    # ---- if data for tests is not downloaded, then download it
    source_dir_contents = os.listdir(SOURCE_TEST_DATA_DIR)
    if len(source_dir_contents) == 1 and source_dir_contents[0] == ".gitkeep":
        session.log("Found only .gitkeep in ./tests/data-for-tests/source, downloading source data for tests")
        _test_data_download_source(session)
    
    if all(
        [(el == GENERATED_TEST_DATA_DIR / ".gitkeep" or el in GENERATED_TEST_DATA_SUBDIRS)
         for el in GENERATED_TEST_DATA_DIR.iterdir()]
    ) and all(
        [os.listdir(subdir) == [".gitkeep"]
         for subdir in GENERATED_TEST_DATA_SUBDIRS]
    ):
        session.log(
            "Found only .gitkeep in sub-directories of ./tests/data-for-tests/generated, downloading generated data for tests"
        )
        _test_data_download_generated(session)
