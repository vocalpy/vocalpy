import shutil
from pathlib import Path

import nox

nox.options.sessions = ["lint", "tests"]

DIR = Path(__file__).parent.resolve()

DATA_FOR_TESTS_DIR = Path("./tests/data_for_tests/")
SOURCE_TEST_DATA_DIR = DATA_FOR_TESTS_DIR / "source"
GENERATED_TEST_DATA_DIR = DATA_FOR_TESTS_DIR / "generated"

SOURCE_TEST_DATA_URL = "https://osf.io/ubhsj/download"

SOURCE_TEST_DATA_TAR = SOURCE_TEST_DATA_DIR / "source_test_data.tar.gz"
# GENERATED_TEST_DATA_TAR = GENERATED_TEST_DATA_DIR / 'generated_test_data.tar.gz'


def is_test_data_subdir_empty(test_data_subdir):
    listdir = [path.name for path in sorted(test_data_subdir.iterdir())]
    return listdir == [".gitkeep"] or len(listdir) < 1


@nox.session
def download_test_data(session: nox.Session) -> None:
    """
    Download data for tests.
    """
    session.run(
        "wget", "-q", f"{SOURCE_TEST_DATA_URL}", "-O", f"{SOURCE_TEST_DATA_TAR}"
    )
    session.run("tar", "-xzf", f"{SOURCE_TEST_DATA_TAR}")


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit tests.
    """
    if is_test_data_subdir_empty(SOURCE_TEST_DATA_DIR):
        download_test_data(session)

    session.install("-e", ".[test]")
    if session.posargs:
        session.run("pytest", *session.posargs)
    else:
        session.run("pytest")


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
    session.install("-e", ".[doc]")

    if session.posargs:
        if "serve" in session.posargs:
            session.run("mkdocs", "serve")
        else:
            session.error("Unrecognized args, use 'serve'")
    else:
        session.run("mkdocs", "build")


@nox.session
def coverage(session: nox.Session) -> None:
    """Run tests and measure coverage"""
    session.run(
        "pytest",
        "--cov=./",
        "--cov-report=xml",
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
