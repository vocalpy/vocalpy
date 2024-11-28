import argparse

from .fixtures import *


def pytest_addoption(parser):
    parser.addoption(
        "--biosound-fast",
        default=True, 
        action=argparse.BooleanOptionalAction,
        help=(
            "If `--biosound-fast` flag is set, only test functionality that uses `vocalpy.feature.biosound` with 1-2 files, "
            "so tests run fast. Default is to set flag. To run with all files, set flag `--no-biosound-fast`."
            )
        )


def pytest_generate_tests(metafunc):
    if "all_elie_theunissen_2016_wav_paths" in metafunc.fixturenames:
        if metafunc.config.getoption("--biosound-fast"):
            metafunc.parametrize("all_elie_theunissen_2016_wav_paths", ELIE_THEUNISSEN_2016_WAV_LIST[:1])
        else:
            metafunc.parametrize("all_elie_theunissen_2016_wav_paths", ELIE_THEUNISSEN_2016_WAV_LIST)
