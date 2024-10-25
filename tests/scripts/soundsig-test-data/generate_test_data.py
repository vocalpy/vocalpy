import pathlib

from soundsig.sound import BioSound

TEST_DATA_ROOT = pathlib.Path(__file__).parent / ".." / ".." / "data-for-tests"
assert TEST_DATA_ROOT.exists()

print("BioSound is:", BioSound)
