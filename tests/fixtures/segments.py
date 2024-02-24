import json
import pathlib

from .test_data import GENERATED_TEST_DATA_ROOT, SOURCE_TEST_DATA_ROOT


SOURCE_SEGMENT_ROOT = SOURCE_TEST_DATA_ROOT / 'segment'
EVSONGANALY_SEGMENT_DIR = SOURCE_SEGMENT_ROOT / 'evsonganaly'
EVSONGANALY_SEGMENT_MAT_LIST = sorted(EVSONGANALY_SEGMENT_DIR.glob('*.mat'))

GENERATED_SEGMENT_ROOT = GENERATED_TEST_DATA_ROOT / 'segment'
AVA_SEGMENT_TXT_DIR = GENERATED_SEGMENT_ROOT / 'ava-segment-txt'
AVA_SEGMENT_TXT_LIST = sorted(AVA_SEGMENT_TXT_DIR.glob('*.txt'))
EVSONGANALY_SEGMENT_DIR = GENERATED_SEGMENT_ROOT / 'evsonganaly'
EVSONGANALY_SEGMENT_JSON_PATH = EVSONGANALY_SEGMENT_DIR / 'evsonganaly-cbin-notmat-segment-mat-paths.json'
with EVSONGANALY_SEGMENT_JSON_PATH.open('r') as fp:
    EVSONGANALY_SEGMENT_JSON = json.load(fp)
# convert all key-value pairs in dicts from json to the form {'name_path': pathlib.Path(string path)}
EVSONGANALY_SEGMENT_JSON = [
    {k: pathlib.Path(v) for k,v in dict_.items()}
    for dict_ in EVSONGANALY_SEGMENT_JSON
]
