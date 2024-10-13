import json
import pathlib

from .test_data import GENERATED_TEST_DATA_ROOT, SOURCE_TEST_DATA_ROOT


# ---- copied from evfuncs: oracle data for testing the evsonganaly segmenting method
# that we now call 'meansquared'
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

# ---- ava segmentation from Jourjine et al. 2023
JOURJINE_ET_AL_2023_GO_SEGMENT_CSV_PATH = SOURCE_TEST_DATA_ROOT /  "jourjine-et-al-2023/GO/GO_24860x23748_ltr2_pup3_ch4_4800_m_337_295_fr1_p9_2021-10-02_12-35-01.csv"