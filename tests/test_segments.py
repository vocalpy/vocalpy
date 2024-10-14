import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import vocalpy.segments

from .fixtures.audio import AUDIO_LIST_CBIN, JOURJINE_ET_AL_GO_WAV_PATH
from .fixtures.segments import JOURJINE_ET_AL_2023_GO_SEGMENT_CSV_PATH


@pytest.fixture
def bfsongrepo_segments_csv_path(tmp_path):
    sound = vocalpy.Sound.read(AUDIO_LIST_CBIN[0])
    segments = vocalpy.segment.meansquared(sound, threshold=5000, min_dur=0.02, min_silent_dur=0.004)
    segments_df = pd.DataFrame.from_records(
        dict(start_ind=segments.start_inds, length=segments.lengths)
    )
    bfsongrepo_segments_csv_path = tmp_path / f"{AUDIO_LIST_CBIN[0].name}.segments.csv"
    segments_df.to_csv(bfsongrepo_segments_csv_path, index=False)
    return bfsongrepo_segments_csv_path


TEST_SOUND = vocalpy.Sound(
    data=np.random.rand(1, 32000),
    samplerate=32000,
)

TEST_SOUND_DATA = xr.DataArray(TEST_SOUND.data)


# we use these to parametrize methods on TestSegments class
SEGMENTS_ARGNAMES = 'start_inds, lengths, samplerate, labels'
SEGMENTS_ARGVALS = [
    (
        np.array([0, 10, 20, 30, 40]),
        np.array([10, 10, 10, 10, 10]),
        TEST_SOUND.samplerate,
        [''] * 5,
    ),
    (
        np.array([0, 10, 20, 30, 40]),
        np.array([10, 10, 10, 10, 10]),
        TEST_SOUND.samplerate,
        None,
    ),
    # empty segments
    (
        np.array([]).astype(int),
        np.array([]).astype(int),
        TEST_SOUND.samplerate,
        None,
    ),
    (
        np.array([0, 10, 20, 30, 40]),
        np.array([10, 10, 10, 10, 10]),
        TEST_SOUND.samplerate,
        list('abcde'),
    ),
]
SEGMENTS_FOR_FIXTURE = [
    vocalpy.Segments(**{
        k: v
        for k, v in zip(SEGMENTS_ARGNAMES.split(sep=', '), argvals)
    })
    for argvals in SEGMENTS_ARGVALS
]
@pytest.fixture(params=SEGMENTS_FOR_FIXTURE)
def a_segments(request):
    return request.param


class TestSegments:
    @pytest.mark.parametrize(
        SEGMENTS_ARGNAMES,
        SEGMENTS_ARGVALS
    )
    def test_init(self, start_inds, lengths, samplerate, labels):
        if labels is not None:
            segments = vocalpy.segments.Segments(
                start_inds,
                lengths,
                samplerate,
                labels,
            )
        else:
            # test we get default labels, empty strings
            segments = vocalpy.segments.Segments(
                start_inds,
                lengths,
                samplerate,
            )
        assert isinstance(segments, vocalpy.segments.Segments)
        for attr_name, attr_val in zip(
            ['start_inds', 'lengths', 'samplerate', 'labels'],
            [start_inds, lengths, samplerate, labels],
        ):
            if attr_name == 'labels' and attr_val is None:
                # have to special case this, to test attr_val is correct
                attr_val = np.array([''] * start_inds.shape[0])

            if isinstance(attr_val, np.ndarray):
                assert np.array_equal(
                    getattr(segments, attr_name),
                    attr_val,
                )
            else:
                assert getattr(segments, attr_name) == attr_val

    @pytest.mark.parametrize(
        'start_inds, lengths, samplerate, labels, expected_exception',
        # test cases should be in order of pre-conditions in __init__
        # so we can hope to make sense of this giant list
        [
            # start inds not numpy array
            (
                [0, 10, 20, 30, 40],
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND.samplerate,
                [''] * 5,
                TypeError
            ),
            # lengths not numpy array
            (
                np.array([0, 10, 20, 30, 40]),
                [10, 10, 10, 10, 10],
                TEST_SOUND.samplerate,
                [''] * 5,
                TypeError
            ),
            # start inds not dtype int
            (
                np.array([0, 10, 20, 30, 40]).astype(np.float32),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND.samplerate,
                [''] * 5,
                ValueError
            ),
            # lengths not dtype int
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([10, 10, 10, 10, 10]).astype(np.float32),
                TEST_SOUND.samplerate,
                [''] * 5,
                ValueError
            ),
            # start inds not 1-d
            (
                np.array([[0, 10, 20, 30, 40]]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND.samplerate,
                [''] * 5,
                ValueError
            ),
            # start inds not 1-d
            (
                np.array(0),
                np.array([10]),
                TEST_SOUND.samplerate,
                [''] * 5,
                ValueError
            ),
            # lengths not 1-d
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([[10, 10, 10, 10, 10]]),
                TEST_SOUND.samplerate,
                [''] * 5,
                ValueError
            ),
            # lengths not 1-d
            (
                np.array([0]),
                np.array(10),
                TEST_SOUND.samplerate,
                [''] * 5,
                ValueError
            ),
            # start_inds.size != lengths.size
            (
                np.array([0, 10]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND.samplerate,
                [''] * 5,
                ValueError
            ),
            # start inds not all non-negative
            (
                np.array([-10, 10, 20, 30, 40]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND.samplerate,
                [''] * 5,
                ValueError
            ),
            # start inds not monotonically increasing
            (
                np.array([40, 30, 20, 10, 0]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND.samplerate,
                [''] * 5,
                ValueError
            ),
            # lengths not all positive
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([0, 10, 10, 10, 10]),
                TEST_SOUND.samplerate,
                [''] * 5,
                ValueError
            ),
            # lengths not all positive
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([-1, 10, 10, 10, 10]),
                TEST_SOUND.samplerate,
                [''] * 5,
                ValueError
            ),

            # labels not list
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND.samplerate,
                np.array([''] * 5),
                TypeError
            ),
            # labels not str
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND.samplerate,
                [5] * 5,
                ValueError
            ),
            # len(labels) != start_inds.size
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND.samplerate,
                [''] * 2,
                ValueError
            ),
        ],
    )
    def test_init_raises(self, start_inds, lengths, samplerate, labels, expected_exception):
        with pytest.raises(expected_exception):
            vocalpy.segments.Segments(
                start_inds,
                lengths,
                samplerate,
                labels,
            )

    def test_stop_inds_property(self, a_segments):
        assert np.array_equal(
            a_segments.stop_inds,
            a_segments.start_inds + a_segments.lengths
        )

    def test_start_times_property(self, a_segments):
        assert np.array_equal(
            a_segments.start_times,
            a_segments.start_inds / a_segments.samplerate
        )

    def test_durations_property(self, a_segments):
        assert np.array_equal(
            a_segments.durations,
            a_segments.lengths / a_segments.samplerate
        )

    def test_stop_times_property(self, a_segments):
        assert np.array_equal(
            a_segments.stop_times,
            a_segments.start_times + a_segments.durations
        )

    def test_all_times_property(self, a_segments):
        assert np.array_equal(
            a_segments.all_times,
            np.unique(
                np.concatenate(
                    (a_segments.start_times, a_segments.stop_times)
                )
            )
        )

    def test_to_json(self, a_segments, tmp_path):
        json_path = tmp_path / 'a_segments.json'
        a_segments.to_json(json_path)
        assert json_path.exists

        with json_path.open('r') as fp:
            json_dict = json.load(fp)

        for key in ('start_inds', 'lengths', 'samplerate', 'labels'):
            assert key in json_dict

        assert np.array_equal(
            a_segments.start_inds,
            np.array(json_dict['start_inds'])
        )
        assert np.array_equal(
            a_segments.lengths,
            np.array(json_dict['lengths'])
        )
        assert json_dict['samplerate'] == a_segments.samplerate
        assert json_dict['labels'] == a_segments.labels

    def test_from_json(self, a_segments, tmp_path):
        json_path = tmp_path / 'a_segments.json'
        a_segments.to_json(json_path)
        a_segments_from_json = vocalpy.segments.Segments.from_json(
            json_path
        )
        # we check attributes are equal "manually", so as to not
        # assume that Segments.__eq__ works
        assert np.array_equal(a_segments.start_inds, a_segments_from_json.start_inds)
        assert np.array_equal(a_segments.lengths, a_segments_from_json.lengths)
        assert a_segments.samplerate == a_segments_from_json.samplerate
        assert a_segments.labels == a_segments_from_json.labels

    @pytest.mark.parametrize(
        'start_inds, lengths, sound, labels, expected_len',
        [
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND,
                [''] * 5,
                5
            ),
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND,
                None,
                5
            ),
            # empty segments
            (
                np.array([]).astype(int),
                np.array([]).astype(int),
                TEST_SOUND,
                None,
                0
            ),
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND,
                list('abcde'),
                5,
            ),
        ]
    )
    def test___len__(self, start_inds, lengths, sound, labels, expected_len):
        segments = vocalpy.Segments(
            start_inds, lengths, sound.samplerate, labels
        )
        assert len(segments) == expected_len

    @pytest.mark.parametrize(
        'segments, other, expected_eq',
        [
            (
                vocalpy.Segments(
                    np.array([0, 10, 20, 30, 40]),
                    np.array([10, 10, 10, 10, 10]),
                    TEST_SOUND.samplerate,
                    [''] * 5,
                ),
                vocalpy.Segments(
                    np.array([0, 10, 20, 30, 40]),
                    np.array([10, 10, 10, 10, 10]),
                    TEST_SOUND.samplerate,
                    [''] * 5,
                ),
                True,
            ),
            (
                vocalpy.Segments(
                    np.array([0, 10, 20, 30, 40]),
                    np.array([10, 10, 10, 10, 10]),
                    TEST_SOUND.samplerate,
                    [''] * 5,
                ),
                vocalpy.Segments(
                    np.array([0, 10, 20, 30, 40]),
                    np.array([10, 10, 10, 10, 10]),
                    TEST_SOUND.samplerate,
                    None,
                ),
                # because labels is None,
                # we will end up with the default empty string '' * len(start_inds)
                # so these Segments will be equal
                True,
            ),
            # empty segments
            (
                vocalpy.Segments(
                    np.array([]).astype(int),
                    np.array([]).astype(int),
                    TEST_SOUND.samplerate,
                    None,
                ),
                vocalpy.Segments(
                    np.array([]).astype(int),
                    np.array([]).astype(int),
                    TEST_SOUND.samplerate,
                    None,
                ),
                True,
            ),
            (
                vocalpy.Segments(
                    np.array([0, 10, 20, 30, 40]),
                    np.array([10, 10, 10, 10, 10]),
                    TEST_SOUND.samplerate,
                    list('abcde'),
                ),
                vocalpy.Segments(
                    np.array([0, 10, 20, 30, 40]),
                    np.array([10, 10, 10, 10, 10]),
                    TEST_SOUND.samplerate,
                    list('abcde'),
                ),
                True,
            ),
            (
                vocalpy.Segments(
                    np.array([0, 10, 20, 30, 40]),
                    np.array([10, 10, 10, 10, 10]),
                    TEST_SOUND.samplerate,
                    list('abcde'),
                ),
                vocalpy.Segments(
                    np.array([]).astype(int),
                    np.array([]).astype(int),
                    TEST_SOUND.samplerate,
                    None,
                ),
                False,
            ),
        ]
    )
    def test___eq__(self, segments, other, expected_eq):
        assert (
            (segments == other) is expected_eq
        )

    @pytest.mark.parametrize(
            'segments_csv_path, samplerate, columns_map, default_label',
            [
                (
                    "bfsongrepo_segments_csv_path",
                    vocalpy.Sound.read(AUDIO_LIST_CBIN[0]).samplerate,
                    None, 
                    None
                ),
                (
                    JOURJINE_ET_AL_2023_GO_SEGMENT_CSV_PATH,
                    vocalpy.Sound.read(JOURJINE_ET_AL_GO_WAV_PATH).samplerate,
                    {"start_seconds": "start_s", "stop_seconds": "stop_s"},
                    None,
                ),
                (
                    JOURJINE_ET_AL_2023_GO_SEGMENT_CSV_PATH,
                    vocalpy.Sound.read(JOURJINE_ET_AL_GO_WAV_PATH).samplerate,
                    {"start_seconds": "start_s", "stop_seconds": "stop_s"},
                    '-',
                ),
            ]
    )
    def test_from_csv(self, segments_csv_path, samplerate, columns_map, default_label, request):
        if isinstance(segments_csv_path, str):
            segments_csv_path = request.getfixturevalue(segments_csv_path)

        segments = vocalpy.Segments.from_csv(
            segments_csv_path, samplerate, columns_map=columns_map, default_label=default_label
        )
        assert isinstance(segments, vocalpy.Segments)
        
    @pytest.mark.parametrize(
            'segments_csv_path, samplerate, columns_map, default_label, expected_exception',
            [
                # `samplerate` is not an int
                (
                    JOURJINE_ET_AL_2023_GO_SEGMENT_CSV_PATH,
                    32000.0,
                    None,
                    None,
                    TypeError
                ),
                # `samplerate` is negative
                (
                    JOURJINE_ET_AL_2023_GO_SEGMENT_CSV_PATH,
                    -32000,
                    None,
                    None,
                    ValueError
                ),
                # `columns_map` not a dict, throws a TypeError
                (
                    JOURJINE_ET_AL_2023_GO_SEGMENT_CSV_PATH,
                    vocalpy.Sound.read(JOURJINE_ET_AL_GO_WAV_PATH).samplerate,
                    [],
                    None,
                    TypeError
                ),
                # not all key-value pairs in `columns_map` are string to string
                (
                    JOURJINE_ET_AL_2023_GO_SEGMENT_CSV_PATH,
                    vocalpy.Sound.read(JOURJINE_ET_AL_GO_WAV_PATH).samplerate,
                    {"start_seconds": "onset_s", "stop_seconds": 0},
                    None,
                    ValueError
                ),
                # invalid values for `columns_map`, not in `Segments.COLUMNS_MAP_VALID_VALUES`
                (
                    JOURJINE_ET_AL_2023_GO_SEGMENT_CSV_PATH,
                    vocalpy.Sound.read(JOURJINE_ET_AL_GO_WAV_PATH).samplerate,
                    {"start_seconds": "start_sec", "stop_seconds": "stop_sec"},
                    None,
                    ValueError
                ),
                # `default_label` is not a string
                (
                    JOURJINE_ET_AL_2023_GO_SEGMENT_CSV_PATH,
                    -32000,
                    None,
                    1,
                    ValueError
                ),
            ]
    )
    def test_from_csv_raises(self, segments_csv_path, samplerate, columns_map, default_label, expected_exception):
        with pytest.raises(expected_exception):
            vocalpy.Segments.from_csv(
            segments_csv_path, samplerate, columns_map=columns_map, default_label=default_label
        )
