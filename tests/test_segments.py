import numpy as np
import pandas as pd
import pytest
import xarray as xr

import vocalpy.segments

N_AUDIO_FILES_TO_USE = 5
from .fixtures.audio import AUDIO_LIST_CBIN
from .fixtures.audio import JOURJINE_ETAL_2023_WAV_LIST
AUDIO_FILES = AUDIO_LIST_CBIN[:N_AUDIO_FILES_TO_USE] + JOURJINE_ETAL_2023_WAV_LIST[:N_AUDIO_FILES_TO_USE]


TEST_SOUND = vocalpy.Sound(
    data=np.random.rand(1, 32000),
    samplerate=32000
)

TEST_SOUND_DATA = xr.DataArray(TEST_SOUND.data.squeeze(0))


class TestSegment:

    @pytest.mark.parametrize(
        'start_ind, length, data, label',
        [
            (0, 100, TEST_SOUND_DATA[:100], 'x'),
            (1, 100, TEST_SOUND_DATA[:100], 'x'),
            (0, 100, TEST_SOUND_DATA[:100], None),
            (1, 100, TEST_SOUND_DATA[:100], None),
        ]
    )
    def test_init(self, start_ind, length, data, label):
        if label is not None:
            segment = vocalpy.segments.Segment(
                start_ind,
                length,
                data,
                label
            )
        else:
            segment = vocalpy.segments.Segment(
                start_ind,
                length,
                data,
            )
        assert isinstance(segment, vocalpy.segments.Segment)
        for attr_name, attr_val in zip(
                ['start_ind', 'length', 'data', 'label'],
                [start_ind, length, data, label],
        ):
            if attr_name == 'label' and attr_val is None:
                # we're testing default label is empty string
                assert getattr(segment, attr_name) == ''
            elif attr_name == 'data':
                # for DataArray, need to use method `.equals`
                assert getattr(segment, attr_name).equals(attr_val)
            else:
                assert getattr(segment, attr_name) == attr_val

    @pytest.mark.parametrize(
        'start_ind, length, data, label, expected_exception',
        [
            # start_ind is not an int
            (1.0, 100, TEST_SOUND_DATA[:100], 'x', TypeError),
            # start_ind is not non-negative
            (-1, 100, TEST_SOUND_DATA[:100], 'x', ValueError),
            # length is not an int
            (0, 100.0, TEST_SOUND_DATA[:100], 'x', TypeError),
            # length is not positive
            (0, 0, TEST_SOUND_DATA[:100], 'x', ValueError),
            # length is not positive
            (0, -100, TEST_SOUND_DATA[:100], 'x', ValueError),
            # data is not an xarray.DataArray
            (0, 100, np.random.rand(100), 'x', TypeError),
            # data.ndim != 1
            (0, -100, TEST_SOUND_DATA[:100].expand_dims({'channel': 0}), 'x', ValueError),
            # data.size != length
            (0, 100, TEST_SOUND_DATA[:500], 'x', ValueError),
        ],
    )
    def test_init_raises(self, start_ind, length, data, label, expected_exception):
        with pytest.raises(expected_exception):
            vocalpy.segments.Segment(
                start_ind,
                length,
                data,
                label
            )

    @pytest.mark.parametrize(
        'start_ind, length, data, label',
        [
            (0, 100, TEST_SOUND_DATA[:100], 'x'),
            (1, 100, TEST_SOUND_DATA[:100], 'x'),
        ]
    )
    def test_write(self, start_ind, length, data, label, tmp_path):
        segment = vocalpy.segments.Segment(
            start_ind,
            length,
            data,
            label
        )
        segment_path = tmp_path / 'segment.h5'
        segment.write(segment_path)
        assert segment_path.exists()
        dataarr = xr.open_dataarray(segment_path)
        for (attr_name, attr_val) in zip(
            ['start_ind', 'length', 'label'],
            [start_ind, length, label]
        ):
            assert attr_name in dataarr.attrs
            assert dataarr.attrs[attr_name] == attr_val
        assert np.array_equal(
            dataarr.values,
            segment.data.values
        )

    @pytest.mark.parametrize(
        'start_ind, length, data, label',
        [
            (0, 100, TEST_SOUND_DATA[:100], 'x'),
            (1, 100, TEST_SOUND_DATA[:100], 'x'),
        ]
    )
    def test_read(self, start_ind, length, data, label, tmp_path):
        segment = vocalpy.segments.Segment(
            start_ind,
            length,
            data,
            label
        )
        segment_path = tmp_path / 'segment.h5'
        segment.write(segment_path)
        segment_loaded = vocalpy.Segment.read(segment_path)
        assert segment_loaded == segment


    @pytest.mark.parametrize(
        'segment, other_segment, expected_result',
        [
            (
                vocalpy.Segment(0, 100, TEST_SOUND_DATA[:100]),
                vocalpy.Segment(0, 100, TEST_SOUND_DATA[:100]),
                True,
            ),
            (
                vocalpy.Segment(0, 100, TEST_SOUND_DATA[:100]),
                vocalpy.Segment(1, 100, TEST_SOUND_DATA[:100]),
                False,
            ),
            (
                vocalpy.Segment(0, 100, TEST_SOUND_DATA[:100], 'x'),
                vocalpy.Segment(0, 100, TEST_SOUND_DATA[:100], 'x'),
                True,
            ),
            (
                vocalpy.Segment(0, 100, TEST_SOUND_DATA[:100], 'x'),
                vocalpy.Segment(0, 100, TEST_SOUND_DATA[:100], 'y'),
                False,
            ),
        ]
    )
    def test___eq__(self, segment, other_segment, expected_result):
        result = segment == other_segment
        assert result is expected_result


# we use these to parametrize methods on TestSegments class
SEGMENTS_ARGNAMES = 'start_inds, lengths, sound, labels'
SEGMENTS_ARGVALS = [
    (
        np.array([0, 10, 20, 30, 40]),
        np.array([10, 10, 10, 10, 10]),
        TEST_SOUND,
        np.array([''] * 5),
    ),
    (
        np.array([0, 10, 20, 30, 40]),
        np.array([10, 10, 10, 10, 10]),
        TEST_SOUND,
        None,
    ),
    # empty segments
    (
        np.array([]).astype(int),
        np.array([]).astype(int),
        TEST_SOUND,
        None,
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
    def test_init(self, start_inds, lengths, sound, labels):
        if labels is not None:
            segments = vocalpy.segments.Segments(
                start_inds,
                lengths,
                sound,
                labels,
            )
        else:
            # test we get default labels, empty strings
            segments = vocalpy.segments.Segments(
                start_inds,
                lengths,
                sound,
            )
        assert isinstance(segments, vocalpy.segments.Segments)
        for attr_name, attr_val in zip(
            ['start_inds', 'lengths', 'sound', 'labels'],
            [start_inds, lengths, sound, labels],
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
        'start_inds, lengths, sound, labels, expected_exception',
        # test cases should be in order of pre-conditions in __init__
        # so we can hope to make sense of this giant list
        [
            # sound is not a Sound
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([10, 10, 10, 10, 10]),
                np.random.rand(1, 100),
                np.array([''] * 5),
                TypeError
            ),
            # start inds not numpy array
            (
                [0, 10, 20, 30, 40],
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND,
                np.array([''] * 5),
                TypeError
            ),
            # lengths not numpy array
            (
                np.array([0, 10, 20, 30, 40]),
                [10, 10, 10, 10, 10],
                TEST_SOUND,
                np.array([''] * 5),
                TypeError
            ),
            # start inds not dtype int
            (
                np.array([0, 10, 20, 30, 40]).astype(np.float32),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND,
                np.array([''] * 5),
                ValueError
            ),
            # lengths not dtype int
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([10, 10, 10, 10, 10]).astype(np.float32),
                TEST_SOUND,
                np.array([''] * 5),
                ValueError
            ),
            # start inds not 1-d
            (
                np.array([[0, 10, 20, 30, 40]]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND,
                np.array([''] * 5),
                ValueError
            ),
            # start inds not 1-d
            (
                np.array(0),
                np.array([10]),
                TEST_SOUND,
                np.array([''] * 5),
                ValueError
            ),
            # lengths not 1-d
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([[10, 10, 10, 10, 10]]),
                TEST_SOUND,
                np.array([''] * 5),
                ValueError
            ),
            # lengths not 1-d
            (
                np.array([0]),
                np.array(10),
                TEST_SOUND,
                np.array([''] * 5),
                ValueError
            ),
            # start_inds.size != lengths.size
            (
                np.array([0, 10]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND,
                np.array([''] * 5),
                ValueError
            ),
            # start inds not all non-negative
            (
                np.array([-10, 10, 20, 30, 40]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND,
                np.array([''] * 5),
                ValueError
            ),
            # start inds not monotonically increasing
            (
                np.array([40, 30, 20, 10, 0]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND,
                np.array([''] * 5),
                ValueError
            ),
            # lengths not all positive
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([0, 10, 10, 10, 10]),
                TEST_SOUND,
                np.array([''] * 5),
                ValueError
            ),
            # lengths not all positive
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([-1, 10, 10, 10, 10]),
                TEST_SOUND,
                np.array([''] * 5),
                ValueError
            ),
            # total length of start_ind[-1] + lengths[-1] > length of sound.data
            (
                np.array([0, 10, 20, 30, TEST_SOUND.data.shape[1] - 10]),
                np.array([10, 10, 10, 10, TEST_SOUND.data.shape[1] + 11]),
                TEST_SOUND,
                [5] * 5,
                ValueError
            ),
            # labels not ndarray
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND,
                [''] * 5,
                TypeError
            ),
            # labels not str
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND,
                np.array([5] * 5),
                ValueError
            ),
            # labels not 1-d
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND,
                np.array([5] * 5)[np.newaxis, :],
                ValueError
            ),
            # labels.size != start_inds.size
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([10, 10, 10, 10, 10]),
                TEST_SOUND,
                np.array([''] * 2),
                ValueError
            ),
        ],
    )
    def test_init_raises(self, start_inds, lengths, sound, labels, expected_exception):
        with pytest.raises(expected_exception):
            vocalpy.segments.Segments(
                start_inds,
                lengths,
                sound,
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
            a_segments.start_inds / a_segments.sound.samplerate
        )

    def test_durations_property(self, a_segments):
        assert np.array_equal(
            a_segments.durations,
            a_segments.lengths / a_segments.sound.samplerate
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

    def test_to_df(self, a_segments):
        df = a_segments.to_df()

        assert isinstance(df, pd.DataFrame)
        assert np.array_equal(
            a_segments.start_inds,
            df['start_ind'].values
        )
        assert np.array_equal(
            a_segments.lengths,
            df['length'].values
        )
        assert np.array_equal(
            a_segments.labels,
            df['label'].values
        )

    def test_to_csv(self, a_segments, tmp_path):
        csv_path = tmp_path / 'a_segments.csv'

        a_segments.to_csv(csv_path)

        assert csv_path.exists
        df = pd.read_csv(csv_path)
        assert np.array_equal(
            a_segments.start_inds,
            df['start_ind'].values
        )
        assert np.array_equal(
            a_segments.lengths,
            df['length'].values
        )
        assert np.array_equal(
            a_segments.labels,
            df['label'].values
        )

    def test_from_csv(self, a_segments, tmp_path):
        csv_path = tmp_path / 'a_segments.csv'
        a_segments.to_csv(csv_path)
        sound_path = tmp_path / 'sound.wav'
        TEST_SOUND.write(sound_path)
        a_segments_from_csv = vocalpy.segments.Segments.from_csv(csv_path, sound_path=sound_path)
        # TODO: implement __eq__ for Segments
        assert a_segments == a_segments_from_csv

    def test___len__(self, a_segments):
        # TODO: test empty Segment has len 0
        assert False

    def test___iter__(self, a_segments):
        segment_list = list(iter(a_segments))
        assert all(
            [isinstance(segment, vocalpy.segments.Segment)
             for segment in segment_list]
        )

    @pytest.mark.parametrize(
        'key',
        [
            1,
            slice(0, 3)
        ]
    )
    def test___getitem__(self, key, a_segments):
        assert False

    def test_to_json(self, a_segments, tmp_path):
        json_path = tmp_path / 'a_segments.json'
        a_segments.to_json(json_path)
        assert json_path.exists
        with json_path.open('r') as fp:
            segments_json = json.load(fp)
        # TODO: assert df.values == a_segments.attributes
        assert False

    def test_from_json(self, a_segments, tmp_path):
        json_path = tmp_path / 'a_segments.json'
        a_segments.to_json(json_path)
        sound_path = tmp_path / 'sound.wav'
        TEST_SOUND.write(sound_path)
        a_segments_from_csv = vocalpy.segments.Segments.from_json(
            json_path, sound_path=sound_path
        )
        # TODO: implement __eq__ for Segments
        assert a_segments == a_segments_from_csv
        assert False