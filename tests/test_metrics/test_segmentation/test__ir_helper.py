from collections import defaultdict

import numpy as np
import pytest

import vocalpy.metrics.segmentation._ir_helper


@pytest.mark.parametrize(
    'hypothesis, reference, tolerance, expected_hit_ref, expected_hit_hyp',
    [
        # ---- int values -----
        # all hits
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10, 15]),
                0,
                [0, 1, 2, 3],
                [0, 1, 2, 3],
        ),
        # no hits
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                0,
                [],
                [],
        ),
        # no > hits > all
        (
                np.array([1, 6, 10, 16]),
                np.array([0, 5, 10, 15]),
                0,
                [2],
                [2],
        ),
        (
                np.array([0, 5, 10]),
                np.array([0, 5, 10, 15]),
                0,
                [0, 1, 2],
                [0, 1, 2],
        ),
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10]),
                0,
                [0, 1, 2],
                [0, 1, 2],
        ),
        # ---- int values -----
        # ---- tolerance of 1
        # all hits
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                1,
                [0, 1, 2, 3],
                [0, 1, 2, 3],
        ),
        # no hits, tolerance of one
        (
                np.array([2, 7, 12, 17]),
                np.array([0, 5, 10, 15]),
                1,
                [],
                [],
        ),
        # no > hits > all, tolerance of one
        (
                np.array([2, 7, 11, 17]),
                np.array([0, 5, 10, 15]),
                1,
                [2],
                [2],
        ),
        (
                np.array([1, 6, 11]),
                np.array([0, 5, 10, 15]),
                1,
                [0, 1, 2],
                [0, 1, 2],
        ),
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10]),
                1,
                [0, 1, 2],
                [0, 1, 2],
        ),
        (
                np.array([0, 1, 6, 11]),
                np.array([0, 5, 10]),
                1,
                [0, 0, 1, 2],
                [0, 1, 2, 3],
        ),
        (
                np.array([0, 1, 2, 5, 6, 7, 10, 11, 13]),
                np.array([0, 5, 10]),
                1,
                [0, 0, 1, 1, 2, 2],
                [0, 1, 3, 4, 6, 7],
        ),
        # ---- float values -----
        # float values, tolerance=0, all hits
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0,
                [0, 1, 2, 3],
                [0, 1, 2, 3],
        ),
        # float values, tolerance=0, no hits
        (
                np.array([1.000, 6.000, 11.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0,
                [],
                [],
        ),
        # float values, tolerance=0, none < hits < all
        (
                np.array([1.000, 6.000, 10.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0,
                [2],
                [2],
        ),
        (
                np.array([0.000, 5.000, 10.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0,
                [0, 1, 2],
                [0, 1, 2],
        ),
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000]),
                0,
                [0, 1, 2],
                [0, 1, 2],
        ),
        # float values, all hits, tolerance of 0.5
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                [0, 1, 2, 3],
                [0, 1, 2, 3],
        ),
        # float values, no hits, tolerance of 0.5
        (
                np.array([1.500, 6.500, 11.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                [],
                [],
        ),
        # float values, none < hits < all, tolerance of 0.5
        (
                np.array([1.500, 6.500, 10.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                [2],
                [2],
        ),
        (
                np.array([0.500, 5.500, 10.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                [0, 1, 2],
                [0, 1, 2],
        ),
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000]),
                0.5,
                [0, 1, 2],
                [0, 1, 2],
        ),
        (
                np.array([0.500, 1.500, 5.500, 10.500]),
                np.array([0.000, 5.000, 10.000]),
                0.5,
                [0, 1, 2],
                [0, 2, 3]
        ),
        (
                np.array([0.250, 0.500, 2.500, 5.000, 5.500, 7.500, 10.500, 11.500, 13.500]),
                np.array([0.000, 5.000, 10.000]),
                0.5,
                [0, 0, 1, 1, 2],
                [0, 1, 3, 4, 6],
        ),
    ]
)
def test_fast_hit_windows(hypothesis, reference, tolerance, expected_hit_ref, expected_hit_hyp):
    hit_ref, hit_hyp = vocalpy.metrics.segmentation._ir_helper.fast_hit_windows(reference, hypothesis, tolerance)
    assert hit_ref == expected_hit_ref
    assert hit_hyp == expected_hit_hyp



@pytest.mark.parametrize(
    'hit_ref, hit_hyp, expected_matches',
    [
        # ---- int values -----
        # all hits
        (
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                {0: 0, 1: 1, 2: 2, 3: 3}
        ),
        # no hits
        (
                [],
                [],
                {},
        ),
        # no > hits > all
        (
                [2],
                [2],
                {2: 2},
        ),
        (
                [0, 1, 2],
                [0, 1, 2],
                {0:0, 1:1, 2:2}
        ),
        (
                [0, 0, 1, 2],
                [0, 1, 2, 3],
                {0: 0, 1: 2, 2: 3}
        ),
        (
                [0, 0, 1, 1, 2, 2],
                [0, 1, 3, 4, 6, 7],
                {0: 0, 1: 3, 2: 6}
        ),
    ]
)
def test_bipartite_match(hit_ref, hit_hyp, expected_matches):
    G = defaultdict(list)
    for ref_i, hypothesis_i in zip(hit_ref, hit_hyp):
        G[hypothesis_i].append(ref_i)
    matches = vocalpy.metrics.segmentation._ir_helper.bipartite_match(G)
    assert matches == expected_matches


@pytest.mark.parametrize(
    'hypothesis, reference, tolerance, decimals, expected_matches',
    [
        # ---- int values -----
        # ---- default tolerance and decimals ----
        # all hits
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                [(0, 0), (1, 1), (2, 2), (3, 3)]
        ),
        # no hits
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                [],
        ),
        # no > hits > all
        (
                np.array([1, 6, 10, 16]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                [(2, 2)],
        ),
        (
                np.array([0, 5, 10]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                [(0, 0), (1, 1), (2, 2)],
        ),
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10]),
                'default',
                'default',
                [(0, 0), (1, 1), (2, 2)],
        ),
        # ---- int values -----
        # ---- non-default tolerance ----
        # all hits, tolerance of one
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                [(0, 0), (1, 1), (2, 2), (3, 3)]
        ),
        # no hits, tolerance of one
        (
                np.array([2, 7, 12, 17]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                []
        ),
        # no > hits > all, tolerance of one
        (
                np.array([2, 7, 11, 17]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                [(2, 2)],
        ),
        (
                np.array([1, 6, 11]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                [(0, 0), (1, 1), (2, 2)],
        ),
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10]),
                1,
                'default',
                [(0, 0), (1, 1), (2, 2)],
        ),
        (
                np.array([0, 1, 6, 11]),
                np.array([0, 5, 10]),
                1,
                'default',
                [(0, 0), (1, 2), (2, 3)],
        ),
        (
                np.array([0, 1, 2, 5, 6, 7, 10, 11, 13]),
                np.array([0, 5, 10]),
                1,
                'default',
                [(0, 0), (1, 3), (2, 6)],
        ),
        # ---- float values -----
        # float values, all hits, default tolerance and decimals
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                [(0, 0), (1, 1), (2, 2), (3, 3)]
        ),
        # float values, no hits, default tolerance and decimals
        (
                np.array([1.000, 6.000, 11.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                [],
        ),
        # float values, none < hits < all, default tolerance and decimals
        (
                np.array([1.000, 6.000, 10.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                [(2, 2)],
        ),
        (
                np.array([0.000, 5.000, 10.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                [(0, 0), (1, 1), (2, 2)],
        ),
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000]),
                'default',
                'default',
                [(0, 0), (1, 1), (2, 2)],
        ),
        # float values, all hits, tolerance of 0.5, default decimals
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                [(0, 0), (1, 1), (2, 2), (3, 3)]
        ),
        # float values, no hits, tolerance of 0.5, default decimals
        (
                np.array([1.500, 6.500, 11.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                [],
        ),
        # float values, none < hits < all, tolerance of 0.5, default decimals
        (
                np.array([1.500, 6.500, 10.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                [(2, 2)],
        ),
        (
                np.array([0.500, 5.500, 10.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                [(0, 0), (1, 1), (2, 2)],
        ),
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000]),
                0.5,
                'default',
                [(0, 0), (1, 1), (2, 2)],
        ),
        (
                np.array([0.500, 1.500, 5.500, 10.500]),
                np.array([0.000, 5.000, 10.000]),
                0.5,
                'default',
                [(0, 0), (1, 2), (2, 3)],
        ),
        (
                np.array([0.500, 1.500, 2.500, 5.500, 6.500, 7.500, 10.500, 11.500, 13.500]),
                np.array([0.000, 5.000, 10.000]),
                0.5,
                'default',
                [(0, 0), (1, 3), (2, 6)],
        ),
        # float values, all hits, default tolerance, decimals=3 (default)
        (
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                [(0, 0), (1, 1), (2, 2), (3, 3)]
        ),
        # float values, no hits, default tolerance, decimals=3 (default)
        (
                np.array([1.0001, 6.0001, 11.0001, 16.0001]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                [],
        ),
        # float values, none < hits < all, default tolerance, decimals=3 (default)
        (
                np.array([1.0001, 6.0001, 10.0004, 16.0001]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                [(2, 2)],
        ),
        (
                np.array([0.0004, 5.0004, 10.0004]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                [(0, 0), (1, 1), (2, 2)],
        ),
        (
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                np.array([0.0001, 5.0001, 10.0001]),
                'default',
                3,
                [(0, 0), (1, 1), (2, 2)],
        ),
        # float values, all hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([0.5001, 5.5001, 10.5001, 15.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                [(0, 0), (1, 1), (2, 2), (3, 3)]
        ),
        # float values, no hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5001, 6.5001, 11.5001, 16.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                [],
        ),
        # float values, none < hits < all, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5001, 6.5001, 10.5001, 16.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                [(2, 2)],
        ),
        (
                np.array([0.5001, 5.5001, 10.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                [(0, 0), (1, 1), (2, 2)],
        ),
        (
                np.array([0.5001, 5.5001, 10.5001, 15.5001]),
                np.array([0.0004, 5.0004, 10.0004]),
                0.5,
                3,
                [(0, 0), (1, 1), (2, 2)],
        ),
        (
                np.array([0.5004, 1.5004, 5.5004, 10.5004]),
                np.array([0.0001, 5.0001, 10.0001]),
                0.5,
                3,
                [(0, 0), (1, 2), (2, 3)],
        ),
        (
                np.array([0.5004, 1.5004, 2.5004, 5.5004, 6.5004, 7.5004, 10.5004, 11.5004, 13.5004]),
                np.array([0.0001, 5.0001, 10.0001]),
                0.5,
                'default',
                [(0, 0), (1, 3), (2, 6)],
        ),
    ]
)
def test_match_event_times(hypothesis, reference, tolerance, decimals, expected_matches):
    if tolerance == 'default' and decimals == 'default':
        matches = vocalpy.metrics.segmentation.ir.match_event_times(
            hypothesis, reference
        )
    elif tolerance != 'default' and decimals == 'default':
        matches = vocalpy.metrics.segmentation.ir.match_event_times(
            hypothesis, reference, tolerance=tolerance,
        )
    elif tolerance == 'default' and decimals != 'default':
        matches = vocalpy.metrics.segmentation.ir.match_event_times(
            hypothesis, reference, decimals=decimals
        )
    elif tolerance != 'default' and decimals != 'default':
        matches = vocalpy.metrics.segmentation.ir.match_event_times(
            hypothesis, reference, tolerance=tolerance, decimals=decimals
        )
    assert matches == expected_matches


@pytest.mark.parametrize(
    'hypothesis, reference, tolerance, decimals, expected_error',
    [
        # int values with decimals should raise ValueError
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                'default',
                3,
                ValueError,
        ),
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                1,
                3,
                ValueError,
        ),
    ]
)
def test_match_event_times_raises(hypothesis, reference, tolerance, decimals, expected_error):
    if tolerance == 'default' and decimals == 'default':
        with pytest.raises(expected_error):
            vocalpy.metrics.segmentation.ir.match_event_times(
                hypothesis, reference
            )
    elif tolerance != 'default' and decimals == 'default':
        with pytest.raises(expected_error):
            vocalpy.metrics.segmentation.ir.match_event_times(
                hypothesis, reference, tolerance=tolerance,
            )
    elif tolerance == 'default' and decimals != 'default':
        with pytest.raises(expected_error):
            vocalpy.metrics.segmentation.ir.match_event_times(
                hypothesis, reference, decimals=decimals
            )
    elif tolerance != 'default' and decimals != 'default':
        with pytest.raises(expected_error):
            vocalpy.metrics.segmentation.ir.match_event_times(
                hypothesis, reference, tolerance=tolerance, decimals=decimals
            )
