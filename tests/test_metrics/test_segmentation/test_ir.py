import math

import numpy as np
import pytest

import vocalpy.metrics.segmentation.ir


@pytest.mark.parametrize(
    'hypothesis, reference, tolerance, decimals, expected_n_tp, expected_hits',
    [
        # ---- int values -----
        # ---- default tolerance and decimals ----
        # all hits
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                4,
                np.array([0, 1, 2, 3]),
        ),
        # no hits
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                0,
                np.array([]),
        ),
        # no > hits > all
        (
                np.array([1, 6, 10, 16]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                1,
                np.array([2]),
        ),
        (
                np.array([0, 5, 10]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10]),
                'default',
                'default',
                3,
                np.array([0, 1, 2]),
        ),
        # ---- int values -----
        # ---- non-default tolerance, default decimals ----
        # all hits, tolerance of one
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                4,
                np.array([0, 1, 2, 3]),
        ),
        # no hits, tolerance of one
        (
                np.array([2, 7, 12, 17]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                0,
                np.array([]),
        ),
        # no > hits > all, tolerance of one
        (
                np.array([2, 7, 11, 17]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                1,
                np.array([2]),
        ),
        (
                np.array([1, 6, 11]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10]),
                1,
                'default',
                3,
                np.array([0, 1, 2]),
        ),
        # ---- float values -----
        # float values, all hits, default tolerance and decimals
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, default tolerance and decimals
        (
                np.array([1.000, 6.000, 11.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                0,
                np.array([]),
        ),
        # float values, none < hits < all, default tolerance and decimals
        (
                np.array([1.000, 6.000, 10.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                1,
                np.array([2]),
        ),
        (
                np.array([0.000, 5.000, 10.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000]),
                'default',
                'default',
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, tolerance of 0.5, default decimals
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, tolerance of 0.5, default decimals
        (
                np.array([1.500, 6.500, 11.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                0,
                np.array([]),
        ),
        # float values, none < hits < all, tolerance of 0.5, default decimals
        (
                np.array([1.500, 6.500, 10.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                1,
                np.array([2]),
        ),
        (
                np.array([0.500, 5.500, 10.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000]),
                0.5,
                'default',
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, default tolerance, decimals=3 (default)
        (
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, default tolerance, decimals=3 (default)
        (
                np.array([1.0001, 6.0001, 11.0001, 16.0001]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, default tolerance, decimals=3 (default)
        (
                np.array([1.0001, 6.0001, 10.0004, 16.0001]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                1,
                np.array([2]),
        ),
        (
                np.array([0.0004, 5.0004, 10.0004]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                np.array([0.0001, 5.0001, 10.0001]),
                'default',
                3,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([0.5001, 5.5001, 10.5001, 15.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5001, 6.5001, 11.5001, 16.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5001, 6.5001, 10.5001, 16.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                1,
                np.array([2]),
        ),
        (
                np.array([0.5001, 5.5001, 10.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.5001, 5.5001, 10.5001, 15.5001]),
                np.array([0.0004, 5.0004, 10.0004]),
                0.5,
                3,
                3,
                np.array([0, 1, 2]),
        ),
    ]
)
def test_compute_true_positives(hypothesis, reference, tolerance, decimals, expected_n_tp, expected_hits):
    if tolerance == 'default' and decimals == 'default':
        n_tp, hits = vocalpy.metrics.segmentation.ir.compute_true_positives(
            hypothesis, reference
        )
    elif tolerance != 'default' and decimals == 'default':
        n_tp, hits = vocalpy.metrics.segmentation.ir.compute_true_positives(
            hypothesis, reference, tolerance=tolerance,
        )
    elif tolerance == 'default' and decimals != 'default':
        n_tp, hits = vocalpy.metrics.segmentation.ir.compute_true_positives(
            hypothesis, reference, decimals=decimals
        )
    elif tolerance != 'default' and decimals != 'default':
        n_tp, hits = vocalpy.metrics.segmentation.ir.compute_true_positives(
            hypothesis, reference, tolerance=tolerance, decimals=decimals
        )

    assert n_tp == expected_n_tp
    assert np.array_equal(hits, expected_hits)

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
def test_compute_true_positives_raises(hypothesis, reference, tolerance, decimals, expected_error):
    if tolerance == 'default' and decimals == 'default':
        with pytest.raises(expected_error):
            vocalpy.metrics.segmentation.ir.compute_true_positives(
                hypothesis, reference
            )
    elif tolerance != 'default' and decimals == 'default':
        with pytest.raises(expected_error):
            vocalpy.metrics.segmentation.ir.compute_true_positives(
                hypothesis, reference, tolerance=tolerance,
            )
    elif tolerance == 'default' and decimals != 'default':
        with pytest.raises(expected_error):
            vocalpy.metrics.segmentation.ir.compute_true_positives(
                hypothesis, reference, decimals=decimals
            )
    elif tolerance != 'default' and decimals != 'default':
        with pytest.raises(expected_error):
            vocalpy.metrics.segmentation.ir.compute_true_positives(
                hypothesis, reference, tolerance=tolerance, decimals=decimals
            )


@pytest.mark.parametrize(
    'hypothesis, reference, metric, tolerance, decimals, expected_metric_value, expected_n_tp, expected_hits',
    [
        # ---- int values -----
        # int values, all hits, default tolerance and decimals
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10, 15]),
                'precision',
                'default',
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # int values, all misses, default tolerance and decimals
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                'precision',
                'default',
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # int values, no > hits > all, default tolerance and decimals
        (
                np.array([0, 5, 10, 15]),
                np.array([1, 6, 10, 16]),
                'precision',
                'default',
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10]),
                'precision',
                'default',
                'default',
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0, 5, 10]),
                np.array([0, 5, 10, 15]),
                'precision',
                'default',
                'default',
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        # int values, all hits, tolerance of one, default decimals
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                'precision',
                1,
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        (
                np.array([0., 5., 10., 15.]),
                np.array([0., 5., 10., 15.]),
                'precision',
                'default',
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        (
                np.array([0., 5., 10., 15.]),
                np.array([0.5, 5.5, 10.5, 15.5]),
                'precision',
                0.5,
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # ---- recall ----
        # ---- int values -----
        # ---- default tolerance and decimals ----
        # all hits
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10, 15]),
                'recall',
                'default',
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # no hits
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                'recall',
                'default',
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # no > hits > all
        (
                np.array([1, 6, 10, 16]),
                np.array([0, 5, 10, 15]),
                'recall',
                'default',
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0, 5, 10]),
                np.array([0, 5, 10, 15]),
                'recall',
                'default',
                'default',
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10]),
                'recall',
                'default',
                'default',
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        # ---- int values -----
        # ---- non-default tolerance, default decimals ----
        # all hits, tolerance of one
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                'recall',
                1,
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # no hits, tolerance of one
        (
                np.array([2, 7, 12, 17]),
                np.array([0, 5, 10, 15]),
                'recall',
                1,
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # no > hits > all, tolerance of one
        (
                np.array([2, 7, 11, 17]),
                np.array([0, 5, 10, 15]),
                'recall',
                1,
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([1, 6, 11]),
                np.array([0, 5, 10, 15]),
                'recall',
                1,
                'default',
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10]),
                'recall',
                1,
                'default',
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        # ---- float values -----
        # float values, all hits, default tolerance and decimals
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'recall',
                'default',
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, default tolerance and decimals
        (
                np.array([1.000, 6.000, 11.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'recall',
                'default',
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, default tolerance and decimals
        (
                np.array([1.000, 6.000, 10.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'recall',
                'default',
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.000, 5.000, 10.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'recall',
                'default',
                'default',
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000]),
                'recall',
                'default',
                'default',
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, tolerance of 0.5, default decimals
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'recall',
                0.5,
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, tolerance of 0.5, default decimals
        (
                np.array([1.500, 6.500, 11.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'recall',
                0.5,
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, tolerance of 0.5, default decimals
        (
                np.array([1.500, 6.500, 10.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'recall',
                0.5,
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.500, 5.500, 10.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'recall',
                0.5,
                'default',
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000]),
                'recall',
                0.5,
                'default',
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, default tolerance, decimals=3 (default)
        (
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'recall',
                'default',
                3,
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, default tolerance, decimals=3 (default)
        (
                np.array([1.0001, 6.0001, 11.0001, 16.0001]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'recall',
                'default',
                3,
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, default tolerance, decimals=3 (default)
        (
                np.array([1.0001, 6.0001, 10.0004, 16.0001]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'recall',
                'default',
                3,
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.0004, 5.0004, 10.0004]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'recall',
                'default',
                3,
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                np.array([0.0001, 5.0001, 10.0001]),
                'recall',
                'default',
                3,
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([0.5001, 5.5001, 10.5001, 15.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                'recall',
                0.5,
                3,
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5001, 6.5001, 11.5001, 16.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                'recall',
                0.5,
                3,
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5001, 6.5001, 10.5001, 16.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                'recall',
                0.5,
                3,
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.5001, 5.5001, 10.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                'recall',
                0.5,
                3,
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.5001, 5.5001, 10.5001, 15.5001]),
                np.array([0.0004, 5.0004, 10.0004]),
                'recall',
                0.5,
                3,
                1.0,
                3,
                np.array([0, 1, 2]),
        ),

        # ---- int values -----
        # ---- default tolerance and decimals ----
        # all hits
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10, 15]),
                'fscore',
                'default',
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # no hits
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                'fscore',
                'default',
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # no > hits > all
        (
                np.array([1, 6, 10, 16]),
                np.array([0, 5, 10, 15]),
                'fscore',
                'default',
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0, 5, 10]),
                np.array([0, 5, 10, 15]),
                'fscore',
                'default',
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10]),
                'fscore',
                'default',
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        # ---- int values -----
        # ---- non-default tolerance, default decimals ----
        # all hits, tolerance of one
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                'fscore',
                1,
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # no hits, tolerance of one
        (
                np.array([2, 7, 12, 17]),
                np.array([0, 5, 10, 15]),
                'fscore',
                1,
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # no > hits > all, tolerance of one
        (
                np.array([2, 7, 11, 17]),
                np.array([0, 5, 10, 15]),
                'fscore',
                1,
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([1, 6, 11]),
                np.array([0, 5, 10, 15]),
                'fscore',
                1,
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10]),
                'fscore',
                1,
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        # ---- float values -----
        # float values, all hits, default tolerance and decimals
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'fscore',
                'default',
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, default tolerance and decimals
        (
                np.array([1.000, 6.000, 11.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'fscore',
                'default',
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, default tolerance and decimals
        (
                np.array([1.000, 6.000, 10.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'fscore',
                'default',
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.000, 5.000, 10.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'fscore',
                'default',
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000]),
                'fscore',
                'default',
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, tolerance of 0.5, default decimals
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'fscore',
                0.5,
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, tolerance of 0.5, default decimals
        (
                np.array([1.500, 6.500, 11.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'fscore',
                0.5,
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, tolerance of 0.5, default decimals
        (
                np.array([1.500, 6.500, 10.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'fscore',
                0.5,
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.500, 5.500, 10.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'fscore',
                0.5,
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000]),
                'fscore',
                0.5,
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, default tolerance, decimals=3 (default)
        (
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'fscore',
                'default',
                3,
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, default tolerance, decimals=3 (default)
        (
                np.array([1.0001, 6.0001, 11.0001, 16.0001]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'fscore',
                'default',
                3,
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, default tolerance, decimals=3 (default)
        (
                np.array([1.0001, 6.0001, 10.0004, 16.0001]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'fscore',
                'default',
                3,
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.0004, 5.0004, 10.0004]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'fscore',
                'default',
                3,
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                np.array([0.0001, 5.0001, 10.0001]),
                'fscore',
                'default',
                3,
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([0.5001, 5.5001, 10.5001, 15.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                'fscore',
                0.5,
                3,
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5001, 6.5001, 11.5001, 16.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                'fscore',
                0.5,
                3,
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5001, 6.5001, 10.5001, 16.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                'fscore',
                0.5,
                3,
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.5001, 5.5001, 10.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                'fscore',
                0.5,
                3,
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.5001, 5.5001, 10.5001, 15.5001]),
                np.array([0.0004, 5.0004, 10.0004]),
                'fscore',
                0.5,
                3,
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
    ]
)
def test_precision_recall_fscore(hypothesis, reference, metric, tolerance, decimals,
                                 expected_metric_value, expected_n_tp, expected_hits):
    if tolerance == 'default' and decimals == 'default':
        metric_value, n_tp, hits = vocalpy.metrics.segmentation.ir.precision_recall_fscore(
            hypothesis, reference, metric
        )
    elif tolerance != 'default' and decimals == 'default':
        metric_value, n_tp, hits = vocalpy.metrics.segmentation.ir.precision_recall_fscore(
            hypothesis, reference, metric, tolerance=tolerance,
        )
    elif tolerance == 'default' and decimals != 'default':
        metric_value, n_tp, hits = vocalpy.metrics.segmentation.ir.precision_recall_fscore(
            hypothesis, reference, metric, decimals=decimals
        )
    elif tolerance != 'default' and decimals != 'default':
        metric_value, n_tp, hits = vocalpy.metrics.segmentation.ir.precision_recall_fscore(
            hypothesis, reference, metric, tolerance=tolerance, decimals=decimals
        )

    assert math.isclose(metric_value, expected_metric_value)
    assert n_tp == expected_n_tp
    assert np.array_equal(hits, expected_hits)


@pytest.mark.parametrize(
    'hypothesis, reference, tolerance, decimals, expected_precision, expected_n_tp, expected_hits',
    [
        # ---- int values -----
        # ---- default tolerance and decimals ----
        # all hits
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # no hits
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # no > hits > all
        (
                np.array([1, 6, 10, 16]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0, 5, 10]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10]),
                'default',
                'default',
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        # ---- int values -----
        # ---- non-default tolerance, default decimals ----
        # all hits, tolerance of one
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # no hits, tolerance of one
        (
                np.array([2, 7, 12, 17]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # no > hits > all, tolerance of one
        (
                np.array([2, 7, 11, 17]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([1, 6, 11]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10]),
                1,
                'default',
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        # ---- float values -----
        # float values, all hits, default tolerance and decimals
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, default tolerance and decimals
        (
                np.array([1.000, 6.000, 11.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, default tolerance and decimals
        (
                np.array([1.000, 6.000, 10.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.000, 5.000, 10.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000]),
                'default',
                'default',
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, tolerance of 0.5, default decimals
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, tolerance of 0.5, default decimals
        (
                np.array([1.500, 6.500, 11.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, tolerance of 0.5, default decimals
        (
                np.array([1.500, 6.500, 10.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.500, 5.500, 10.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000]),
                0.5,
                'default',
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, default tolerance, decimals=3 (default)
        (
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, default tolerance, decimals=3 (default)
        (
                np.array([1.0001, 6.0001, 11.0001, 16.0001]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, default tolerance, decimals=3 (default)
        (
                np.array([1.0001, 6.0001, 10.0004, 16.0001]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.0004, 5.0004, 10.0004]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                np.array([0.0001, 5.0001, 10.0001]),
                'default',
                3,
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([0.5001, 5.5001, 10.5001, 15.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5001, 6.5001, 11.5001, 16.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5001, 6.5001, 10.5001, 16.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.5001, 5.5001, 10.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.5001, 5.5001, 10.5001, 15.5001]),
                np.array([0.0004, 5.0004, 10.0004]),
                0.5,
                3,
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
    ]
)
def test__precision(hypothesis, reference, tolerance, decimals, expected_precision, expected_n_tp, expected_hits):
    if tolerance == 'default' and decimals == 'default':
        precision, n_tp, hits = vocalpy.metrics.segmentation.ir._precision(
            hypothesis, reference
        )
    elif tolerance != 'default' and decimals == 'default':
        precision, n_tp, hits = vocalpy.metrics.segmentation.ir._precision(
            hypothesis, reference, tolerance=tolerance,
        )
    elif tolerance == 'default' and decimals != 'default':
        precision, n_tp, hits = vocalpy.metrics.segmentation.ir._precision(
            hypothesis, reference, decimals=decimals
        )
    elif tolerance != 'default' and decimals != 'default':
        precision, n_tp, hits = vocalpy.metrics.segmentation.ir._precision(
            hypothesis, reference, tolerance=tolerance, decimals=decimals
        )

    assert math.isclose(precision, expected_precision)
    assert n_tp == expected_n_tp
    assert np.array_equal(hits, expected_hits)


@pytest.mark.parametrize(
    'hypothesis, reference, tolerance, decimals, expected_recall, expected_n_tp, expected_hits',
    [
        # ---- int values -----
        # ---- default tolerance and decimals ----
        # all hits
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # no hits
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # no > hits > all
        (
                np.array([1, 6, 10, 16]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0, 5, 10]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10]),
                'default',
                'default',
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        # ---- int values -----
        # ---- non-default tolerance, default decimals ----
        # all hits, tolerance of one
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # no hits, tolerance of one
        (
                np.array([2, 7, 12, 17]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # no > hits > all, tolerance of one
        (
                np.array([2, 7, 11, 17]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([1, 6, 11]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10]),
                1,
                'default',
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        # ---- float values -----
        # float values, all hits, default tolerance and decimals
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, default tolerance and decimals
        (
                np.array([1.000, 6.000, 11.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, default tolerance and decimals
        (
                np.array([1.000, 6.000, 10.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.000, 5.000, 10.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000]),
                'default',
                'default',
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, tolerance of 0.5, default decimals
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, tolerance of 0.5, default decimals
        (
                np.array([1.500, 6.500, 11.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, tolerance of 0.5, default decimals
        (
                np.array([1.500, 6.500, 10.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.500, 5.500, 10.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000]),
                0.5,
                'default',
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, default tolerance, decimals=3 (default)
        (
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, default tolerance, decimals=3 (default)
        (
                np.array([1.0001, 6.0001, 11.0001, 16.0001]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, default tolerance, decimals=3 (default)
        (
                np.array([1.0001, 6.0001, 10.0004, 16.0001]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.0004, 5.0004, 10.0004]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                np.array([0.0001, 5.0001, 10.0001]),
                'default',
                3,
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([0.5001, 5.5001, 10.5001, 15.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5001, 6.5001, 11.5001, 16.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5001, 6.5001, 10.5001, 16.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.5001, 5.5001, 10.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                0.75,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.5001, 5.5001, 10.5001, 15.5001]),
                np.array([0.0004, 5.0004, 10.0004]),
                0.5,
                3,
                1.0,
                3,
                np.array([0, 1, 2]),
        ),
    ]
)
def test__recall(hypothesis, reference, tolerance, decimals, expected_recall, expected_n_tp, expected_hits):
    if tolerance == 'default' and decimals == 'default':
        recall, n_tp, hits = vocalpy.metrics.segmentation.ir._recall(
            hypothesis, reference
        )
    elif tolerance != 'default' and decimals == 'default':
        recall, n_tp, hits = vocalpy.metrics.segmentation.ir._recall(
            hypothesis, reference, tolerance=tolerance,
        )
    elif tolerance == 'default' and decimals != 'default':
        recall, n_tp, hits = vocalpy.metrics.segmentation.ir._recall(
            hypothesis, reference, decimals=decimals
        )
    elif tolerance != 'default' and decimals != 'default':
        recall, n_tp, hits = vocalpy.metrics.segmentation.ir._recall(
            hypothesis, reference, tolerance=tolerance, decimals=decimals
        )

    assert math.isclose(recall, expected_recall)
    assert n_tp == expected_n_tp
    assert np.array_equal(hits, expected_hits)


@pytest.mark.parametrize(
    'hypothesis, reference, tolerance, decimals, expected_fscore, expected_n_tp, expected_hits',
    [
        # ---- int values -----
        # ---- default tolerance and decimals ----
        # all hits
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # no hits
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # no > hits > all
        (
                np.array([1, 6, 10, 16]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0, 5, 10]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0, 5, 10, 15]),
                np.array([0, 5, 10]),
                'default',
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        # ---- int values -----
        # ---- non-default tolerance, default decimals ----
        # all hits, tolerance of one
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # no hits, tolerance of one
        (
                np.array([2, 7, 12, 17]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # no > hits > all, tolerance of one
        (
                np.array([2, 7, 11, 17]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([1, 6, 11]),
                np.array([0, 5, 10, 15]),
                1,
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10]),
                1,
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        # ---- float values -----
        # float values, all hits, default tolerance and decimals
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, default tolerance and decimals
        (
                np.array([1.000, 6.000, 11.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, default tolerance and decimals
        (
                np.array([1.000, 6.000, 10.000, 16.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.000, 5.000, 10.000]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                'default',
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.000, 5.000, 10.000]),
                'default',
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, tolerance of 0.5, default decimals
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, tolerance of 0.5, default decimals
        (
                np.array([1.500, 6.500, 11.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, tolerance of 0.5, default decimals
        (
                np.array([1.500, 6.500, 10.500, 16.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.500, 5.500, 10.500]),
                np.array([0.000, 5.000, 10.000, 15.000]),
                0.5,
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.500, 5.500, 10.500, 15.500]),
                np.array([0.000, 5.000, 10.000]),
                0.5,
                'default',
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, default tolerance, decimals=3 (default)
        (
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, default tolerance, decimals=3 (default)
        (
                np.array([1.0001, 6.0001, 11.0001, 16.0001]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, default tolerance, decimals=3 (default)
        (
                np.array([1.0001, 6.0001, 10.0004, 16.0001]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.0004, 5.0004, 10.0004]),
                np.array([0.0001, 5.0001, 10.0001, 15.0001]),
                'default',
                3,
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                np.array([0.0001, 5.0001, 10.0001]),
                'default',
                3,
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        # float values, all hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([0.5001, 5.5001, 10.5001, 15.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5001, 6.5001, 11.5001, 16.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5001, 6.5001, 10.5001, 16.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                0.25,
                1,
                np.array([2]),
        ),
        (
                np.array([0.5001, 5.5001, 10.5001]),
                np.array([0.0004, 5.0004, 10.0004, 15.0004]),
                0.5,
                3,
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0.5001, 5.5001, 10.5001, 15.5001]),
                np.array([0.0004, 5.0004, 10.0004]),
                0.5,
                3,
                0.8571428571428571,
                3,
                np.array([0, 1, 2]),
        ),
    ]
)
def test__fscore(hypothesis, reference, tolerance, decimals, expected_fscore, expected_n_tp, expected_hits):
    if tolerance == 'default' and decimals == 'default':
        fscore, n_tp, hits = vocalpy.metrics.segmentation.ir._fscore(
            hypothesis, reference
        )
    elif tolerance != 'default' and decimals == 'default':
        fscore, n_tp, hits = vocalpy.metrics.segmentation.ir._fscore(
            hypothesis, reference, tolerance=tolerance,
        )
    elif tolerance == 'default' and decimals != 'default':
        fscore, n_tp, hits = vocalpy.metrics.segmentation.ir._fscore(
            hypothesis, reference, decimals=decimals
        )
    elif tolerance != 'default' and decimals != 'default':
        fscore, n_tp, hits = vocalpy.metrics.segmentation.ir._fscore(
            hypothesis, reference, tolerance=tolerance, decimals=decimals
        )

    assert math.isclose(fscore, expected_fscore)
    assert n_tp == expected_n_tp
    assert np.array_equal(hits, expected_hits)


@pytest.mark.parametrize(
    'onsets_hyp, offset_hyp, onsets_ref, offset_ref, tolerance, decimals, method, expected_precision, expected_n_tp, expected_hits',
    [
        # ---- int values -----
        # ---- default tolerance and decimals ----
        # all hits
        (
                np.array([0, 8, 16, 24]),
                np.array([4, 12, 20, 28]),
                np.array([0, 8, 16, 24]),
                np.array([4, 12, 20, 28]),
                'default',
                'default',
                'combine',
                1.0,
                8,
                np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        ),
        # no hits
        (
                np.array([1, 9, 17, 25]),
                np.array([5, 13, 21, 29]),
                np.array([0, 8, 16, 24]),
                np.array([4, 12, 20, 28]),
                'default',
                'default',
                'combine',
                0.0,
                0,
                np.array([]),
        ),
        # no > hits > all
        (
                np.array([1, 9, 16, 25]),
                np.array([5, 13, 20, 29]),
                np.array([0, 8, 16, 24]),
                np.array([4, 12, 20, 28]),
                'default',
                'default',
                'combine',
                0.25,
                2,
                np.array([4, 5]),
        ),
        (
                np.array([0, 8, 16]),
                np.array([4, 12, 20]),
                np.array([0, 8, 16, 24]),
                np.array([4, 12, 20, 28]),
                'default',
                'default',
                'combine',
                1.0,
                6,
                np.array([0, 1, 2, 3, 4, 5]),
        ),
        (
                np.array([0, 8, 16, 24]),
                np.array([4, 12, 20, 28]),
                np.array([0, 8, 16]),
                np.array([4, 12, 20]),
                'default',
                'default',
                'combine',
                0.75,
                6,
                np.array([0, 1, 2, 3, 4, 5]),
        ),
        # ---- int values -----
        # ---- non-default tolerance, default decimals ----
        # all hits, tolerance of one
        (
                np.array([1, 6, 11, 16]),
                np.array([3, 8, 13, 18]),
                np.array([0, 5, 10, 15]),
                np.array([2, 7, 12, 17]),
                1,
                'default',
                'combine',
                1.0,
                8,
                np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        ),
        # no hits, tolerance of one
        (
                np.array([2, 10, 18, 26]),
                np.array([6, 14, 22, 30]),
                np.array([0, 8, 16, 24]),
                np.array([4, 12, 20, 28]),
                1,
                'default',
                'combine',
                0.0,
                0,
                np.array([]),
        ),
        # no > hits > all, tolerance of one
        (
                np.array([2, 10, 17, 26]),
                np.array([6, 14, 21, 30]),
                np.array([0, 8, 16, 24]),
                np.array([4, 12, 20, 28]),
                1,
                'default',
                'combine',
                0.25,
                2,
                np.array([4, 5]),
        ),
        (
                np.array([1, 9, 17]),
                np.array([5, 13, 21]),
                np.array([0, 8, 16, 24]),
                np.array([4, 12, 20, 28]),
                1,
                'default',
                'combine',
                1.0,
                6,
                np.array([0, 1, 2, 3, 4, 5]),
        ),
        (
                np.array([1, 9, 17, 25,]),
                np.array([5, 13, 21, 29]),
                np.array([0, 8, 16]),
                np.array([4, 12, 20]),
                1,
                'default',
                'combine',
                0.75,
                6,
                np.array([0, 1, 2, 3, 4, 5]),
        ),
        # ---- float values -----
        # float values, all hits, default tolerance and decimals
        (
                np.array([0.000, 8.000, 16.000, 24.000]),
                np.array([4.000, 12.000, 20.000, 28.000]),
                np.array([0.000, 8.000, 16.000, 24.000]),
                np.array([4.000, 12.000, 20.000, 28.000]),
                'default',
                'default',
                'combine',
                1.0,
                8,
                np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        ),
        # float values, no hits, default tolerance and decimals
        (
                np.array([1.000, 9.000, 17.000, 25.000]),
                np.array([5.000, 13.000, 21.000, 29.000]),
                np.array([0.000, 8.000, 16.000, 24.000]),
                np.array([4.000, 12.000, 20.000, 28.000]),
                'default',
                'default',
                'combine',
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, default tolerance and decimals
        (
                np.array([1.000, 9.000, 16.000, 25.000]),
                np.array([5.000, 13.000, 20.000, 29.000]),
                np.array([0.000, 8.000, 16.000, 24.000]),
                np.array([4.000, 12.000, 20.000, 28.000]),
                'default',
                'default',
                'combine',
                0.25,
                2,
                np.array([4, 5]),
        ),
        (
                np.array([0.000, 8.000, 16.000]),
                np.array([4.000, 12.000, 20.000]),
                np.array([0.000, 8.000, 16.000, 24.000]),
                np.array([4.000, 12.000, 20.000, 28.000]),
                'default',
                'default',
                'combine',
                1.0,
                6,
                np.array([0, 1, 2, 3, 4, 5]),
        ),
        (
                np.array([0.000, 8.000, 16.000, 24.000]),
                np.array([4.000, 12.000, 20.000, 28.000]),
                np.array([0.000, 8.000, 16.000]),
                np.array([4.000, 12.000, 20.000]),
                'default',
                'default',
                'combine',
                0.75,
                6,
                np.array([0, 1, 2, 3, 4, 5]),
        ),
        # float values, all hits, tolerance of 0.5, default decimals
        (
                np.array([0.500, 8.500, 16.500, 24.500]),
                np.array([4.500, 12.500, 20.500, 28.500]),
                np.array([0.000, 8.000, 16.000, 24.000]),
                np.array([4.000, 12.000, 20.000, 28.000]),
                0.5,
                'default',
                'combine',
                1.0,
                8,
                np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        ),
        # float values, no hits, tolerance of 0.5, default decimals
        (
                np.array([1.500, 9.500, 17.500, 25.500]),
                np.array([5.500, 13.500, 21.500, 29.500]),
                np.array([0.000, 8.000, 16.000, 24.000]),
                np.array([4.000, 12.000, 20.000, 28.000]),
                0.5,
                'default',
                'combine',
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, tolerance of 0.5, default decimals
        (
                np.array([1.500, 9.500, 16.500, 25.500]),
                np.array([5.500, 13.500, 20.500, 29.500]),
                np.array([0.000, 8.000, 16.000, 24.000]),
                np.array([4.000, 12.000, 20.000, 28.000]),
                0.5,
                'default',
                'combine',
                0.25,
                2,
                np.array([4, 5]),
        ),
        (
                np.array([0.500, 8.500, 16.500]),
                np.array([4.500, 12.500, 20.500]),
                np.array([0.000, 8.000, 16.000, 24.000]),
                np.array([4.000, 12.000, 20.000, 28.000]),
                0.5,
                'default',
                'combine',
                1.0,
                6,
                np.array([0, 1, 2, 3, 4, 5]),
        ),
        (
                np.array([0.500, 8.500, 16.500, 24.500]),
                np.array([4.500, 12.500, 20.500, 28.500]),
                np.array([0.000, 8.000, 16.000]),
                np.array([4.000, 12.000, 20.000]),
                0.5,
                'default',
                'combine',
                0.75,
                6,
                np.array([0, 1, 2, 3, 4, 5]),
        ),
        # float values, all hits, default tolerance, decimals=3 (default)
        (
                np.array([0.0004, 8.0004, 16.0004, 24.0004]),
                np.array([4.0004, 12.0004, 20.0004, 28.0004]),
                np.array([0.0001, 8.0001, 16.0001, 24.0001]),
                np.array([4.0001, 12.0001, 20.0001, 28.0001]),
                'default',
                3,
                'combine',
                1.0,
                8,
                np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        ),
        # # float values, no hits, default tolerance, decimals=3 (default)
        (
                np.array([1.0004, 9.0004, 17.0004, 25.0004]),
                np.array([5.0004, 13.0004, 21.0004, 29.0004]),
                np.array([0.0001, 8.0001, 16.0001, 24.0001]),
                np.array([4.0001, 12.0001, 20.0001, 28.0001]),
                'default',
                3,
                'combine',
                0.0,
                0,
                np.array([]),
        ),
        # # float values, none < hits < all, default tolerance, decimals=3 (default)
        (
                np.array([1.0004, 9.0004, 16.0004, 25.0004]),
                np.array([5.0004, 13.0004, 20.0004, 29.0004]),
                np.array([0.0001, 8.0001, 16.0001, 24.0001]),
                np.array([4.0001, 12.0001, 20.0001, 28.0001]),
                'default',
                3,
                'combine',
                0.25,
                2,
                np.array([4, 5]),
        ),
        (
                np.array([0.0004, 8.0004, 16.0004]),
                np.array([4.0004, 12.0004, 20.0004]),
                np.array([0.0001, 8.0001, 16.0001, 24.0001]),
                np.array([4.0001, 12.0001, 20.0001, 28.0001]),
                'default',
                3,
                'combine',
                1.0,
                6,
                np.array([0, 1, 2, 3, 4, 5]),
        ),
        (
                np.array([0.0004, 8.0004, 16.0004, 24.0004]),
                np.array([4.0004, 12.0004, 20.0004, 28.0004]),
                np.array([0.0001, 8.0001, 16.0001]),
                np.array([4.0001, 12.0001, 20.0001]),
                'default',
                3,
                'combine',
                0.75,
                6,
                np.array([0, 1, 2, 3, 4, 5]),
        ),
        # # float values, all hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5004, 9.5004, 17.5004, 25.5004]),
                np.array([5.5004, 13.5004, 21.5004, 29.5004]),
                np.array([0.0001, 8.0001, 16.0001, 24.0001]),
                np.array([4.0001, 12.0001, 20.0001, 28.0001]),
                0.5,
                3,
                'combine',
                0.0,
                0,
                np.array([]),
        ),
        # # float values, no hits, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5004, 9.5004, 17.5004, 25.5004]),
                np.array([5.5004, 13.5004, 21.5004, 29.5004]),
                np.array([0.0001, 8.0001, 16.0001, 24.0001]),
                np.array([4.0001, 12.0001, 20.0001, 28.0001]),
                0.5,
                'default',
                'combine',
                0.0,
                0,
                np.array([]),
        ),
        # float values, none < hits < all, tolerance of 0.5, decimals=3 (default)
        (
                np.array([1.5004, 9.5004, 16.5004, 25.5004]),
                np.array([5.5004, 13.5004, 20.5004, 29.5004]),
                np.array([0.0001, 8.0001, 16.0001, 24.0001]),
                np.array([4.0001, 12.0001, 20.0001, 28.0001]),
                0.5,
                3,
                'combine',
                0.25,
                2,
                np.array([4, 5]),
        ),
        (
                np.array([0.5004, 8.5004, 16.5004]),
                np.array([4.5004, 12.5004, 20.5004]),
                np.array([0.0001, 8.0001, 16.0001, 24.0001]),
                np.array([4.0001, 12.0001, 20.0001, 28.0001]),
                0.5,
                3,
                'combine',
                1.0,
                6,
                np.array([0, 1, 2, 3, 4, 5]),
        ),
        (
                np.array([0.5004, 8.5004, 16.5004, 24.5004]),
                np.array([4.5004, 12.5004, 20.5004, 28.5004]),
                np.array([0.0001, 8.0001, 16.0001]),
                np.array([4.0001, 12.0001, 20.0001]),
                0.5,
                3,
                'combine',
                0.75,
                6,
                np.array([0, 1, 2, 3, 4, 5]),
        ),
    ]
)
def test_precision(onsets_hyp, offset_hyp, onsets_ref, offset_ref, tolerance, decimals, method,
                   expected_precision, expected_n_tp, expected_hits):
    if method == 'combine':
        if tolerance == 'default' and decimals == 'default':
            precision, n_tp, hits = vocalpy.metrics.segmentation.ir.precision(
                onsets_hyp, offset_hyp, onsets_ref, offset_ref, method=method,
            )
        elif tolerance != 'default' and decimals == 'default':
            precision, n_tp, hits = vocalpy.metrics.segmentation.ir.precision(
                onsets_hyp, offset_hyp, onsets_ref, offset_ref, tolerance=tolerance, method=method,
            )
        elif tolerance == 'default' and decimals != 'default':
            precision, n_tp, hits = vocalpy.metrics.segmentation.ir.precision(
                onsets_hyp, offset_hyp, onsets_ref, offset_ref, decimals=decimals, method=method,
            )
        elif tolerance != 'default' and decimals != 'default':
            precision, n_tp, hits = vocalpy.metrics.segmentation.ir.precision(
                onsets_hyp, offset_hyp, onsets_ref, offset_ref,
                tolerance=tolerance, decimals=decimals, method=method,
            )
        assert math.isclose(precision, expected_precision)
        assert n_tp == expected_n_tp
        assert np.array_equal(hits, expected_hits)
    elif method == 'separate':
        expected_precision_on, expected_precision_off = expected_precision
        expected_n_tp_on, expected_n_tp_off = expected_n_tp
        expected_hits_on, expected_hits_off = expected_hits
        if tolerance == 'default' and decimals == 'default':
            (precision_on,
             precision_off,
             n_tp_on,
             n_tp_off,
             hits_on,
             hits_off) = vocalpy.metrics.segmentation.ir.precision(
                onsets_hyp, offset_hyp, onsets_ref, offset_ref, method=method,
            )
        elif tolerance != 'default' and decimals == 'default':
            (precision_on,
             precision_off,
             n_tp_on,
             n_tp_off,
             hits_on,
             hits_off) = vocalpy.metrics.segmentation.ir.precision(
                onsets_hyp, offset_hyp, onsets_ref, offset_ref, tolerance=tolerance, method=method,
            )
        elif tolerance == 'default' and decimals != 'default':
            (precision_on,
             precision_off,
             n_tp_on,
             n_tp_off,
             hits_on,
             hits_off) = vocalpy.metrics.segmentation.ir.precision(
                onsets_hyp, offset_hyp, onsets_ref, offset_ref, decimals=decimals, method=method,
            )
        elif tolerance != 'default' and decimals != 'default':
            (precision_on,
             precision_off,
             n_tp_on,
             n_tp_off,
             hits_on,
             hits_off) = vocalpy.metrics.segmentation.ir.precision(
                onsets_hyp, offset_hyp, onsets_ref, offset_ref,
                tolerance=tolerance, decimals=decimals, method=method,
            )
        assert math.isclose(precision_on, expected_precision_on)
        assert math.isclose(precision_off, expected_precision_off)
        assert n_tp_on == expected_n_tp_on
        assert n_tp_off == expected_n_tp_off
        assert np.array_equal(hits_on, expected_hits_on)
        assert np.array_equal(hits_off, expected_hits_off)


def test_recall(onsets_hyp, offset_hyp, onsets_ref, offset_ref, tolerance, decimals, method,
                expected_recall, expected_n_tp, expected_hits):
    assert False

def test_fscore(onsets_hyp, offset_hyp, onsets_ref, offset_ref, tolerance, decimals, method,
                expected_fscore, expected_n_tp, expected_hits):
    assert False
