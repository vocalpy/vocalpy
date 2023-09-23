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
                -100,
                3,
                np.array([0, 1, 2]),
        ),
        (
                np.array([0, 5, 10]),
                np.array([0, 5, 10, 15]),
                'precision',
                'default',
                'default',
                0.75,
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
        (
            np.array([0, 5, 10, 15]),
            np.array([0, 5, 10, 15]),
            'default',
            'default',
            1.0,
            4,
            np.array([0, 1, 2, 3]),
        ),
        (
                np.array([0, 5, 10, 15]),
                np.array([1, 6, 11, 16]),
                1,
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        (
                np.array([0., 5., 10., 15.]),
                np.array([0., 5., 10., 15.]),
                'default',
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
        ),
        (
                np.array([0., 5., 10., 15.]),
                np.array([0.5, 5.5, 10.5, 15.5]),
                0.5,
                'default',
                1.0,
                4,
                np.array([0, 1, 2, 3]),
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
