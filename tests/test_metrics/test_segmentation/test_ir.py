import math

import numpy as np
import pytest

import vocalpy.metrics.segmentation.ir


@pytest.mark.parametrize(
    'hypothesis, reference, tolerance, decimals, expected_n_tp, expected_hits',
    [
        # ---- int values -----
        # int values, all hits, default tolerance and decimals
        (
            np.array([0, 5, 10, 15]),
            np.array([0, 5, 10, 15]),
            'default',
            'default',
            4,
            np.array([0, 1, 2, 3]),
        ),
        # int values, no hits, default tolerance and decimals
        (
                np.array([0, 5, 10, 15]),
                np.array([1, 6, 11, 16]),
                'default',
                'default',
                0,
                np.array([]),
        ),
        # int values, no > hits > all, default tolerance and decimals
        (
                np.array([0, 5, 10, 15]),
                np.array([1, 6, 10, 16]),
                'default',
                'default',
                1,
                np.array([2]),
        ),
        # int values, all hits, tolerance of one, default decimals
        (
                np.array([0, 5, 10, 15]),
                np.array([1, 6, 11, 16]),
                1,
                'default',
                4,
                np.array([0, 1, 2, 3]),
        ),
        # int values, no hits, tolerance of one, default decimals
        (
                np.array([0, 5, 10, 15]),
                np.array([2, 7, 12, 17]),
                1,
                'default',
                0,
                np.array([]),
        ),
        # int values, no > hits > all, tolerance of one, default decimals
        (
                np.array([0, 5, 10, 15]),
                np.array([2, 7, 11, 17]),
                1,
                'default',
                1,
                np.array([2]),
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
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([1.000, 6.000, 11.000, 16.000]),
                'default',
                'default',
                0,
                np.array([]),
        ),
        # float values, none < hits < all, default tolerance and decimals
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([1.000, 6.000, 10.000, 16.000]),
                'default',
                'default',
                1,
                np.array([2]),
        ),
        # float values, all hits, tolerance of 0.5, default decimals
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([0.500, 5.500, 10.500, 15.500]),
                0.5,
                'default',
                4,
                np.array([0, 1, 2, 3]),
        ),
        # float values, no hits, tolerance of 0.5, default decimals
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([1.500, 6.500, 11.500, 16.500]),
                0.5,
                'default',
                0,
                np.array([]),
        ),
        # float values, none < hits < all, tolerance of 0.5, default decimals
        (
                np.array([0.000, 5.000, 10.000, 15.000]),
                np.array([1.500, 6.500, 11.500, 16.500]),
                0.5,
                'default',
                0,
                np.array([]),
        ),
    ]
)
def test_compute_true_positives(hypothesis, reference, tolerance, decimals, expected_n_tp, expected_hits):
    if tolerance == 'default' and decimals == 'default':
        n_tp, hits = vocalpy.metrics.segmentation.ir.compute_true_positives(
            hypothesis, reference
        )
        assert n_tp == expected_n_tp
        assert np.array_equal(hits, expected_hits)


@pytest.mark.parametrize(
    'hypothesis, reference, tolerance, decimals, expected_precision, expected_n_tp, expected_hits',
    [
        # ---- int values -----
        # int values, all hits, default tolerance and decimals
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
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                0.0,
                0,
                np.array([]),
        ),
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
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
def test___precision(hypothesis, reference, tolerance, decimals, expected_precision, expected_n_tp, expected_hits):
    if tolerance == 'default' and decimals == 'default':
        precision, n_tp, hits = vocalpy.metrics.segmentation.ir._precision(
            hypothesis, reference
        )
        assert math.isclose(precision, expected_precision)
        assert n_tp == expected_n_tp
        assert np.array_equal(hits, expected_hits)


@pytest.mark.parametrize(
    'hypothesis, reference, tolerance, decimals, expected_precision, expected_n_tp, expected_hits',
    [
        # ---- int values -----
        # int values, all hits, default tolerance and decimals
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
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
                'default',
                'default',
                0.0,
                0,
                np.array([]),
        ),
        (
                np.array([1, 6, 11, 16]),
                np.array([0, 5, 10, 15]),
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
def test___precision(hypothesis, reference, tolerance, decimals, expected_precision, expected_n_tp, expected_hits):
    if tolerance == 'default' and decimals == 'default':
        precision, n_tp, hits = vocalpy.metrics.segmentation.ir._precision(
            hypothesis, reference
        )
        assert math.isclose(precision, expected_precision)
        assert n_tp == expected_n_tp
        assert np.array_equal(hits, expected_hits)


@pytest.mark.parametrize(
    'hypothesis, reference, tolerance, decimals, expected_recall, expected_n_tp, expected_hits',
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
def test___recall(hypothesis, reference, tolerance, decimals, expected_recall, expected_n_tp, expected_hits):
    if tolerance == 'default' and decimals == 'default':
        recall, n_tp, hits = vocalpy.metrics.segmentation.ir._recall(
            hypothesis, reference
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
def test___fscore(hypothesis, reference, tolerance, decimals, expected_fscore, expected_n_tp, expected_hits):
    if tolerance == 'default' and decimals == 'default':
        fscore, n_tp, hits = vocalpy.metrics.segmentation.ir._fscore(
            hypothesis, reference
        )
        assert math.isclose(fscore, expected_fscore)
        assert n_tp == expected_n_tp
        assert np.array_equal(hits, expected_hits)
