import numpy as np
import pytest

import vocalpy.metrics.segmentation.ir


@pytest.mark.parametrize(
    'hypothesis, reference, tolerance, decimals, expected_n_tp, expected_hits',
    [
        (
            np.array([0, 5, 10, 15]),
            np.array([0, 5, 10, 15]),
            'default',
            'default',
            4,
            np.array([0, 1, 2, 3]),
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
