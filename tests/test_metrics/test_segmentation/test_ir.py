from __future__ import annotations

import math

import attr
import attrs
import numpy as np
import numpy.typing as npt
import pytest

import vocalpy.metrics.segmentation.ir


@attrs.define
class IRMetricTestCase:
    """Class representing a test case for a unit test
    in the vocalpy.metric.segmentation.ir module

    This avoids repeating the same test case for different
    functions that only differ by a single value
    (e.g., precision v. recall)
    """
    reference: npt.NDArray = attr.field()
    hypothesis: npt.NDArray = attr.field()
    expected_hits_ref: npt.NDArray = attr.field()
    expected_hits_hyp: npt.NDArray = attr.field()
    expected_diffs: npt.NDArray = attr.field()
    expected_precision: float = attr.field()
    expected_recall: float = attr.field()
    expected_fscore: float = attr.field()
    tolerance: float | None | str = attr.field(default=None)
    decimals: bool | int | None  = attr.field(default=None)
    expected_n_tp: int = attr.field(init=False)

    def __attrs_post_init__(self):
        self.expected_n_tp = self.expected_hits_ref.size


IR_METRICS_PARAMS_VALS = [
    # # integer event times
    # ## default tolerance + precision
    # ### all hits
    IRMetricTestCase(
        reference=np.array([0, 5, 10, 15]),
        hypothesis=np.array([0, 5, 10, 15]),
        tolerance=None,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2, 3]),
        expected_hits_hyp=np.array([0, 1, 2, 3]),
        expected_diffs=np.array([0, 0, 0, 0]),
        expected_precision=1.0,
        expected_recall=1.0,
        expected_fscore=1.0,
    ),
    # ### no hits
    IRMetricTestCase(
        reference=np.array([0, 5, 10, 15]),
        hypothesis=np.array([1, 6, 11, 16]),
        tolerance=None,
        decimals=None,
        expected_hits_ref=np.array([]),
        expected_hits_hyp=np.array([]),
        expected_diffs=np.array([]),
        expected_precision=0.0,
        expected_recall=0.0,
        expected_fscore=0.0,
    ),
    # ### no > hits > all
    IRMetricTestCase(
        reference=np.array([0, 5, 10, 15]),
        hypothesis=np.array([1, 6, 10, 16]),
        tolerance=None,
        decimals=None,
        expected_hits_ref=np.array([2]),
        expected_hits_hyp=np.array([2]),
        expected_diffs=np.array([0]),
        expected_precision=0.25,
        expected_recall=0.25,
        expected_fscore=0.25,
    ),
    IRMetricTestCase(
        reference=np.array([0, 5, 10, 15]),
        hypothesis=np.array([0, 5, 10]),
        tolerance=None,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 1, 2]),
        expected_diffs=np.array([0, 0, 0]),
        expected_precision=1.0,
        expected_recall=0.75,
        expected_fscore=0.8571428571428571,
    ),
    IRMetricTestCase(
        reference=np.array([0, 5, 10]),
        hypothesis=np.array([0, 5, 10, 15]),
        tolerance=None,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 1, 2]),
        expected_diffs=np.array([0, 0, 0]),
        expected_precision=0.75,
        expected_recall=1.0,
        expected_fscore=0.8571428571428571,
    ),
    # ## tolerance of 1 (default precision)
    # ### all hits
    IRMetricTestCase(
        reference=np.array([0, 5, 10, 15]),
        hypothesis=np.array([1, 6, 11, 16]),
        tolerance=1,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2, 3]),
        expected_hits_hyp=np.array([0, 1, 2, 3]),
        expected_diffs=np.array([1, 1, 1, 1]),
        expected_precision=1.0,
        expected_recall=1.0,
        expected_fscore=1.0,
    ),
    # ### no hits
    IRMetricTestCase(
        reference=np.array([0, 5, 10, 15]),
        hypothesis=np.array([2, 7, 12, 17]),
        tolerance=1,
        decimals=None,
        expected_hits_ref=np.array([]),
        expected_hits_hyp=np.array([]),
        expected_diffs=np.array([]),
        expected_precision=0.0,
        expected_recall=0.0,
        expected_fscore=0.0,
    ),
    # ### no > hits > all
    IRMetricTestCase(
        reference=np.array([0, 5, 10, 15]),
        hypothesis=np.array([2, 7, 11, 17]),
        tolerance=1,
        decimals=None,
        expected_hits_ref=np.array([2]),
        expected_hits_hyp=np.array([2]),
        expected_diffs=np.array([1]),
        expected_precision=0.25,
        expected_recall=0.25,
        expected_fscore=0.25,
    ),
    IRMetricTestCase(
        reference=np.array([0, 5, 10, 15]),
        hypothesis=np.array([1, 6, 11]),
        tolerance=1,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 1, 2]),
        expected_diffs=np.array([1, 1, 1]),
        expected_precision=1.0,
        expected_recall=0.75,
        expected_fscore=0.8571428571428571,
    ),
    IRMetricTestCase(
        reference=np.array([0, 5, 10]),
        hypothesis=np.array([1, 6, 11, 16]),
        tolerance=1,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 1, 2]),
        expected_diffs=np.array([1, 1, 1]),
        expected_precision=0.75,
        expected_recall=1.0,
        expected_fscore=0.8571428571428571,
    ),
    # ### multiple hits, tests we only keep one
    IRMetricTestCase(
        reference=np.array([0, 5, 10]),
        hypothesis=np.array([0, 1, 6, 11]),
        tolerance=1,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 2, 3]),
        expected_diffs=np.array([0, 1, 1]),
        expected_precision=0.75,
        expected_recall=1.0,
        expected_fscore=0.8571428571428571,
    ),
    IRMetricTestCase(
        reference=np.array([0, 5, 10]),
        hypothesis=np.array([0, 1, 2, 5, 6, 7, 10, 11]),
        tolerance=1,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 3, 6]),
        expected_diffs=np.array([0, 0, 0]),
        expected_precision=0.375,
        expected_recall=1.0,
        expected_fscore=0.5454545454545454,
    ),
    # # float event times
    # ## default tolerance and precision
    # ### all hits
    IRMetricTestCase(
        reference=np.array([0.000, 5.000, 10.000, 15.000]),
        hypothesis=np.array([0.000, 5.000, 10.000, 15.000]),
        tolerance=None,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2, 3]),
        expected_hits_hyp=np.array([0, 1, 2, 3]),
        expected_diffs=np.array([0, 0, 0, 0]),
        expected_precision=1.0,
        expected_recall=1.0,
        expected_fscore=1.0,
    ),
    # ### no hits
    IRMetricTestCase(
        reference=np.array([0.000, 5.000, 10.000, 15.000]),
        hypothesis=np.array([1.000, 6.000, 11.000, 16.000]),
        tolerance=None,
        decimals=None,
        expected_hits_ref=np.array([]),
        expected_hits_hyp=np.array([]),
        expected_diffs=np.array([]),
        expected_precision=0.0,
        expected_recall=0.0,
        expected_fscore=0.0,
    ),
    # ### no > hits > all
    IRMetricTestCase(
        reference=np.array([0.000, 5.000, 10.000, 15.000]),
        hypothesis=np.array([1.000, 6.000, 10.000, 16.000]),
        tolerance=None,
        decimals=None,
        expected_hits_ref=np.array([2]),
        expected_hits_hyp=np.array([2]),
        expected_diffs=np.array([0]),
        expected_precision=0.25,
        expected_recall=0.25,
        expected_fscore=0.25,
    ),
    IRMetricTestCase(
        reference=np.array([0.000, 5.000, 10.000, 15.000]),
        hypothesis=np.array([0.000, 5.000, 10.000]),
        tolerance=None,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 1, 2]),
        expected_diffs=np.array([0, 0, 0]),
        expected_precision=1.0,
        expected_recall=0.75,
        expected_fscore=0.8571428571428571,
    ),
    IRMetricTestCase(
        reference=np.array([0.000, 5.000, 10.000]),
        hypothesis=np.array([0.000, 5.000, 10.000, 15.000]),
        tolerance=None,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 1, 2]),
        expected_diffs=np.array([0, 0, 0]),
        expected_precision=0.75,
        expected_recall=1.0,
        expected_fscore=0.8571428571428571,
    ),
    # ## tolerance of 0.5
    # ### all hits
    IRMetricTestCase(
        reference=np.array([0.000, 5.000, 10.000, 15.000]),
        hypothesis=np.array([0.500, 5.500, 10.500, 15.500]),
        tolerance=0.5,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2, 3]),
        expected_hits_hyp=np.array([0, 1, 2, 3]),
        expected_diffs=np.array([0.5, 0.5, 0.5, 0.5]),
        expected_precision=1.0,
        expected_recall=1.0,
        expected_fscore=1.0,
    ),
    # ### no hits
    IRMetricTestCase(
        reference=np.array([0.000, 5.000, 10.000, 15.000]),
        hypothesis=np.array([1.500, 6.500, 11.500, 16.500]),
        tolerance=0.5,
        decimals=None,
        expected_hits_ref=np.array([]),
        expected_hits_hyp=np.array([]),
        expected_diffs=np.array([]),
        expected_precision=0.0,
        expected_recall=0.0,
        expected_fscore=0.0,
    ),
    # ### no > hits > all
    IRMetricTestCase(
        reference=np.array([0.000, 5.000, 10.000, 15.000]),
        hypothesis=np.array([1.500, 6.500, 10.500, 16.500]),
        tolerance=0.5,
        decimals=None,
        expected_hits_ref=np.array([2]),
        expected_hits_hyp=np.array([2]),
        expected_diffs=np.array([0.5]),
        expected_precision=0.25,
        expected_recall=0.25,
        expected_fscore=0.25,
    ),
    IRMetricTestCase(
        reference=np.array([0.000, 5.000, 10.000, 15.000]),
        hypothesis=np.array([0.500, 5.500, 10.500]),
        tolerance=0.5,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 1, 2]),
        expected_diffs=np.array([0.5, 0.5, 0.5]),
        expected_precision=1.0,
        expected_recall=0.75,
        expected_fscore=0.8571428571428571,
    ),
    IRMetricTestCase(
        reference=np.array([0.000, 5.000, 10.000]),
        hypothesis=np.array([0.500, 5.500, 10.500, 15.500]),
        tolerance=0.5,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 1, 2]),
        expected_diffs=np.array([0.5, 0.5, 0.5]),
        expected_precision=0.75,
        expected_recall=1.0,
        expected_fscore=0.8571428571428571,
    ),
    # ### multiple hits, tests we only keep one
    IRMetricTestCase(
        reference=np.array([0.000, 5.000, 10.000]),
        hypothesis=np.array([0.500, 1.500, 5.500, 10.500]),
        tolerance=0.5,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 2, 3]),
        expected_diffs=np.array([0.5, 0.5, 0.5]),
        expected_precision=0.75,
        expected_recall=1.0,
        expected_fscore=0.8571428571428571,
    ),
    IRMetricTestCase(
        reference=np.array([0.000, 5.000, 10.000]),
        hypothesis=np.array([0.250, 0.500, 2.500, 5.000, 5.500, 7.500, 10.500, 11.500]),
        tolerance=0.5,
        decimals=None,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 3, 6]),
        expected_diffs=np.array([0.25, 0, 0.5]),
        expected_precision=0.375,
        expected_recall=1.0,
        expected_fscore=0.5454545454545454,
    ),
    # ## default tolerance, precision=3 (happens to be default)
    # ### all hits
    IRMetricTestCase(
        reference=np.array([0.0001, 5.0001, 10.0001, 15.0001]),
        hypothesis=np.array([0.0004, 5.0004, 10.0004, 15.0004]),
        tolerance=None,
        decimals=3,
        expected_hits_ref=np.array([0, 1, 2, 3]),
        expected_hits_hyp=np.array([0, 1, 2, 3]),
        expected_diffs=np.array([0., 0., 0., 0.]),
        expected_precision=1.0,
        expected_recall=1.0,
        expected_fscore=1.0,
    ),
    # ### no hits
    IRMetricTestCase(
        reference=np.array([0.0001, 5.0001, 10.0001, 15.0001]),
        hypothesis=np.array([1.0001, 6.0001, 11.0001, 16.0001]),
        tolerance=None,
        decimals=3,
        expected_hits_ref=np.array([]),
        expected_hits_hyp=np.array([]),
        expected_diffs=np.array([]),
        expected_precision=0.0,
        expected_recall=0.0,
        expected_fscore=0.0,
    ),
    # ### no > hits > all
    IRMetricTestCase(
        reference=np.array([1.0001, 6.0001, 10.0004, 16.0001]),
        hypothesis=np.array([0.0001, 5.0001, 10.0001, 15.0001]),
        tolerance=None,
        decimals=3,
        expected_hits_ref=np.array([2]),
        expected_hits_hyp=np.array([2]),
        expected_diffs=np.array([0]),
        expected_precision=0.25,
        expected_recall=0.25,
        expected_fscore=0.25,
    ),
    IRMetricTestCase(
        reference=np.array([0.0001, 5.0001, 10.0001, 15.0001]),
        hypothesis=np.array([0.0004, 5.0004, 10.0004]),
        tolerance=None,
        decimals=3,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 1, 2]),
        expected_diffs=np.array([0, 0, 0]),
        expected_precision=1.0,
        expected_recall=0.75,
        expected_fscore=0.8571428571428571,
    ),
    IRMetricTestCase(
        reference=np.array([0.0001, 5.0001, 10.0001]),
        hypothesis=np.array([0.0004, 5.0004, 10.0004, 15.0004]),
        tolerance=None,
        decimals=3,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 1, 2]),
        expected_diffs=np.array([0, 0, 0]),
        expected_precision=0.75,
        expected_recall=1.0,
        expected_fscore=0.8571428571428571,
    ),
    # ## tolerance of 0.5, decimals=3 (default)
    # ### all hits
    IRMetricTestCase(
        reference=np.array([0.0004, 5.0004, 10.0004, 15.0004]),
        hypothesis=np.array([0.5001, 5.5001, 10.5001, 15.5001]),
        tolerance=0.5,
        decimals=3,
        expected_hits_ref=np.array([0, 1, 2, 3]),
        expected_hits_hyp=np.array([0, 1, 2, 3]),
        expected_diffs=np.array([0.5, 0.5, 0.5, 0.5]),
        expected_precision=1.0,
        expected_recall=1.0,
        expected_fscore=1.0,
    ),
    # ### no hits
    IRMetricTestCase(
        reference=np.array([0.0004, 5.0004, 10.0004, 15.0004]),
        hypothesis=np.array([1.5001, 6.5001, 11.5001, 16.5001]),
        tolerance=0.5,
        decimals=3,
        expected_hits_ref=np.array([]),
        expected_hits_hyp=np.array([]),
        expected_diffs=np.array([]),
        expected_precision=0.0,
        expected_recall=0.0,
        expected_fscore=0.0,
    ),
    # ### no > hits > all
    IRMetricTestCase(
        reference=np.array([0.0004, 5.0004, 10.0004, 15.0004]),
        hypothesis=np.array([1.5001, 6.5001, 10.5001, 16.5001]),
        tolerance=0.5,
        decimals=3,
        expected_hits_ref=np.array([2]),
        expected_hits_hyp=np.array([2]),
        expected_diffs=np.array([0.5]),
        expected_precision=0.25,
        expected_recall=0.25,
        expected_fscore=0.25,
    ),
    IRMetricTestCase(
        reference=np.array([0.0001, 5.0001, 10.0001, 15.0001]),
        hypothesis=np.array([0.5004, 5.5004, 10.5004]),
        tolerance=0.5,
        decimals=3,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 1, 2]),
        expected_diffs=np.array([0.5, 0.5, 0.5]),
        expected_precision=1.0,
        expected_recall=0.75,
        expected_fscore=0.8571428571428571,
    ),
    IRMetricTestCase(
        reference=np.array([0.0001, 5.0001, 10.0001]),
        hypothesis=np.array([0.5004, 5.5004, 10.5004, 15.5004]),
        tolerance=0.5,
        decimals=3,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 1, 2]),
        expected_diffs=np.array([0.5, 0.5, 0.5]),
        expected_precision=0.75,
        expected_recall=1.0,
        expected_fscore=0.8571428571428571,
    ),
    # ### multiple hits, tests we only keep one
    # TODO: fix this to use more decimal places
    IRMetricTestCase(
        reference=np.array([0.000, 5.000, 10.000]),
        hypothesis=np.array([0.500, 1.500, 5.500, 10.500]),
        tolerance=0.5,
        decimals=3,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 2, 3]),
        expected_diffs=np.array([0.5, 0.5, 0.5]),
        expected_precision=0.75,
        expected_recall=1.0,
        expected_fscore=0.8571428571428571,
    ),
    IRMetricTestCase(
        reference=np.array([0.000, 5.000, 10.000]),
        hypothesis=np.array([0.250, 0.500, 2.500, 5.000, 5.500, 7.500, 10.500, 11.500]),
        tolerance=0.5,
        decimals=3,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 3, 6]),
        expected_diffs=np.array([0.25, 0, 0.5]),
        expected_precision=0.375,
        expected_recall=1.0,
        expected_fscore=0.5454545454545454,
    ),
]


@pytest.mark.parametrize(
    'ir_metric_test_case',
    IR_METRICS_PARAMS_VALS,
)
def test_find_hits(ir_metric_test_case):
    (hypothesis,
     reference,
     tolerance,
     decimals,
     expected_hits_ref,
     expected_hits_hyp,
     expected_diffs,
     expected_n_tp) = (
        ir_metric_test_case.hypothesis,
        ir_metric_test_case.reference,
        ir_metric_test_case.tolerance,
        ir_metric_test_case.decimals,
        ir_metric_test_case.expected_hits_ref,
        ir_metric_test_case.expected_hits_hyp,
        ir_metric_test_case.expected_diffs,
        ir_metric_test_case.expected_n_tp,
    )

    hits_ref, hits_hyp, diffs = vocalpy.metrics.segmentation.ir.find_hits(
        hypothesis, reference, tolerance=tolerance, decimals=decimals
    )

    assert np.array_equal(hits_ref, expected_hits_ref)
    assert np.array_equal(hits_hyp, expected_hits_hyp)
    assert np.array_equal(diffs, expected_diffs)


# TODO: define ``metric`` fixture we use to parametrize the test for `precision_recall_fscore`
# so we can trivially get that Cartesian product of tests


IR_METRIC_NAMES =[
    "precision",
    "recall",
    "fscore",
]


@pytest.fixture(params=IR_METRIC_NAMES)
def ir_metric_name(request):
    return request.param


@pytest.mark.parametrize(
    'ir_metric_test_case',
    IR_METRICS_PARAMS_VALS
)
def test_precision_recall_fscore(ir_metric_test_case, ir_metric_name):
    (hypothesis,
     reference,
     tolerance,
     decimals,
     expected_hits_ref,
     expected_hits_hyp,
     expected_diffs,
     expected_n_tp) = (
        ir_metric_test_case.hypothesis,
        ir_metric_test_case.reference,
        ir_metric_test_case.tolerance,
        ir_metric_test_case.decimals,
        ir_metric_test_case.expected_hits_ref,
        ir_metric_test_case.expected_hits_hyp,
        ir_metric_test_case.expected_diffs,
        ir_metric_test_case.expected_n_tp,
    )

    metric_value, n_tp, ir_metric_data = vocalpy.metrics.segmentation.ir.precision_recall_fscore(
        hypothesis, reference, ir_metric_name, tolerance=tolerance, decimals=decimals
    )

    if ir_metric_name == "precision":
        expected_metric_value = ir_metric_test_case.expected_precision
    elif ir_metric_name == "recall":
        expected_metric_value = ir_metric_test_case.expected_recall
    elif ir_metric_name == "fscore":
        expected_metric_value = ir_metric_test_case.expected_fscore
    else:
        raise ValueError(f"unknown ir_metric_name: {ir_metric_name}")

    assert math.isclose(metric_value, expected_metric_value)
    assert n_tp == expected_n_tp
    assert np.array_equal(ir_metric_data.hits_ref, expected_hits_ref)
    assert np.array_equal(ir_metric_data.hits_hyp, expected_hits_hyp)
    assert np.array_equal(ir_metric_data.diffs, expected_diffs)


@pytest.mark.parametrize(
    'ir_metric_test_case',
    IR_METRICS_PARAMS_VALS
)
def test__precision(ir_metric_test_case):
    (hypothesis,
     reference,
     tolerance,
     decimals,
     expected_hits_ref,
     expected_hits_hyp,
     expected_diffs,
     expected_n_tp) = (
        ir_metric_test_case.hypothesis,
        ir_metric_test_case.reference,
        ir_metric_test_case.tolerance,
        ir_metric_test_case.decimals,
        ir_metric_test_case.expected_hits_ref,
        ir_metric_test_case.expected_hits_hyp,
        ir_metric_test_case.expected_diffs,
        ir_metric_test_case.expected_n_tp,
    )

    precision, n_tp, ir_metric_data = vocalpy.metrics.segmentation.ir._precision(
        hypothesis, reference, tolerance=tolerance, decimals=decimals
    )

    assert math.isclose(precision, ir_metric_test_case.expected_precision)
    assert n_tp == expected_n_tp
    assert np.array_equal(ir_metric_data.hits_ref, expected_hits_ref)
    assert np.array_equal(ir_metric_data.hits_hyp, expected_hits_hyp)
    assert np.array_equal(ir_metric_data.diffs, expected_diffs)


@pytest.mark.parametrize(
    'ir_metric_test_case',
    IR_METRICS_PARAMS_VALS
)
def test__recall(ir_metric_test_case):
    (hypothesis,
     reference,
     tolerance,
     decimals,
     expected_hits_ref,
     expected_hits_hyp,
     expected_diffs,
     expected_n_tp) = (
        ir_metric_test_case.hypothesis,
        ir_metric_test_case.reference,
        ir_metric_test_case.tolerance,
        ir_metric_test_case.decimals,
        ir_metric_test_case.expected_hits_ref,
        ir_metric_test_case.expected_hits_hyp,
        ir_metric_test_case.expected_diffs,
        ir_metric_test_case.expected_n_tp,
    )

    recall, n_tp, ir_metric_data = vocalpy.metrics.segmentation.ir._recall(
        hypothesis, reference, tolerance=tolerance, decimals=decimals
    )

    assert math.isclose(recall, ir_metric_test_case.expected_recall)
    assert n_tp == expected_n_tp
    assert np.array_equal(ir_metric_data.hits_ref, expected_hits_ref)
    assert np.array_equal(ir_metric_data.hits_hyp, expected_hits_hyp)
    assert np.array_equal(ir_metric_data.diffs, expected_diffs)


@pytest.mark.parametrize(
    'ir_metric_test_case',
    IR_METRICS_PARAMS_VALS
)
def test__fscore(ir_metric_test_case):
    (hypothesis,
     reference,
     tolerance,
     decimals,
     expected_hits_ref,
     expected_hits_hyp,
     expected_diffs,
     expected_n_tp) = (
        ir_metric_test_case.hypothesis,
        ir_metric_test_case.reference,
        ir_metric_test_case.tolerance,
        ir_metric_test_case.decimals,
        ir_metric_test_case.expected_hits_ref,
        ir_metric_test_case.expected_hits_hyp,
        ir_metric_test_case.expected_diffs,
        ir_metric_test_case.expected_n_tp,
    )

    fscore, n_tp, ir_metric_data = vocalpy.metrics.segmentation.ir._fscore(
        hypothesis, reference, tolerance=tolerance, decimals=decimals
    )

    assert math.isclose(fscore, ir_metric_test_case.expected_fscore)
    assert n_tp == expected_n_tp
    assert np.array_equal(ir_metric_data.hits_ref, expected_hits_ref)
    assert np.array_equal(ir_metric_data.hits_hyp, expected_hits_hyp)
    assert np.array_equal(ir_metric_data.diffs, expected_diffs)



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
