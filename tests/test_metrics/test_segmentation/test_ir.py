from __future__ import annotations

import math

import attr
import attrs
import numpy as np
import numpy.typing as npt
import pytest

import vocalpy.metrics.segmentation.ir


@attrs.define
class IRMetricSingleBoundaryArrayTestCase:
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


IR_METRICS_SINGLE_BOUNDARY_ARRAY_PARAMS_VALS = [
    # # integer event times
    # ## default tolerance + precision
    # ### all hits
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
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
    IRMetricSingleBoundaryArrayTestCase(
        reference=np.array([0.0001, 5.0001, 10.0001]),
        hypothesis=np.array([0.5004, 1.5004, 5.5004, 10.5004]),
        tolerance=0.5,
        decimals=3,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 2, 3]),
        expected_diffs=np.array([0.5, 0.5, 0.5]),
        expected_precision=0.75,
        expected_recall=1.0,
        expected_fscore=0.8571428571428571,
    ),
    IRMetricSingleBoundaryArrayTestCase(
        reference=np.array([0.0001, 5.0001, 10.0001]),
        hypothesis=np.array([0.2504, 0.5004, 2.5004, 5.0004, 5.5004, 7.5004, 10.5004, 11.5004]),
        tolerance=0.5,
        decimals=3,
        expected_hits_ref=np.array([0, 1, 2]),
        expected_hits_hyp=np.array([0, 3, 6]),
        expected_diffs=np.array([0.25, 0, 0.5]),
        expected_precision=0.375,
        expected_recall=1.0,
        expected_fscore=0.5454545454545454,
    ),
    # # edge cases
    # no boundaries in reference
    IRMetricSingleBoundaryArrayTestCase(
        reference=np.array([]),
        hypothesis=np.array([1.0, 2.0, 3.0]),
        tolerance=0.5,
        decimals=3,
        expected_hits_ref=np.array([]),
        expected_hits_hyp=np.array([]),
        expected_diffs=np.array([]),
        expected_precision=0.0,
        expected_recall=0.0,
        expected_fscore=0.0,
    ),
    # no boundaries in hypothesis
    IRMetricSingleBoundaryArrayTestCase(
        reference=np.array([1.0, 2.0, 3.0]),
        hypothesis=np.array([]),
        tolerance=0.5,
        decimals=3,
        expected_hits_ref=np.array([]),
        expected_hits_hyp=np.array([]),
        expected_diffs=np.array([]),
        expected_precision=0.0,
        expected_recall=0.0,
        expected_fscore=0.0,
    ),
    # only one boundary in ref/hyp
    IRMetricSingleBoundaryArrayTestCase(
        reference=np.array([1.0]),
        hypothesis=np.array([1.0]),
        tolerance=0.5,
        decimals=3,
        expected_hits_ref=np.array([0]),
        expected_hits_hyp=np.array([0]),
        expected_diffs=np.array([0]),
        expected_precision=1.0,
        expected_recall=1.0,
        expected_fscore=1.0,
    ),
    # this is a regression test
    # see https://github.com/vocalpy/vocalpy/issues/119
    IRMetricSingleBoundaryArrayTestCase(
        reference=np.array([2.244, 2.262]),
        hypothesis=np.array([2.254]),
        tolerance=0.01,
        decimals=3,
        expected_hits_ref=np.array([1]),
        expected_hits_hyp=np.array([0]),
        expected_diffs=np.array([0.008]),
        expected_precision=1.0,
        expected_recall=0.5,
        expected_fscore=(2 * 1.0 * 0.5) / (1 + 0.5),  # 0.6666666666666666 (repeating)
    ),
    # this is a regression test
    # see https://github.com/vocalpy/vocalpy/issues/170
    IRMetricSingleBoundaryArrayTestCase(
        reference=np.array([]),
        hypothesis=np.array([]),
        tolerance=None,
        decimals=None,
        expected_hits_ref=np.array([]),
        expected_hits_hyp=np.array([]),
        expected_diffs=np.array([]),
        expected_precision=1.0,
        expected_recall=1.0,
        expected_fscore=1.0,
    ),
]


@pytest.mark.parametrize(
    'ir_metric_test_case',
    IR_METRICS_SINGLE_BOUNDARY_ARRAY_PARAMS_VALS,
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
    assert np.allclose(diffs, expected_diffs)


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
    IR_METRICS_SINGLE_BOUNDARY_ARRAY_PARAMS_VALS
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
    assert np.allclose(ir_metric_data.diffs, expected_diffs)


@pytest.mark.parametrize(
    'ir_metric_test_case',
    IR_METRICS_SINGLE_BOUNDARY_ARRAY_PARAMS_VALS
)
def test_precision(ir_metric_test_case):
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

    precision, n_tp, ir_metric_data = vocalpy.metrics.segmentation.ir.precision(
        hypothesis, reference, tolerance=tolerance, decimals=decimals
    )

    assert math.isclose(precision, ir_metric_test_case.expected_precision)
    assert n_tp == expected_n_tp
    assert np.array_equal(ir_metric_data.hits_ref, expected_hits_ref)
    assert np.array_equal(ir_metric_data.hits_hyp, expected_hits_hyp)
    assert np.allclose(ir_metric_data.diffs, expected_diffs)


@pytest.mark.parametrize(
    'ir_metric_test_case',
    IR_METRICS_SINGLE_BOUNDARY_ARRAY_PARAMS_VALS
)
def test_recall(ir_metric_test_case):
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

    recall, n_tp, ir_metric_data = vocalpy.metrics.segmentation.ir.recall(
        hypothesis, reference, tolerance=tolerance, decimals=decimals
    )

    assert math.isclose(recall, ir_metric_test_case.expected_recall)
    assert n_tp == expected_n_tp
    assert np.array_equal(ir_metric_data.hits_ref, expected_hits_ref)
    assert np.array_equal(ir_metric_data.hits_hyp, expected_hits_hyp)
    assert np.allclose(ir_metric_data.diffs, expected_diffs)


@pytest.mark.parametrize(
    'ir_metric_test_case',
    IR_METRICS_SINGLE_BOUNDARY_ARRAY_PARAMS_VALS
)
def test_fscore(ir_metric_test_case):
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

    fscore, n_tp, ir_metric_data = vocalpy.metrics.segmentation.ir.fscore(
        hypothesis, reference, tolerance=tolerance, decimals=decimals
    )

    assert math.isclose(fscore, ir_metric_test_case.expected_fscore)
    assert n_tp == expected_n_tp
    assert np.array_equal(ir_metric_data.hits_ref, expected_hits_ref)
    assert np.array_equal(ir_metric_data.hits_hyp, expected_hits_hyp)
    assert np.allclose(ir_metric_data.diffs, expected_diffs)


@pytest.mark.parametrize(
    'starts, stops, expected_out',
    [
        (
            np.array([0, 8, 16, 24]),
            np.array([4, 12, 20, 28]),
            np.array([0, 4, 8, 12, 16, 20, 24, 28])
        ),
        (
            np.array([0.000, 8.000, 16.000, 24.000]),
            np.array([4.000, 12.000, 20.000, 28.000]),
            np.array([0.000, 4.000, 8.000, 12.000, 16.000, 20.000, 24.000, 28.000])
        )
    ]
)
def test_concat_starts_and_stops(starts, stops, expected_out):
    out = vocalpy.metrics.segmentation.ir.concat_starts_and_stops(
        starts, stops
    )
    assert np.array_equal(out, expected_out)
