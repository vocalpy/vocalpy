import pytest

import vocalpy


class TestUnit:
    @pytest.mark.parametrize(
        "onset, offset",
        [
            (0.75, 1.5),
        ],
    )
    def test_init(self, onset, offset):
        unit = vocalpy.Unit(onset, offset)
        assert isinstance(unit, vocalpy.Unit)
        for attr_name, attr_val in zip(
            ("onset", "offset"),
            (onset, offset),
        ):
            assert hasattr(unit, attr_name)
            assert getattr(unit, attr_name) == attr_val

    @pytest.mark.parametrize(
        "onset, offset, expected_exception",
        [
            # because onset is greater than offset
            (1.5, 0.75, ValueError),
        ],
    )
    def test_post_init_raises(self, onset, offset, expected_exception):
        with pytest.raises(expected_exception):
            vocalpy.Unit(onset, offset)
