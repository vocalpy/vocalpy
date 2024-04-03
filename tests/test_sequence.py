import pytest

import vocalpy


@pytest.fixture
def units():
    onsets = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
    offsets = [0.02, 0.07, 0.125, 0.175, 0.225, 0.275]
    units = []
    for onset, offset in zip(onsets, offsets):
        units.append(vocalpy.Unit(onset=onset, offset=offset))
    return units


class TestSequence:
    def test_init(self, units):
        seq = vocalpy.Sequence(units=units)
        assert isinstance(seq, vocalpy.Sequence)
        assert seq.units == units
