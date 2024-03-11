import dataclasses

import pytest

import vocalpy.params


@dataclasses.dataclass
class ParamsSubclass(vocalpy.params.Params):
    param1: int
    param2: float


class TestParams:

    @pytest.mark.parametrize(
        'param1, param2',
        [
            (100, 0.5)
        ]
    )
    def test_unpack(self, param1, param2):
        params_subclass_instance = ParamsSubclass(param1=param1, param2=param2)
        out = {**params_subclass_instance}
        assert list(out.keys()) == ['param1', 'param2']
        assert out['param1'] == param1
        assert out['param2'] == param2

    @pytest.mark.parametrize(
        'param1, param2',
        [
            (100, 0.5)
        ]
    )
    def test_isinstance(self, param1, param2):
        params_subclass_instance = ParamsSubclass(param1=param1, param2=param2)
        assert isinstance(params_subclass_instance, vocalpy.params.Params)
