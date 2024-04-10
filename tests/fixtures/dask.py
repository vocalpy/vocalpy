import pytest


PARALLEL = [
    True,
    False,
]

@pytest.fixture(params=PARALLEL)
def parallel(request):
    return request.param
