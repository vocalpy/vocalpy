import pytest

import vocalpy.examples._registry


def test_EXAMPLES():
    assert isinstance(
        vocalpy.examples._registry.EXAMPLES, list
    )
    assert len(vocalpy.examples._registry.EXAMPLES) > 0
    assert all(
        [isinstance(
            example, vocalpy.examples._registry.Example
        ) for example in vocalpy.examples._registry.EXAMPLES]
    )


@pytest.mark.parametrize(
    'example',
    vocalpy.examples._registry.EXAMPLES
)
def test_example(example):
    out = vocalpy.examples._registry.example(example.name)
    if example.type == 'audio':
        assert isinstance(out, vocalpy.Audio)


def test_list(capsys):
    vocalpy.examples._registry.list()
    captured = capsys.readouterr()
    for example in vocalpy.examples._registry.EXAMPLES:
        assert example.name in captured.out
        assert example.metadata in captured.out
