import pathlib
import socket
import unittest.mock

import pytest

import vocalpy.examples._examples


def test_EXAMPLE():
    assert isinstance(
        vocalpy.examples._examples.EXAMPLES, list
    )
    assert len(vocalpy.examples._examples.EXAMPLES) > 0
    assert all(
        [isinstance(
            example, vocalpy.examples._examples.Example
        ) for example in vocalpy.examples._examples.EXAMPLES]
    )


@pytest.fixture(params=[False, True])
def return_path(request):
    return request.param


@pytest.mark.parametrize(
    'example',
    vocalpy.examples._examples.EXAMPLES
)
def test_example(example, return_path):
    out = vocalpy.examples._examples.example(example.name, return_path=return_path)
    if example.type == vocalpy.examples._examples.ExampleTypes.ExampleData:
        assert isinstance(out, vocalpy.examples._examples.ExampleData)
        if return_path:
            for val in out.values():
                assert isinstance(val, (pathlib.Path, list))
                if isinstance(val, list):
                    assert all([isinstance(el, pathlib.Path) for el in val])
    else:
        if return_path:
            assert isinstance(out, pathlib.Path)
        else:
            if example.type == vocalpy.examples._examples.ExampleTypes.Sound:
                assert isinstance(out, vocalpy.Sound)
            elif example.type == vocalpy.examples._examples.ExampleTypes.Spectrogram:
                assert isinstance(out, vocalpy.Spectrogram)
            elif example.type == vocalpy.examples._examples.ExampleTypes.Annotation:
                assert isinstance(out, vocalpy.Annotation)


def test_show(capsys):
    vocalpy.examples._examples.show()
    captured = capsys.readouterr()
    for example in vocalpy.examples._examples.EXAMPLES:
        assert example.name in captured.out
        assert example.description in captured.out


@pytest.mark.parametrize(
    'name',
    [
        example.name
        for example in vocalpy.examples._examples.EXAMPLES
        if example.requires_download
    ]
)
def test_example_raises(name):
    with unittest.mock.patch(
        'urllib3.connection.connection.create_connection',
        side_effect=socket.gaierror
    ):
        with pytest.raises(ConnectionError):
            vocalpy.example(name)
