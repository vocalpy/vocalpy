import socket
import unittest.mock

import pytest

import vocalpy.examples._examples


def test_EXAMPLE_METADATA():
    assert isinstance(
        vocalpy.examples._examples.EXAMPLE_METADATA, list
    )
    assert len(vocalpy.examples._examples.EXAMPLE_METADATA) > 0
    assert all(
        [isinstance(
            example, vocalpy.examples._examples.ExampleMeta
        ) for example in vocalpy.examples._examples.EXAMPLE_METADATA]
    )


@pytest.mark.parametrize(
    'example',
    vocalpy.examples._examples.EXAMPLE_METADATA
)
def test_example(example):
    out = vocalpy.examples._examples.example(example.name)
    if example.type == 'audio':
        assert isinstance(out, vocalpy.Sound)


def test_show(capsys):
    vocalpy.examples._examples.show()
    captured = capsys.readouterr()
    for example in vocalpy.examples._examples.EXAMPLE_METADATA:
        assert example.name in captured.out
        assert example.metadata in captured.out


@pytest.mark.parametrize(
    'example_name',
    [
        metadata.name
        for metadata in vocalpy.examples._examples.EXAMPLE_METADATA
        if metadata.requires_download
    ]
)
def test_example_raises(example_name):
    with unittest.mock.patch(
        'urllib3.connection.connection.create_connection',
        side_effect=socket.gaierror
    ):
        with pytest.raises(ConnectionError):
            vocalpy.example(example_name)
