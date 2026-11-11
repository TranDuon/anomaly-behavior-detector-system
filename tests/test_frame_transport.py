import os

import numpy as np
import pytest

from mmaction2_demo.frame_transport import SharedFrameBuffer, SharedFrameReader, validate_shape


@pytest.mark.parametrize("shape", [None, [9, -1, 10, 3], [8, 10, 10, 3],
                                    [9, 10000, 10000, 3], [9, 10, 10, 4]])
def test_invalid_shapes_are_rejected(shape):
    with pytest.raises(ValueError):
        validate_shape(shape)


@pytest.mark.skipif(os.name != "nt", reason="Windows named mappings")
def test_exact_frames_and_reuse_without_temporary_files():
    writer, reader = SharedFrameBuffer(), SharedFrameReader()
    try:
        frames = [np.full((20, 30, 3), index, np.uint8) for index in range(9)]
        request = writer.write(frames)
        actual = reader.read(request)
        assert np.array_equal(actual, np.stack(frames))
        name = request["mapping"]
        frames[3][:] = 99
        assert writer.write(frames)["mapping"] == name
        assert reader.read(request)[3, 0, 0, 0] == 99
        assert "frames" not in request
        with pytest.raises(ValueError):
            reader.read({**request, "mapping": "unrelated_mapping"})
    finally:
        reader.close()
        writer.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows named mappings")
def test_mapping_reallocates_when_source_shape_changes():
    writer, reader = SharedFrameBuffer(), SharedFrameReader()
    try:
        first = writer.write([np.zeros((20, 30, 3), np.uint8)] * 9)
        reader.read(first)
        second = writer.write([np.ones((24, 32, 3), np.uint8)] * 9)
        assert first["mapping"] != second["mapping"]
        assert reader.read(second).shape == (9, 24, 32, 3)
        assert reader.array.min() == 1
    finally:
        reader.close()
        writer.close()
