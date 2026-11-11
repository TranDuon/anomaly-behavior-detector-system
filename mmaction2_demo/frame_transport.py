"""Bounded Windows shared-memory frames for the isolated local GPU worker."""

import math
import mmap
import os
import uuid

import numpy as np


PREFIX = "FightWatchFrames_"
MAX_BYTES = 256 * 1024 * 1024


def validate_shape(shape):
    if (not isinstance(shape, (list, tuple)) or len(shape) != 4
            or any(type(value) is not int or value < 1 for value in shape)
            or shape[0] != 9 or shape[3] != 3
            or math.prod(shape) > MAX_BYTES):
        raise ValueError("Invalid shared frame shape")
    return tuple(shape)


class SharedFrameBuffer:
    def __init__(self):
        self.mapping = self.array = self.name = None

    def write(self, frames):
        if os.name != "nt":
            raise OSError("Named frame mappings require Windows")
        shape = validate_shape((len(frames), *frames[0].shape))
        if any(frame.shape != shape[1:] or frame.dtype != np.uint8 for frame in frames):
            raise ValueError("Uniform BGR uint8 frames required")
        if self.array is None or self.array.shape != shape:
            self.close()
            self.name = PREFIX + uuid.uuid4().hex
            self.mapping = mmap.mmap(-1, math.prod(shape), tagname=self.name)
            self.array = np.ndarray(shape, dtype=np.uint8, buffer=self.mapping)
        for index, frame in enumerate(frames):
            self.array[index] = frame
        return {"transport": "shared_memory", "mapping": self.name, "shape": list(shape)}

    def close(self):
        self.array = None
        if self.mapping is not None:
            self.mapping.close()
        self.mapping = self.name = None


class SharedFrameReader:
    def __init__(self):
        self.mapping = self.array = self.name = None

    def read(self, request):
        name = request.get("mapping", "")
        suffix = name.removeprefix(PREFIX)
        if not name.startswith(PREFIX) or len(suffix) != 32 or any(value not in "0123456789abcdef" for value in suffix):
            raise ValueError("Invalid shared frame mapping")
        shape = validate_shape(request.get("shape"))
        if self.name != name or self.array is None or self.array.shape != shape:
            self.close()
            self.mapping = mmap.mmap(-1, math.prod(shape), tagname=name, access=mmap.ACCESS_READ)
            self.array = np.ndarray(shape, dtype=np.uint8, buffer=self.mapping)
            self.name = name
        return self.array

    def close(self):
        self.array = None
        if self.mapping is not None:
            self.mapping.close()
        self.mapping = self.name = None
