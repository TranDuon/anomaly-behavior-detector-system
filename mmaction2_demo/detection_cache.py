"""Reuse detector output only for byte-identical frames in one worker."""

from collections import OrderedDict
import hashlib

import numpy as np


class ExactDetectionCache:
    def __init__(self, detect, capacity=24):
        self.detect = detect
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = 0

    def __call__(self, frame):
        pixels = np.ascontiguousarray(frame)
        key = (pixels.shape, pixels.dtype.str, hashlib.blake2b(pixels, digest_size=16).digest())
        if key not in self.entries:
            self.entries[key] = np.asarray(self.detect(frame)).copy()
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        else:
            self.hits += 1
            self.entries.move_to_end(key)
        return self.entries[key].copy()
