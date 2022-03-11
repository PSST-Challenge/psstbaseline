import os
from dataclasses import dataclass
from typing import Dict

import numpy


@dataclass
class ASRLogits(Dict[str, numpy.ndarray]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, logits in self.items():
            assert isinstance(key, str)
            assert isinstance(logits, numpy.ndarray)
            assert logits.ndim == 2

    def save(self, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        numpy.savez_compressed(filename, **self)

    @classmethod
    def load(cls, filename: str):
        with numpy.load(filename, allow_pickle=True) as npz:
            return cls(npz.items())
