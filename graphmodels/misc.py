import numpy as np
from copy import copy


class constant:
    def __init__(self, val):
        self.val = val

    def rvs(self, size=1):
        if size == 1:
            return copy(self.val)
        return np.asarray([copy(self.val) for _ in range(size)])


def invert_value_mapping(value_mapping):
    if value_mapping is None:
        return None
    result = {}
    for name, mapping in value_mapping.items():
        if isinstance(mapping, dict):
            result[name] = {val: key for key, val in mapping.items()}
    return result
