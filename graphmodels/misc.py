import numpy as np
from copy import copy


class constant:
    def __init__(self, val):
        self.val = val

    def rvs(self, size=1):
        if size == 1:
            return copy(self.val)
        return np.asarray([copy(self.val) for _ in range(size)])
