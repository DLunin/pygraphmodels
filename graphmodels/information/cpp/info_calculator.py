import numpy as np
from .entcalc import EntCalc


class InfoCalculator:
    def __init__(self, data):
        self.entc = EntCalc(data.values.T)
        self.name_to_pos = {name : i for i, name in enumerate(data.columns.values)}

    def __call__(self, xvars, yvars):
        x = np.zeros(len(xvars), dtype="i4")
        y = np.zeros(len(yvars), dtype="i4")
        for i, v in enumerate(xvars):
            x[i] = self.name_to_pos[v]
        for i, v in enumerate(yvars):
            y[i] = self.name_to_pos[v]
        return self.entc.mi(x, y, 1000)