from .factor import IdentityFactor
import numpy as np
from functools import reduce
import operator

def random_ordering(dgm, target):
    return np.random.permutation(target)


class InferenceStrategy:
    def __init__(self, gm):
        self.gm = gm

    @property
    def arguments(self):
        return list(self.gm.nodes())


class NaiveInference(InferenceStrategy):
    def __init__(self, gm):
        InferenceStrategy.__init__(self, gm)

    def __call__(self, query, observed=None):
        if observed is not None:
            raise NotImplementedError()
        eliminated = [var for var in self.arguments if var not in query]
        result = IdentityFactor(self.arguments)
        for factor in self.gm.factors:
            result = result * factor
        result.marginalize(*eliminated, copy=False)
        return result.normalize(*query)


class SumProductInference(InferenceStrategy):
    def __init__(self, gm, ordering_strategy):
        InferenceStrategy.__init__(self, gm)
        self.ordering_strategy = ordering_strategy

    def __call__(self, query, observed=None):
        if observed is None:
            observed = {}
        eliminated = [var for var in self.arguments if var not in query and var not in observed]
        factors = [factor.observe(observed, copy=True) for factor in self.gm.factors]
        for var in self.ordering_strategy(self.gm, eliminated):
            new_factors = []
            current = IdentityFactor(self.arguments)
            for factor in factors:
                if var in factor.scope:
                    current *= factor
                else:
                    new_factors.append(factor)
            current.marginalize(var, copy=False)
            new_factors.append(current)
            factors = new_factors
        result = reduce(operator.mul, factors)
        return result.normalize(*result.scope)