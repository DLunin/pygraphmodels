from ..factor import IdentityFactor
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

    def _call(self, query, observed=None):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        observed = {}
        query = args
        if 'observed' in kwargs:
            observed = kwargs['observed']
            del kwargs['observed']
        if 'target' in kwargs:
            query += kwargs['query']
            del kwargs['query']
        elif len(args) >= 1 and isinstance(args[-1], dict):
            observed.update(args[-1])
            query = args[:-1]
        observed.update(kwargs)
        return self._call(query, observed=observed)


class NaiveInference(InferenceStrategy):
    def __init__(self, gm):
        InferenceStrategy.__init__(self, gm)

    def __call__(self, query, observed=None):
        if observed is None:
            observed = {}
        eliminated = [var for var in self.arguments if var not in query]
        result = IdentityFactor(self.arguments)
        for factor in self.gm.factors:
            result = result * factor.observe(observed)
        result.marginalize(*eliminated, copy=False)
        return result.normalize(*query)


class SumProductInference(InferenceStrategy):
    def __init__(self, gm, ordering_strategy=random_ordering):
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