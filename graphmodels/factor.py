from .output import ListTable
from .decorators import methoddispatch, copy_option
from itertools import repeat
from copy import deepcopy
import numpy as np
import pandas as pd


class Factor(object):
    def __init__(self, arguments, scope):
        self.arguments = list(arguments)
        self.scope = [name for name in arguments if name in scope]

    def _pdf(self, *args):
        raise NotImplementedError()

    def pdf(self, *args, **kwargs):
        assert len(args) <= 1

        arg_dict = args[0] if len(args) else {}
        arg_dict.update(kwargs)

        assert set(arg_dict.keys()) <= set(self.arguments)

        arg_list = [None] * len(self)
        for name, value in arg_dict.items():
            arg_list[self.arguments.index(name)] = value

        return np.squeeze(self._pdf(*arg_list))[()]

    def __call__(self, *args, **kwargs):
        return self.pdf(*args, **kwargs)

    def __len__(self):
        return len(self.scope)

    def __str__(self):
        return type(self).__name__ + '(%s)' % ', '.join([str(arg) for arg in self.scope])

    def __repr__(self):
        return self.__str__()


class IdentityFactor(object):
    def __mul__(self, other):
        return deepcopy(other)

    def __str__(self):
        return 'IdentityFactor'

    def __repr__(self):
        return self.__str__()


class TableFactor(Factor):
    def __init__(self, arguments, scope):
        Factor.__init__(self, arguments, scope)
        self.table = None

    @property
    def fitted(self):
        return self.table is not None

    def _pdf(self, *args):
        return self.table[tuple(arg if arg is not None else 0 for arg in args)]

    @copy_option(default=True)
    def normalize(self, *variables):
        result = self / self.marginalize(*variables)
        self.table = result.table
        return self

    @copy_option(default=True)
    def marginalize(self, *variables):
        if self.table is not None and len(variables) > 0:
            self.table = np.sum(self.table, axis=tuple(map(self.arguments.index, variables)), keepdims=True)
        for var in variables:
            self.scope.remove(var)
        return self

    @methoddispatch
    def __mul__(self, other):
        assert type(other) == TableFactor
        assert self.arguments == other.arguments
        result = TableFactor(self.arguments, self.scope + other.scope)
        result.table = self.table * other.table
        return result

    @__mul__.register(IdentityFactor)
    def _(self, other):
        return deepcopy(self)

    @methoddispatch
    def __div__(self, other):
        assert type(other) == TableFactor
        assert self.arguments == other.arguments

        # here we have to correctly handle 0 entries in the denominator factor
        old_err_state = np.seterr(divide='ignore', invalid='ignore')
        result = TableFactor(self.arguments, self.scope + other.scope)
        result.table = self.table / other.table
        result.table[np.isinf(result.table)] = 0.
        result.table[np.isnan(result.table)] = 0.
        np.seterr(**old_err_state)

        return result

    @__div__.register(IdentityFactor)
    def _(self, other):
        return deepcopy(self)

    def fit(self, data, n_values=None):
        assert isinstance(data, pd.DataFrame)
        data = data[self.scope].values
        if n_values is None:
            n_values = np.max(data, axis=0) + 1
        else:
            n_values = np.asarray([n_values[arg] for arg in self.scope])

        # TODO: this is not certainly correct
        hist, _ = np.histogramdd(data, bins=n_values, normed=True, range=list(zip(repeat(-0.5), n_values - 0.5)))

        semicolon = slice(None, None, None)
        self.table = hist[tuple(semicolon if name in self.scope else np.newaxis for name in self.arguments)]
        return self

    def _repr_html_(self):
        if not self.fitted:
            return 'TableFactor(%s)' % ', '.join(map(str, self.scope))
        squeezed = np.squeeze(self.table)
        assert squeezed.ndim == len(self.scope)
        return ListTable(squeezed, self.scope)._repr_html_()
