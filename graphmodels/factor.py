from .output import ListTable
from .decorators import methoddispatch, copy_option
from .misc import constant, invert_value_mapping
from itertools import repeat
from copy import deepcopy, copy
import numpy as np
import pandas as pd

class Factor(object):
    def __init__(self, arguments, scope, value_mapping=None):
        self.arguments = list(arguments)
        self.scope = [name for name in arguments if name in scope]
        self.value_mapping = value_mapping

    def _pdf(self, args):
        raise NotImplementedError()

    def pdf(self, *args, **kwargs):
        assert len(args) <= 1
        arg_dict = args[0] if len(args) else {}
        arg_dict.update(kwargs)
        assert set(arg_dict.keys()) <= set(self.arguments)

        if self.value_mapping is not None:
            arg_dict = {key: self.value_mapping[key][val] for key, val in arg_dict.items()}

        arg_list = [None] * len(self)

        for name, value in arg_dict.items():
            arg_list[self.arguments.index(name)] = value

        return np.squeeze(self._pdf(arg_list))[()]

    def _observe(self, kwargs):
        raise NotImplementedError()

    def observe(self, *args, **kwargs):
        copy_opt = True
        if 'copy' in kwargs:
            copy_opt = kwargs['copy']
            del kwargs['copy']

        assert len(args) <= 1
        arg_dict = args[0] if len(args) else {}
        arg_dict.update(kwargs)
        assert set(arg_dict.keys()) <= set(self.arguments)

        if self.value_mapping is not None:
            arg_dict = {key: self.value_mapping[key][val] for key, val in arg_dict.items()}

        passed_dict = {key: val for key, val in arg_dict.items() if key in self.scope}

        return self._observe(passed_dict, copy=copy_opt)

    def __call__(self, *args, **kwargs):
        assert len(args) <= 1
        arg_dict = args[0] if len(args) else {}
        arg_dict.update(kwargs)

        if self.value_mapping is not None:
            arg_dict = {key: self.value_mapping[key][val] for key, val in arg_dict.items()}

        arg_set = set(arg_dict.keys())
        assert arg_set <= set(self.arguments)
        if arg_set >= set(self.scope):
            arg_list = [None] * len(self)
            for name, value in arg_dict.items():
                arg_list[self.arguments.index(name)] = value
            return self._pdf(arg_list)
        else:
            return self._observe(arg_dict)

    def __len__(self):
        return len(self.scope)

    def __str__(self):
        return type(self).__name__ + '(%s)' % ', '.join([str(arg) for arg in self.scope])

    def __repr__(self):
        return self.__str__()


class IdentityFactor(object):
    def __init__(self, arguments):
        self.arguments = arguments

    def __mul__(self, other):
        return deepcopy(other)

    def copy(self):
        return IdentityFactor(copy(self.arguments))

    @property
    def scope(self):
        return []

    def marginalize(self, *args, **kwargs):
        return self

    def normalize(self, *args, **kwargs):
        return self

    def __str__(self):
        return 'IdentityFactor'

    def __repr__(self):
        return self.__str__()


class DirichletTableFactorGen:
    def __init__(self, n_values=None, alpha=1):
        if n_values is None:
            n_values = {}
        self.alpha = alpha if hasattr(alpha, 'rvs') else constant(alpha)
        self.n_values = n_values if hasattr(n_values, 'rvs') else constant(n_values)

    def __call__(self, arguments, scope):
        n_values = self.n_values.rvs()
        n_values.update({var: 2 for var in scope if var not in n_values})

        shape = np.asarray([n_values[var] if var in scope else 1 for var in arguments])
        n = np.prod(shape)
        alpha = self.alpha.rvs()
        alpha = np.tile(alpha, n)

        result = TableFactor(arguments, scope)
        result.table = np.random.dirichlet(alpha).reshape(shape)
        return result


class TableFactor(Factor):
    def __init__(self, arguments, scope, value_mapping=None):
        Factor.__init__(self, arguments, scope, value_mapping=value_mapping)
        self.table = None

    def copy(self):
        result = TableFactor(copy(self.arguments), copy(self.scope))
        result.table = np.copy(self.table)
        return result

    @property
    def fitted(self):
        return self.table is not None

    @copy_option(default=False)
    def add_argument(self, argument):
        self.arguments.append(argument)
        self.table = self.table.reshape(self.table.shape + (1,))
        return self

    def _pdf(self, args):
        return self.table[tuple(arg if arg is not None else 0 for arg in args)]

    @copy_option(default=True)
    def _observe(self, kwargs):
        indices = [slice(None, None, None)] * len(self.arguments)
        for name, val in kwargs.items():
            if val is not None:
                indices[self.arguments.index(name)] = slice(val, val+1, None)
                self.scope.remove(name)
        self.table = self.table[tuple(indices)]
        return self

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

    def rvs(self, size=1):
        table = self.table.flatten() / np.sum(self.table)
        indices = np.sum(np.arange(table.shape[0])[None, :] * np.random.multinomial(1, table, size=size), axis=1)
        result = np.asarray(np.unravel_index(indices, self.table.shape)).T
        result = result[:, [i for i, var in enumerate(self.arguments) if var in self.scope]]
        return pd.DataFrame(data=result, columns=self.scope)

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

    def __truediv__(self, other):
        return self.__div__(other)

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
        return ListTable(squeezed, self.scope, value_mapping=invert_value_mapping(self.value_mapping))._repr_html_()
