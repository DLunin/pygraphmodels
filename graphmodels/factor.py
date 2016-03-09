from .output import ListTable
from .meta import copy_option, methoddispatch
from .misc import constant, invert_value_mapping, dataframe_value_mapping, extract_kwarg
from .multinomial import multinomial
from itertools import repeat
from copy import deepcopy, copy
import numpy as np
import pandas as pd


class IdentityValueMapping:
    @property
    def fitted(self):
        return True

    def fit(self, data):
        return self

    def transform(self, data, copy=True):
        if data is None:
            return None
        return data.copy() if copy else data

    def inverse_transform(self, data, copy=True):
        if data is None:
            return None
        return data.copy() if copy else data

    def rename_variable(self, old_name, new_name):
        pass


class DictValueMapping:
    def __init__(self, mapping=None):
        self.mapping = mapping
        self.inverse_mapping = invert_value_mapping(self.mapping)

    def rename_variable(self, old_name, new_name):
        if old_name in self.mapping:
            self.mapping[new_name] = self.mapping[old_name]
            self.inverse_mapping[new_name] = self.inverse_mapping[old_name]
            del self.mapping[old_name]
            del self.inverse_mapping[old_name]
        return self

    @property
    def fitted(self):
        return self.mapping is not None

    def fit(self, data):
        self.mapping = dataframe_value_mapping(data)
        self.inverse_mapping = invert_value_mapping(self.mapping)
        return self

    @staticmethod
    def _transform_column(column, value_mapping):
        if isinstance(column, pd.Series):
            return column.map(value_mapping)
        elif isinstance(column, np.ndarray) or isinstance(column, list):
            return np.asarray([value_mapping[x] for x in column])
        else:
            return value_mapping[column]

    @staticmethod
    def _transform(data, value_mapping):
        if data is None:
            return None
        elif isinstance(data, pd.DataFrame):
            data = data.copy()
            for column in data.columns:
                data[column] = DictValueMapping._transform_column(data[column], value_mapping[column])
            return data
        elif isinstance(data, dict):
            return {key: DictValueMapping._transform_column(val, value_mapping[key]) for key, val in data.items()}

    def transform(self, data, copy=True):
        return DictValueMapping._transform(data, self.mapping)

    def inverse_transform(self, data, copy=True):
        return DictValueMapping._transform(data, self.inverse_mapping)


class Factor(object):
    def __init__(self, arguments, scope, value_mapping=None):
        self.arguments = list(arguments)
        self.scope = [name for name in arguments if name in scope]

        if value_mapping is None:
            self.value_mapping = IdentityValueMapping()
        else:
            self.value_mapping = value_mapping

    def _pdf(self, args):
        raise NotImplementedError()

    def _parse_kwargs(self, *args, **kwargs):
        assert len(args) <= 1
        arg_dict = args[0] if len(args) else {}
        arg_dict.update(kwargs)
        assert set(arg_dict.keys()) <= set(self.arguments)
        return self.value_mapping.transform(arg_dict)

    def _build_arglist(self, arg_dict):
        arg_list = [None] * len(self)
        for name, value in arg_dict.items():
            arg_list[self.arguments.index(name)] = value
        return arg_list

    def pdf(self, *args, **kwargs):
        arg_dict = self._parse_kwargs(*args, **kwargs)

        arg_list = self._build_arglist(arg_dict)
        return np.squeeze(self._pdf(arg_list))[()]

    def _observe(self, kwargs, copy=True):
        raise NotImplementedError()

    def observe(self, *args, **kwargs):
        copy_opt = extract_kwarg('copy', kwargs, default=True)

        arg_dict = self._parse_kwargs(*args, **kwargs)
        passed_dict = {key: val for key, val in arg_dict.items() if key in self.scope}
        return self._observe(passed_dict, copy=copy_opt)

    def __call__(self, *args, **kwargs):
        arg_dict = self._parse_kwargs(*args, **kwargs)

        arg_set = set(arg_dict.keys())
        assert arg_set <= set(self.arguments)
        if arg_set >= set(self.scope):
            arg_list = self._build_arglist(arg_dict)
            return np.squeeze(self._pdf(arg_list))[()]
        else:
            return self._observe(arg_dict)

    def fit(self, data, **kwargs):
        disable_value_mapping = extract_kwarg('disable_value_mapping', kwargs, False)
        if disable_value_mapping:
            self.value_mapping = IdentityValueMapping()
        else:
            self.value_mapping = DictValueMapping().fit(data)
            data = self.value_mapping.transform(data)
        return self._fit(data, **kwargs)

    def _rvs(self, size=1, observed=None):
        raise NotImplementedError()

    def rvs(self, size=1, observed=None):
        observed = self.value_mapping.transform(observed)
        result = self._rvs(size=size, observed=observed)
        return self.value_mapping.inverse_transform(result)

    @copy_option(default=False)
    def rename_variable(self, old_name, new_name):
        self.arguments[self.arguments.index(old_name)] = new_name
        if old_name in self.scope:
            self.scope[self.scope.index(old_name)] = new_name
        self.value_mapping.rename_variable(old_name, new_name)
        return self

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

    def __copy__(self):
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

    def __copy__(self):
        result = TableFactor(copy(self.arguments), copy(self.scope), value_mapping=copy(self.value_mapping))
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

    @copy_option(default=False)
    def set_arguments(self, new_arguments):
        old_n = len(self.arguments)
        new_n = len(new_arguments)
        table = self.table.reshape(self.table.shape + (1,)*(new_n - old_n))
        old_arguments = self.arguments + [arg for arg in new_arguments if arg not in self.arguments]
        self.table = np.transpose(table, [old_arguments.index(arg) for arg in new_arguments])
        self.arguments = new_arguments
        self.scope = [arg for arg in self.arguments if arg in self.scope]
        return self

    def _pdf(self, args):
        return self.table[tuple(arg if arg is not None else 0 for arg in args)]

    def n_values(self, var):
        assert var in self.scope
        return self.table.shape[self.arguments.index(var)]

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
        variables = [var for var in variables if var in self.scope]
        if self.table is not None and len(variables) > 0:
            self.table = np.sum(self.table, axis=tuple(map(self.arguments.index, variables)), keepdims=True)
        for var in variables:
            self.scope.remove(var)
        return self

    def _rvs(self, size=1, observed=None):
        if observed is None:
            observed = pd.DataFrame(index=np.arange(size))
        elif isinstance(observed, dict):
            observed = pd.DataFrame(data=observed)
        size = observed.shape[0]
        n_observed = len(set(observed.columns) & set(self.scope))

        obs_idx = tuple(observed[arg].values if (arg in observed.columns) and (arg in self.scope) else slice(None) for arg in self.arguments)
        fst = [0] * size if n_observed == 0 else [0]
        table = (self.table[None, ..., None])[(fst,) + obs_idx + ([0],)]
        table = table.reshape((table.shape[0], -1))
        table = table / np.sum(table, axis=1, keepdims=True)
        not_observed = set(self.scope) - set(observed.columns)
        rand_shape = tuple(self.table.shape[i] for i, arg in enumerate(self.arguments) if arg in not_observed)
        rand_idx = multinomial(p=table)

        result = np.asarray(np.unravel_index(indices=rand_idx, dims=rand_shape))
        return pd.DataFrame(data=result.T, columns=[arg for arg in self.arguments if arg in not_observed])


    @methoddispatch
    def __mul__(self, other):
        assert type(other) == TableFactor
        assert self.arguments == other.arguments
        result = TableFactor(self.arguments, self.scope + other.scope, value_mapping=self.value_mapping)
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
        result = TableFactor(self.arguments, self.scope + other.scope, value_mapping=self.value_mapping)
        result.table = self.table / other.table
        result.table[np.isinf(result.table)] = 0.
        result.table[np.isnan(result.table)] = 0.
        np.seterr(**old_err_state)

        return result

    @__div__.register(IdentityFactor)
    def _(self, other):
        return deepcopy(self)

    def _fit(self, data, n_values=None):
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
        return ListTable(squeezed, self.scope, value_mapping=self.value_mapping)._repr_html_()



