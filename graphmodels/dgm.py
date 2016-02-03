from .output import pretty_draw
from .misc import constant
import networkx as nx
import pandas as pd
import numpy as np
from .formats import bif_parser
import os.path
from itertools import repeat
from .factor import TableFactor

class ErdosRenyiDGMGen:
    def __init__(self, n=10, p=0.5, factor_gen=None):
        self.n = n if hasattr(n, 'rvs') else constant(n)
        self.p = p if hasattr(p, 'rvs') else constant(p)
        self.factor_gen = factor_gen

    def __call__(self):
        result = DGM(nx.gnp_random_graph(self.n.rvs(), self.p.rvs()))
        for node, node_data in result.nodes(data=True):
            node_data['cpd'] = self.factor_gen(result.nodes(), ([node] + result.predecessors(node)))
            node_data['cpd'].normalize(node, copy=False)
        return result


class DGM(nx.DiGraph):
    def __init__(self, *args):
        super(DGM, self).__init__()
        if len(args) == 1:
            self.add_nodes_from(args[0].nodes(data=True))
            self.add_edges_from(args[0].edges(data=True))

    def add_argument(self, argument):
        for factor in self.factors:
            factor.add_argument(argument, copy=False)
        return self

    def fit(self, data, *args, **kwargs):
        for node, node_data in self.nodes(data=True):
            if 'cpd' not in node_data:
                raise Exception('cpd not specified for node %s' % str(node))
            elif isinstance(node_data['cpd'], type):
                node_data['cpd'] = node_data['cpd'](self.nodes(), ([node] + self.predecessors(node)))
            node_data['cpd'].fit(data, *args, **kwargs)
            node_data['cpd'].normalize(node, copy=False)
        return self

    def _rvs(self):
        result = {}
        for node in nx.topological_sort(self):
            cpd = self.cpd(node)
            conditional_cpd = cpd({parent: result[parent][0] for parent in self.predecessors(node)})
            result.update(conditional_cpd.rvs().to_dict())
        return pd.DataFrame(data=result)

    def rvs(self, size=1):
        result = []
        for i in range(size):
            result.append(self._rvs())
        return pd.concat(result, ignore_index=True)

    @property
    def factors(self):
        for node, node_data in self.nodes(data=True):
            yield node_data['cpd']

    def cpd(self, var):
        return self.node[var]['cpd'].normalize(var)

    def factor(self, var):
        return self.node[var]['cpd']

    def draw(self):
        return pretty_draw(self)

    @staticmethod
    def read(filename):
        name, ext = os.path.splitext(filename)
        return getattr(DGM, '_read_' + ext[1:])(filename)

    @staticmethod
    def _read_bif(filename):
        with open(filename, 'r') as f:
            text = f.read()
        parsed = bif_parser.parse(text)

        result = DGM()
        arguments = [var['name'] for var in parsed['variables']]

        values = {}

        for variable in parsed['variables']:
            name = variable['name']
            result.add_node(name, attr_dict=variable['properties'], n_values=variable['n_values'])
            values[name] = variable['values_list']

        for distribution in parsed['distributions']:
            node = distribution['variables'][0]
            parents = distribution['variables'][1:]
            scope = distribution['variables']
            result.add_edges_from(zip(parents, repeat(node)))
            factor = TableFactor(arguments=arguments, scope=distribution['variables'])

            table_shape = tuple(result.node[var]['n_values'] for var in scope) + (1,) * (len(arguments) - len(scope))
            extended_scope = scope + [var for var in arguments if var not in scope]
            axes_order = [extended_scope.index(arg) for arg in arguments]
            if distribution['table'] is not None:
                factor.table = distribution['table']
                factor.table.resize(np.prod(table_shape))
                factor.table = factor.table.reshape(table_shape)
                factor.table = np.transpose(factor.table, axes=axes_order)
            else:
                factor.table = np.zeros(table_shape)
                for args, prob in distribution['probability'].items():
                    for i, p in enumerate(prob):
                        current_args = (i, ) + tuple(values[var].index(arg) for var, arg in zip(parents, args)) + (0,) * (len(arguments) - len(scope))
                        factor.table[current_args] = p
                factor.table = np.transpose(factor.table, axes=axes_order)

            result.node[node]['cpd'] = factor

        return result