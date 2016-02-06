from .output import pretty_draw
from .misc import constant, dataframe_value_mapping, encode_dataframe
import networkx as nx
import pandas as pd
import numpy as np
from .formats import bif_parser
import os.path
from itertools import repeat, combinations
from .factor import TableFactor

def descendants(G, x):
    """
    Set of all descendants of node in a graph, not including itself.
    :param G: target graph
    :param x: target node
    :return: set of descendants
    """
    return set(nx.dfs_preorder_nodes(G, x)) - {x}


def ancestors(G, x):
    """
    Set of all ancestors of node in a graph, not including itself.
    :param G: target graph
    :param x: target node
    :return: set of ancestors
    """
    G_reversed = G.reverse()
    return descendants(G_reversed, x)


def are_equal_graphs(G1, G2):
    """
    Check graph equality (equal node names, and equal edges between them).
    :param G1: first graph
    :param G2: second graph
    :return: are they equal
    """
    if set(G1.nodes()) != set(G2.nodes()):
        return False
    return all(map(lambda x: G1.has_edge(*x), G2.edges())) and all(map(lambda x: G2.has_edge(*x), G1.edges()))


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
    def __init__(self, *args, **kwargs):
        super(DGM, self).__init__()
        if len(args) == 1:
            self.add_nodes_from(args[0].nodes(data=True))
            self.add_edges_from(args[0].edges(data=True))

        if 'value_mapping' in kwargs:
            self.value_mapping = kwargs['value_mapping']
        else:
            self.value_mapping = None

    def add_argument(self, argument):
        for factor in self.factors:
            factor.add_argument(argument, copy=False)
        return self

    def fit(self, data, *args, **kwargs):
        value_mapping = dataframe_value_mapping(data)
        kwargs.update(value_mapping=value_mapping, already_transformed=True)
        data = encode_dataframe(data, value_mapping)
        for node, node_data in self.nodes(data=True):
            if 'cpd' not in node_data:
                raise Exception('cpd not specified for node %s' % str(node))
            elif isinstance(node_data['cpd'], type):
                node_data['cpd'] = node_data['cpd'](self.nodes(), ([node] + self.predecessors(node)))
            node_data['cpd'].fit(data, *args, **kwargs)
            node_data['cpd'].normalize(node, copy=False)
        return self

    def rvs(self, size=1):
        result = pd.DataFrame(data={}, index=list(range(size)))
        for node in nx.topological_sort(self):
            cpd = self.cpd(node)
            result[node] = cpd.rvs(size=size, observed=result)[node]
        return result

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

    def values(self, var):
        return list(self.node[var]['cpd'].value_mapping[var].keys())

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

        value_mapping = {key: {value: i for i, value in enumerate(lst)} for key, lst in values.items()}

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
                        current_args = (i,) + tuple(values[var].index(arg) for var, arg in zip(parents, args)) \
                                       + (0,) * (len(arguments) - len(scope))
                        factor.table[current_args] = p
                factor.table = np.transpose(factor.table, axes=axes_order)

            factor.value_mapping = value_mapping
            result.node[node]['cpd'] = factor

        result.value_mapping = value_mapping
        return result

    @property
    def is_moral(self):
        """
        A graph is moral if it has no immoralities.
        :param self: target graph
        :return: is target graph moral
        """
        return len(list(self.immoralities)) == 0

    @property
    def immoralities(G):
        """
        Iterate over all immoralities in a graph.
        :param G: target graph
        :return: iterator over immoralities in form (node, parent1, parent2)
        """
        return filter(lambda v: (not G.has_edge(v[1], v[2])) and (not G.has_edge(v[2], v[1])), G.v_structures)

    @property
    def v_structures(G):
        """
        Iterate over all v-structures in a graph.
        :param G: target graph
        :return: iterator over v-structures in form (node, parent1, parent2)
        """
        for x in G.nodes():
            for p1, p2 in combinations(G.predecessors(x), r=2):
                yield x, p1, p2

    def reachable(self, source, observed):
        """
        Finds a set of reachable (in the sense of d-separation) nodes in graph.
        :param self: target graph
        :param source: source node name
        :param observed: a sequence of observed nodes
        :return: a set of reachable nodes
        """
        V = nx.number_of_nodes(self)
        A = set(sum([list(nx.dfs_preorder_nodes(self.reverse(), z)) for z in observed], []))
        Z = observed
        L = [(source, 'up')]
        V = set()
        result = set()
        while len(L) > 0:
            x, d = L.pop()
            if (x, d) in V:
                continue
            if x not in Z:
                result.add((x, d))
            V.add((x, d))
            if d == 'up' and x not in Z:
                for y in self.predecessors_iter(x):
                    L.append((y, 'up'))
                for y in self.successors_iter(x):
                    L.append((y, 'down'))
            elif d == 'down':
                if x in A:
                    for y in self.predecessors_iter(x):
                        L.append((y, 'up'))
                if x not in Z:
                    for y in self.successors_iter(x):
                        L.append((y, 'down'))
        result = set([x[0] for x in result])
        return result - {source}

    def is_covered(self, edge):
        """
        Checks if the edge is covered.
        :param self: graph
        :param edge: edge to check
        :return: is edge covered
        """
        x, y = edge
        return set(self.predecessors_iter(y)) - set(self.predecessors_iter(x)) == {x}

    def is_I_map(self, other):
        """
        Is self an I-map of other?
        """
        for x in self.nodes():
            pa = set(self.predecessors(x))
            non_descendants = set(self.nodes()) - descendants(self, x) - {x}
            if set.intersection(other.reachable(x, pa), non_descendants - pa):
                return False
        return True

    def is_I_equivalent(self, other):
        """
        Check if two graphs are I-equivalent
        :param self: first graph
        :param other: second graph
        :return: are the graphs I-equivalent
        """

        # same undirected skeleton
        if not are_equal_graphs(self.to_undirected(), other.to_undirected()):
            return False

        # same immoralities
        for x in self.nodes():
            for p1, p2 in set.union(set(combinations(self.predecessors(x), r=2)),
                                    set(combinations(other.predecessors(x), r=2))):
                if self.has_edge(p1, p2) or self.has_edge(p2, p1):
                    continue
                if other.has_edge(p1, x) and other.has_edge(p2, x):
                    continue
                return False

        # everything OK
        return True
