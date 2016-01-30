from .output import pretty_draw
from .misc import constant
import networkx as nx
import pandas as pd


class ErdosRenyiDGMGen:
    def __init__(self, n=None, p=None, factor_gen=None):
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
