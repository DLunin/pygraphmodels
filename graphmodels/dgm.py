from .output import pretty_draw
import networkx as nx


class DGM(nx.DiGraph):
    def __init__(self):
        super(DGM, self).__init__()

    def fit(self, data, *args, **kwargs):
        for node, node_data in self.nodes(data=True):
            if 'cpd' not in node_data:
                raise Exception('cpd not specified for node %s' % str(node))
            elif isinstance(node_data['cpd'], type):
                node_data['cpd'] = node_data['cpd'](self.nodes(), ([node] + self.predecessors(node)))
            node_data['cpd'].fit(data, *args, **kwargs)
            node_data['cpd'].normalize(node, copy=False)

    @property
    def factors(self):
        for node, node_data in self.nodes(data=True):
            yield node_data['cpd']

    def cpd(self, var):
        return self.node[var]['cpd']

    def draw(self):
        return pretty_draw(self)
