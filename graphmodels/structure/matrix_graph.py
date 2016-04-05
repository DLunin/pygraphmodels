import numpy as np
from itertools import product
import networkx as nx
from graphmodels import DGM


class MatrixGraph:
    def __init__(self, adjacency_matrix, names_to_idx=None):
        self.adj = np.asarray(adjacency_matrix, dtype=bool)
        self.names_to_idx = names_to_idx
        if names_to_idx is None:
            self.names_to_idx = {i:i for i in range(self.adj.shape[0])}

    @property
    def n(self):
        return self.adj.shape[0]

    @property
    def m(self):
        return np.sum(self.adj)

    @property
    def names(self):
        idx_to_names = self.idx_to_names
        return [idx_to_names[i] for i in range(self.n)]

    def nodes(self):
        return list(range(self.n))

    @property
    def idx_to_names(self):
        return {i:node for node, i in self.names_to_idx.items()}

    @staticmethod
    def from_networkx_DiGraph(graph, order=None):
        if order is None:
            order = graph.nodes()
        names_to_idx = {node:i for i, node in enumerate(order)}
        adj = np.zeros((len(names_to_idx), len(names_to_idx)))
        for u, v in graph.edges():
            adj[names_to_idx[u], names_to_idx[v]] = 1
        return MatrixGraph(adj, names_to_idx=names_to_idx)

    def to_networkx_DiGraph(self):
        result = nx.DiGraph()
        result.add_nodes_from(self.names_to_idx.keys())
        for i, j in product(range(self.n), repeat=2):
            if self.adj[i, j]:
                result.add_edge(self.idx_to_names[i], self.idx_to_names[j])
        return result

    def is_acyclic(self):
        return nx.is_directed_acyclic_graph(self.to_networkx_DiGraph())

    def draw(self):
        return DGM(self.to_networkx_DiGraph()).draw()