from ..dgm import DGM
import networkx as nx
from itertools import combinations, repeat
from ..information import discrete_mutual_information


def chow_liu(data, mi_estimator=discrete_mutual_information):
    arguments = list(data.columns)
    g = nx.Graph()
    g.add_nodes_from(arguments)
    for src, dst in combinations(arguments, 2):
        g.add_edge(src, dst, weight=-mi_estimator(data[[src]], data[[dst]]))
    return DGM(nx.dfs_tree(nx.minimum_spanning_tree(g), arguments[0]))
