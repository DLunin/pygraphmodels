from ..dgm import DGM
from ..factor import TableFactor
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations, repeat


def discrete_mutual_information(x, y):
    def make_factor(data, arguments):
        factor = TableFactor(arguments, list(data.columns))
        factor.fit(data)
        return factor

    xy = pd.concat([x, y], axis=1)
    arguments = list(xy.columns)

    factor_x = make_factor(x, arguments)
    factor_y = make_factor(y, arguments)
    factor_xy = make_factor(xy, arguments).normalize(*arguments, copy=False)

    part1 = factor_xy.table.flatten()
    part2 = (factor_xy / (factor_x * factor_y).normalize(*arguments, copy=False)).table.flatten()

    result = np.sum(part1 * np.log(part2))
    if np.isnan(result):
        return +np.inf
    return result


def chow_liu(data, mi_estimator=discrete_mutual_information):
    arguments = list(data.columns)
    g = nx.Graph()
    g.add_nodes_from(arguments)
    for src, dst in combinations(arguments, 2):
        g.add_edge(src, dst, weight=-mi_estimator(data[[src]], data[[dst]]))
    return DGM(nx.dfs_tree(nx.minimum_spanning_tree(g), arguments[0]))
