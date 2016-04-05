import numpy as np
from graphmodels.information import discrete_entropy


class EntropyEstimator:
    def __init__(self, graph, data):
        self.graph = graph
        self.data = data
        self.cache = {}

    def _footprint(self, nodes):
        return tuple(nodes)

    def __call__(self, nodes):
        fp = self._footprint(nodes)
        if fp in self.cache:
            return self.cache[fp]

        if np.any(nodes):
            subdata = self.data[self.data.columns[nodes]]
            result = discrete_entropy(subdata)
        else:
            result = 0

        self.cache[fp] = result
        return result


class InformationEstimator:
    def __init__(self, graph, data):
        self.graph = graph
        self.entropy_estimator = EntropyEstimator(graph, data)
        self.chosen = []

    def __call__(self, node, parents):
        parents = np.array(parents, dtype=bool)
        h1 = self.entropy_estimator(parents)
        temp = np.zeros(len(parents), dtype=bool)
        temp[node] = True
        h2 = self.entropy_estimator(temp)
        parents[node] = True
        h12 = self.entropy_estimator(parents)
        return h1 + h2 - h12


class ScoreBIC:
    def __init__(self, graph, data):
        self.graph = graph
        self.data = data
        self.n_values = np.asarray([len(self.data[column].value_counts()) for column in self.data.columns])
        self.mi_estimator = InformationEstimator(graph, data)

    def __call__(self, node, parents):
        parents = np.asarray(parents, dtype=bool)
        k = self.n_values[node]*np.prod(self.n_values[parents]) - 1
        n = self.data.shape[0]
        l = n*self.mi_estimator(node, parents)

        result = l - 0.5 * np.log(n) * k
        return result

    def total(self):
        score = 0.
        for node in self.graph.nodes():
            pa = self.graph.adj[:, node].copy()
            score += self(node, pa)
        return score