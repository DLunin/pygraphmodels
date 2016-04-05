from itertools import permutations
import networkx as nx
from .matrix_graph import MatrixGraph
from graphmodels import DGM


class InvalidOperation(Exception):
    pass


class LocalOperation:
    def __init__(self, graph, fscore):
        self.graph = graph
        self.fscore = fscore

    def do(self):
        raise NotImplementedError()

    def undo(self):
        raise NotImplementedError()

    def score(self, **kwargs):
        raise NotImplementedError()


class AddEdge(LocalOperation):
    def __init__(self, graph, fscore, src, dst):
        LocalOperation.__init__(self, graph, fscore)
        self.src = src
        self.dst = dst

    def do(self):
        if self.graph.adj[self.src, self.dst]:
            raise InvalidOperation()
        self.graph.adj[self.src, self.dst] = 1
        if not self.graph.is_acyclic():
            self.graph.adj[self.src, self.dst] = 0
            raise InvalidOperation()
        return self

    def undo(self):
        self.graph.adj[self.src, self.dst] = 0
        return self

    def score(self, **kwargs):
        pa = self.graph.adj[:, self.dst].copy()
        score = -self.fscore(self.dst, pa, **kwargs)
        pa[self.src] = 1
        score += self.fscore(self.dst, pa, **kwargs)
        return score


class RemoveEdge(LocalOperation):
    def __init__(self, graph, fscore, src, dst):
        LocalOperation.__init__(self, graph, fscore)
        self.src = src
        self.dst = dst

    def do(self):
        if not self.graph.adj[self.src, self.dst]:
            raise InvalidOperation()
        self.graph.adj[self.src, self.dst] = 0
        return self

    def undo(self):
        self.graph.adj[self.src, self.dst] = 1
        return self

    def score(self, **kwargs):
        pa = self.graph.adj[:, self.dst].copy()
        score = -self.fscore(self.dst, pa, **kwargs)
        pa[self.src] = 0
        score += self.fscore(self.dst, pa, **kwargs)
        return score


class ReverseEdge(LocalOperation):
    def __init__(self, graph, fscore, src, dst):
        LocalOperation.__init__(self, graph, fscore)
        self.src = src
        self.dst = dst

    def do(self):
        if not self.graph.adj[self.src, self.dst]:
            raise InvalidOperation()
        self.graph.adj[self.src, self.dst] = 0
        self.graph.adj[self.dst, self.src] = 1
        if not self.graph.is_acyclic():
            self.graph.adj[self.src, self.dst] = 1
            self.graph.adj[self.dst, self.src] = 0
            raise InvalidOperation()
        return self

    def undo(self):
        self.graph.adj[self.src, self.dst] = 1
        self.graph.adj[self.dst, self.src] = 0
        return self

    def score(self, **kwargs):
        pa = self.graph.adj[:, self.dst].copy()
        score = -self.fscore(self.dst, pa, **kwargs)
        pa[self.src] = 0
        score += self.fscore(self.dst, pa, **kwargs)

        pa = self.graph.adj[:, self.src].copy()
        score -= self.fscore(self.src, pa, **kwargs)
        pa[self.dst] = 1
        score += self.fscore(self.src, pa, **kwargs)
        return score


class GreedySearch:
    def __init__(self, data, cls_score):
        graph = nx.DiGraph()
        graph.add_nodes_from(data.columns)
        graph = MatrixGraph.from_networkx_DiGraph(graph, order=data.columns)
        self.graph = graph
        self.fscore = cls_score(graph, data)

        self.ops = []
        self.ops += [AddEdge(graph, self.fscore, u, v) for u, v in permutations(graph.nodes(), 2)]
        self.ops += [RemoveEdge(graph, self.fscore, u, v) for u, v in permutations(graph.nodes(), 2)]
        self.ops += [ReverseEdge(graph, self.fscore, u, v) for u, v in permutations(graph.nodes(), 2)]

    def iteration(self):
        [op.score() for op in self.ops]
        self.ops.sort(reverse=True, key=lambda op: op.score())
        for op in self.ops:
            if op.score() <= 1e-5:
                return True
            try:
                op.do()
                op.score()
                return False
            except InvalidOperation:
                pass
        return True

    def __call__(self, max_iter=40, verbose=True):
        counter = 0
        while not self.iteration() and counter < max_iter:
            if verbose:
                print(self.fscore.total())
            counter += 1
        return DGM(self.graph.to_networkx_DiGraph())