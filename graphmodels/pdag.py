import networkx as nx
from itertools import combinations_with_replacement
from .output import pretty_draw


class PDAG(nx.DiGraph):
    def __init__(self):
        nx.DiGraph.__init__(self)

    def has_unoriented_edge(self, x, y):
        return self.has_edge(x, y) and self.has_edge(y, x)

    def has_oriented_edge(self, x, y):
        return self.has_edge(x, y) and not self.has_edge(y, x)

    def has_any_edge(self, x, y):
        return self.has_edge(x, y) or self.has_edge(y, x)

    def orient_edge(self, x, y):
        self.remove_edge(y, x)

    def rule1(self, x, y, z):
        if self.has_oriented_edge(x, y) and self.has_unoriented_edge(y, z) and not self.has_any_edge(x, z):
            self.orient_edge(y, z)
            return True
        return False

    def rule2(self, x, y, z):
        if self.has_oriented_edge(x, y) and self.has_oriented_edge(y, z) and self.has_unoriented_edge(x, z):
            self.orient_edge(x, z)
            return True
        return False

    def rule3(self, x, y1, y2, z):
        if self.has_unoriented_edge(x, y1) and self.has_unoriented_edge(x, y2) and \
                self.has_unoriented_edge(x, z) and not self.has_any_edge(y1, y2) and \
                self.has_oriented_edge(y1, z) and self.has_oriented_edge(y2, z):
            self.orient_edge(x, z)
            return True
        return False

    def _rule3_partial_check1(self, y1, y2, z):
        if self.has_oriented_edge(y1, z) and self.has_oriented_edge(y2, z) and not self.has_any_edge(y1, y2):
            return True
        return False

    def _rule3_partial_check2(self, x, y1, y2, z):
        if self.has_unoriented_edge(x, y1) and self.has_unoriented_edge(x, y2) and self.has_unoriented_edge(x, z):
            self.orient_edge(x, z)
            return True
        return False

    def apply_rule(self):
        for x, y, z in combinations_with_replacement(self.nodes(), 3):
            if self.rule1(x, y, z):
                return True
            if self.rule2(x, y, z):
                return True
            if self._rule3_partial_check1(x, y, z):
                for w in self.nodes():
                    if self._rule3_partial_check2(w, x, y, z):
                        return True
        return False

    def update_directions(self):
        while self.apply_rule():
            pass

    @staticmethod
    def from_dgm(G):
        pdag = PDAG()
        pdag.add_nodes_from(G.nodes())
        for s, t in G.edges():
            pdag.add_edge(s, t)
            pdag.add_edge(t, s)
        pretty_draw(pdag)
        for x, p1, p2 in G.immoralities:
            if pdag.has_edge(x, p1):
                pdag.orient_edge(p1, x)
            if pdag.has_edge(x, p2):
                pdag.orient_edge(p2, x)
        pdag.update_directions()
        return pdag