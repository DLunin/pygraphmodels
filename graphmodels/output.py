import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import networkx as nx


class ListTable(list):
    # from http://calebmadrigal.com/display-list-as-table-in-ipython-notebook/
    """
    Overridden list class which takes a 2-dimensional list of
    the form [[1,2,3],[4,5,6]], and renders an HTML Table in
    IPython Notebook.
    """

    def __init__(self, table, names):
        list.__init__(self)
        table = np.asarray(table)
        self.append(names + ['P'])
        for v in product(*list(map((lambda x: list(range(x))), table.shape))):
            self.append(list(v) + ["%0.3f" % table[v]])

    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")

            for col in row:
                html.append("<td>{0}</td>".format(col))

            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)


def pretty_draw(graph, node_color=lambda node, attr: '#DDDDDD',
                edge_color=lambda node1, node2, attr: '#000000', node_size=lambda node, attr: 300, highres=False):
    """
    Draws a graph. You can specify colors of nodes, colors of edges and size of nodes via lambda
    functions.
    :param graph: target graph
    :param node_color: lambda function mapping node name and its attributes to the desired color
    :param edge_color: lambda function mapping edge and its attributes to the desired color
    :param node_size: lambda function mapping node name and its attributes to the desired size
    :return: None
    """

    def extract_node_attribute(graph, name, default=None):
        """
        Extract attributes of a networx graph nodes to a dict.
        :param graph: target graph
        :param name: name of the attribute
        :param default: default value (used if node doesn't have the specified attribute)
        :return: a dict of attributes in form of { node_name : attribute }
        """
        return {i: d.get(name, default) for i, d in graph.nodes(data=True)}

    def extract_edge_attribute(graph, name, default=None):
        """
        Extract attributes of a networx graph edges to a dict.
        :param graph: target graph
        :param name: name of the attribute
        :param default: default value (used if edge doesn't have the specified attribute)
        :return: a dict of attributes in form of { (from, to) : attribute }
        """
        return {(i, j): d.get(name, default) for i, j, d in graph.edges(data=True)}

    if highres:
        fig = plt.figure(figsize=(100, 100))
    else:
        fig = plt.figure(figsize=(17, 6))

    plt.axis('off')
    if type(node_color) is str:
        node_colors = extract_node_attribute(graph, 'color', default='#DDDDDD')
        node_colors = list(map(node_colors.__getitem__, graph.nodes()))
    else:
        node_colors = list(map(lambda args: node_color(*args), graph.nodes(data=True)))

    if type(edge_color) is str:
        edge_colors = extract_edge_attribute(graph, 'color', default='#000000')
        edge_colors = list(map(edge_colors.__getitem__, graph.edges()))
    else:
        edge_colors = list(map(lambda args: edge_color(*args), graph.edges(data=True)))

    if type(node_size) is str:
        node_sizes = extract_node_attribute(graph, 'size', default='300')
        node_sizes = list(map(node_sizes.__getitem__, graph.nodes()))
    else:
        node_sizes = list(map(lambda args: node_size(*args), graph.nodes(data=True)))

    nx.draw_networkx(graph,
                     with_labels=True,
                     pos=nx.spring_layout(graph),
                     node_color=node_colors,
                     edge_color=edge_colors,
                     node_size=node_sizes
                     )
    return None
