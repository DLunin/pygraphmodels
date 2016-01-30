import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import networkx as nx
import pystache

import mpld3
from mpld3 import plugins

import os
SCRIPT_DIR = os.path.dirname(__file__)


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
        self.append(names + ['Prob.'])
        for v in product(*list(map((lambda x: list(range(x))), table.shape))):
            self.append(list(v) + ["%0.3f" % table[v]])

    def _repr_html_(self):
        def to_rgb_str(rgba):
            return '#%02x%02x%02x' % tuple((np.asarray(rgba[:3]) * 255.).astype('int'))

        params = {
            'header': [{'text': text} for text in self[0]],
            'row': [
                {
                    'cell': [
                        {
                            'bgcolor': '#FFFFFF',
                            'text': cell,
                        }
                        for cell in row[:-1]
                    ] + [
                        {
                            'bgcolor': to_rgb_str(plt.cm.Pastel1(float(row[-1]))),
                            'text': row[-1],
                        }
                    ]
                }
                for row in self[1:]
            ]
        }
        with open(os.path.join(SCRIPT_DIR, 'table_template.txt'), 'r') as f:
            return str(pystache.render(f.read(), params))

def pretty_draw(g):
    css = """
table
{
  border-collapse: collapse;
}
th
{
  color: #ffffff;
  background-color: #000000;
}
td
{
  background-color: #cccccc;
}
table, th, td
{
  font-family:Arial, Helvetica, sans-serif;
  border: 1px solid black;
  text-align: center;
  padding: 3px;
  font-size:11pt;
}
g.mpld3-xaxis, g.mpld3-yaxis {
    display: none;
}
    """

    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()

    layout = nx.spring_layout(g, iterations=10)

    points = []
    labels = []
    for node, (x, y) in layout.items():
        points.append((x, y))
        try:
            try:
                labels.append(g.cpd(node)._repr_html_())
            except:
                labels.append(str(g.cpd(node)))
        except:
            pass
    points_x, points_y = zip(*points)

    ax.set_xlim(min(points_x) - 0.08, max(points_x) + 0.08)
    ax.set_ylim(min(points_y) - 0.08, max(points_y) + 0.08)

    for src, dst in g.edges():
        src_pos = layout[src]
        dst_pos = layout[dst]
        arr_pos = dst_pos - 0.15*(dst_pos - src_pos)

        ax.plot(*list(zip(src_pos, dst_pos)), c='grey')
        ax.plot(*list(zip(arr_pos, dst_pos)), c='black', alpha=.5, linewidth=5)

    ax.plot(points_x, points_y, 'o', color='lightgray',
                     mec='k', ms=20, mew=1, alpha=1.)

    for text, x, y in zip(g.nodes(), points_x, points_y):
        ax.text(x, y, text, horizontalalignment='center', verticalalignment='center')

    pts = ax.plot(points_x, points_y, 'o', color='lightgray',
                     mec='k', ms=40, mew=1, alpha=0.)

    tooltip = plugins.PointHTMLTooltip(pts[0], labels,
                                       voffset=10, hoffset=10, css=css)
    plugins.connect(fig, tooltip)

    return mpld3.display()

