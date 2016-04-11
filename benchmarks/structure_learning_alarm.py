import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os.path
from line_profiler import LineProfiler, show_func

import sys
sys.path.append('/home/wrwt/Programming/pygraphmodels')
import graphmodels as gm
from graphmodels import MatrixGraph, DGM

import cProfile, pstats
from StringIO import StringIO

NETWORKS_PATH = '/home/wrwt/Programming/pygraphmodels/networks/'
true_dgm = gm.DGM.read(os.path.join(NETWORKS_PATH, 'alarm.bif'))
data = true_dgm.rvs(size=1000)


def main():
    gs = gm.GreedySearch(data, gm.ScoreBIC)
    res = gs(max_iter=5)

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

with PyCallGraph(output=GraphvizOutput()):
    main()
