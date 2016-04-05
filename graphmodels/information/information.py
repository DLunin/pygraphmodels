from ..factor import TableFactor
import numpy as np
import pandas as pd
from itertools import combinations


def discrete_entropy(x):
    def make_factor(data, arguments, leak=1e-9):
        factor = TableFactor(arguments, list(data.columns))
        factor.fit(data)
        factor.table += leak
        factor.normalize(*factor.scope, copy=False)
        return factor
    arguments = list(x.columns)
    factor_x = make_factor(x, arguments).normalize(*arguments)
    prob = factor_x.table.flatten()
    return -np.sum(prob * np.log(prob))


def discrete_mutual_information(x, y):
    if x.shape[1] == 0 or y.shape[1] == 0:
        return 0

    def make_factor(data, arguments, leak=1e-9):
        factor = TableFactor(arguments, list(data.columns))
        factor.fit(data)
        factor.table += leak
        factor.normalize(*factor.scope, copy=False)
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


def information_matrix(data, mi_estimator=discrete_mutual_information):
    m = len(data.columns)
    values = data.values
    information_matrix = np.zeros((m, m))
    for (i, fst), (j, snd) in combinations(enumerate(data.columns), 2):
        information_matrix[i, j] = information_matrix[j, i] = mi_estimator(data[[fst]], data[[snd]])
    return pd.DataFrame(data=dict(zip(data.columns, information_matrix)), index=data.columns)
