import numpy as np
from copy import copy
import pandas as pd


class constant:
    def __init__(self, val):
        self.val = val

    def rvs(self, size=1):
        if size == 1:
            return copy(self.val)
        return np.asarray([copy(self.val) for _ in range(size)])


def invert_value_mapping(value_mapping):
    if value_mapping is None:
        return None
    result = {}
    for name, mapping in value_mapping.items():
        if isinstance(mapping, dict):
            result[name] = {val: key for key, val in mapping.items()}
    return result


def column_value_mapping(column):
    unique = pd.unique(column)
    return dict(zip(unique, range(len(unique))))


def dataframe_value_mapping(df):
    result = {}
    for name, column in zip(df.columns, df.values.T):
        result[name] = column_value_mapping(column)
    return result


def encode_dataframe(df, vm):
    return pd.DataFrame(data={column: df[column].map(vm[column].__getitem__) for column in df.columns})
