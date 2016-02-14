import numpy as np
from .multinomial_cpp import gen_multinomial


def multinomial(p, size=1):
    p = np.asarray(p)
    assert (1 <= p.ndim <= 2)
    if p.ndim == 1:
        p = np.array([p] * size)
    res = np.zeros(p.shape[0]).astype('int32')
    cs = np.cumsum(p, axis=1).astype('float64')
    gen_multinomial(res, cs)
    return res
