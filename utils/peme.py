from collections import OrderedDict

import numpy as np


# Performance-Parameter-Efficiency
def ppe(score, r):
    r = r / 100
    return score * np.exp(-np.log10(r + 1))


# Performance-Memory-Efficiency
def pme(score, m, ft_mem):
    mr = m / ft_mem
    return score * np.exp(-np.log10(mr + 1))
