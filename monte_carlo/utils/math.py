import numpy as np


def top_k_indices(array, k):
    a = array.copy()
    return a.argsort(axis=0)[-k:]
