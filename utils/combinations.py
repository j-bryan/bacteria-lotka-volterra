import numpy as np
from math import factorial


def comb_wr_helper(chosen, arr, final, index, r, start, end):
    if index == r:
        tmp = []
        for i in range(r):
            tmp.append(arr[chosen[i]])
        final.append(tmp)
        return

    for i in range(start, end + 1):
        chosen[index] = i
        comb_wr_helper(chosen, arr, final, index + 1, r, i, end)

    return


def comb_wr(n, r):
    arr = np.arange(n)
    chosen = np.empty(r + 1, dtype=int)
    final = []
    comb_wr_helper(chosen, arr, final, 0, r, 0, len(arr) - 1)
    return np.array(final)


def num_comb_wr(n, r):
    return int(factorial(r + n - 1) / (factorial(r) * factorial(n - 1)))
