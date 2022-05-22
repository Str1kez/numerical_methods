import numpy as np


def l(x: float, k: int, x_array: np.array):
    n = len(x_array)
    res = 0
    # prod_up = 1
    # prod_up = np.prod(x - x_array)
    for j in range(n):
        if j == k:
            continue
        prod_up = 1
        for i in range(n):
            if i == k or i == j:
                continue
            prod_up *= x - x_array[i]
        # prod_down = (x - x_array[k]) * (x - x_array[j])
        prod_down = np.prod(x_array[k] - np.concatenate((x_array[:k], x_array[k + 1:])))
        # prod_down *= mul
        res += prod_up / prod_down
    return res
