import math


def _single_taylor(x, n):
    res = math.pow(-1, n) * math.pow(math.pi / 2, 2 * n)
    res /= math.factorial(2 * n) * (4 * n + 1)
    return res * math.pow(x, 4 * n + 1)


def _multiplicator(x, n):
    res = math.pow(math.pi, 2) * math.pow(x, 4) * (4 * n + 1)
    res /= 4 * (4 * n + 5) * (2 * n + 1) * (2 * n + 2)
    return -res


def taylor(x, e, stop=False):
    a = _single_taylor(x, 0)
    n = 0
    s = a
    while abs(a) >= e:
        a *= _multiplicator(x, n)
        n += 1
        s += a
    if stop:
        return s, n
    return s
