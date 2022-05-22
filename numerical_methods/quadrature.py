import numpy as np


def central(a: float, b: float, count: int) -> float:
    x_first = np.linspace(a, b, count)
    x_second = x_first[1:]
    x_first = x_first[:-1]
    x = (x_second + x_first) / 2
    return sum(np.cos(np.pi * x ** 2 / 2) * (x_second - x_first))  # подынтегральная функция


def left(a: float, b: float, count: int) -> float:
    x_first = np.linspace(a, b, count)
    x_second = x_first[1:]
    x_first = x_first[:-1]
    return sum(np.cos(np.pi * x_first ** 2 / 2) * (x_second - x_first))  # подынтегральная функция


def trapezoid(a: float, b: float, count: int) -> float:
    x_first = np.linspace(a, b, count)
    x_second = x_first[1:]
    x_first = x_first[:-1]
    return sum((np.cos(np.pi * x_second ** 2 / 2) + np.cos(np.pi * x_first ** 2 / 2)) / 2 * (x_second - x_first))  # подынтегральная функция
