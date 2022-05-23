import numpy as np


def epsilon_dependency(func):
    def wrapper(a: float, b: float):
        n = 2
        eps = 10 ** -6
        s_n = func(a, b, n)
        s_2_n = func(a, b, 2 * n)
        while abs(s_n - s_2_n) > eps:
            n *= 2
            s_n = func(a, b, n)
            s_2_n = func(a, b, 2 * n)
        return s_n, n
    return wrapper


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


@epsilon_dependency
def trapezoid(a: float, b: float, count: int = 2) -> float:
    x_first = np.linspace(a, b, count)
    x_second = x_first[1:]
    x_first = x_first[:-1]
    return sum((np.cos(np.pi * x_second ** 2 / 2) + np.cos(np.pi * x_first ** 2 / 2)) / 2 * (x_second - x_first))  # подынтегральная функция


@epsilon_dependency
def gauss(a: float, b: float, count: int = 2) -> float:
    x_first = np.linspace(a, b, count)
    h = (b - a) / count
    x_first = x_first[:-1]
    return sum((np.cos(np.pi * (x_first + h / 2 * (1 - 1 / np.sqrt(3))) ** 2 / 2)) + 
               (np.cos(np.pi * (x_first + h / 2 * (1 + 1 / np.sqrt(3))) ** 2 / 2))) * h / 2
