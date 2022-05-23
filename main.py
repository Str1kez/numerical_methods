import json
import math
import numpy as np
import plotly.express as px

from numerical_methods import taylor, l, central, left, trapezoid
from numerical_methods.quadrature import gauss


def drow_graph(x, y):
    fig = px.line(x=x, y=y)
    fig.update_layout(yaxis_title='erf(x)')
    fig.show()


def derivative_complex() -> None:
    x = np.linspace(a, b, 3 * n)  # 20 точек
    gt = np.cos(np.pi * x ** 2 / 2)  # подынтегральная функция
    data = {el: {} for el in x}
    for i in range(n, 50, 2):
        nodes = np.linspace(a, b, i)  # узлы разбиения
        f = [taylor(x, e) for x in nodes]  # Fn
        for point in x:  # расчет для фиксированных точек
            L = 0
            for k in range(i):
                L += f[k] * l(point, k, nodes)
            data[point][i] = L
        x_dev = np.array([data[x][i] for x in data])  # значения в фиксированных точках при разбиении i
        print(np.max(np.abs(x_dev - gt)), f'Узлов: {i}')  # Погрешность
    json.dump(data, open('.result.json', 'w'), indent=4)
    

def derivative_single() -> None:
    x = np.linspace(a, b, 3 * n)  # 30 точек
    gt = np.cos(np.pi * x ** 2 / 2)  # подынтегральная функция
    # узлы Чебышёва
    chebyshev_nodes = (a + b) / 2 + (b - a) / 2 * np.cos([np.pi * (2 * k - 1) / 2 / n for k in range(1, n + 1)])
    # * На узлаx Чебышёва погрешность 0.0006, на равноотдаленных 0.0016
    nodes = np.linspace(a, b, n)  # равноотдаленные узлы разбиения
    f = [taylor(x, e) for x in nodes]  # Fn
    for i, point in enumerate(x):  # расчет для фиксированных точек
        L = sum(f[k] * l(point, k, nodes) for k in range(n))
        print(point, gt[i], L, np.abs(L - gt[i]))


def tabulation() -> None:
    x = np.linspace(a, b, n)
    f = [(i, ) + taylor(i, e, True) for i in x]
    print(*f, sep='\n', end='\n----\n')
    # drow_graph(x, [p[0] for p in f])
    
    
if __name__ == '__main__':
    a = 0
    b = 1.5
    h = .15
    n = 10
    e = math.pow(10, -4)
    # tabulation()
    # derivative()
    # derivative_single()
    print(central(a, b, 1024))
    print(left(a, b, 1024))
    print(trapezoid(a, b))
    print(gauss(a, b))
    