import json
import math
import numpy as np
import plotly.express as px
import pandas as pd

from numerical_methods import taylor, l, central, left, trapezoid, gauss


def draw_graph(x, y, yaxis_title: str):
    fig = px.line(x=x, y=y)
    fig.update_layout(yaxis_title=yaxis_title)
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
    

def derivative_single_analysis() -> None:
    x = np.linspace(a, b, 3 * n)  # 30 точек
    gt = np.cos(np.pi * x ** 2 / 2)  # подынтегральная функция
    L_list = []  # Приближенные значения производной
    dev_list = []  # Погрешность
    # узлы Чебышёва
    nodes = (a + b) / 2 + (b - a) / 2 * np.cos([np.pi * (2 * k - 1) / 2 / n for k in range(1, n + 1)])
    # На узлаx Чебышёва погрешность 0.0006, на равноотдаленных 0.0016
    # nodes = np.linspace(a, b, n)  # равноотдаленные узлы разбиения
    f = [taylor(x, e) for x in nodes]  # Fn

    for i, point in enumerate(x):  # расчет для фиксированных точек
        L = sum(f[k] * l(point, k, nodes) for k in range(n))
        L_list.append(L)
        dev_list.append(np.abs(L - gt[i]))

    derivative_df = pd.DataFrame(data={
        "erf'(zi)=...": gt,
        "L'n(zi)": L_list,
        "Погрешность интерполяции ...": dev_list
    }, index=pd.Index(data=x, name='Точки zi. в которых вычисляется производная'))
    # derivative_df.to_csv('data/derivative.csv')
    derivative_df.to_csv('data/derivative_chebyshev.csv')

    draw_graph(x, gt, yaxis_title="С'(x)")


def tabulation_analysis() -> None:
    x = np.linspace(a, b, n)  # Равноотдаленные узлы
    f = [taylor(i, e, True) for i in x]

    tabulation_df = pd.DataFrame(data={
        'erf(xi)': [erf[0] for erf in f],
        'ni(eps)': [erf[1] for erf in f]
    }, index=pd.Index(data=x, name='xi'))
    tabulation_df.to_csv('data/tabulation.csv')

    draw_graph(x, [p[0] for p in f], 'С(xi)')


def quadrature_analysis(quad_func, count: int = None) -> None:
    x = np.linspace(a, b, n)  # Равноотдаленные узлы
    quad_f_list = []  # Значения квадратур
    erf_list = []  # Значения ф-ции ошибок
    dev_list = []  # Погрешность
    count_list = []  # Количество разбиений
    
    for i in x:
        erf_list.append(taylor(i, e))
        if count:
            quad_f_list.append(quad_func(a, i, count))
            count_list.append(count)
        else:
            quad_temp = quad_func(a, i)
            quad_f_list.append(quad_temp[0])
            count_list.append(quad_temp[1])
        dev_list.append(abs(erf_list[-1] - quad_f_list[-1]))

    quad_df = pd.DataFrame(data={
        'C(xi)': erf_list,
        'Значения по квадратуре': quad_f_list,
        'Погрешность вычислений': dev_list,
        'N - число разбиений': count_list
    }, index=pd.Index(data=x, name='xi'))
    quad_df.to_csv(f'data/{quad_func.__name__}.csv')


if __name__ == '__main__':
    a = 0
    b = 1.5
    h = .15
    n = 10
    e = math.pow(10, -6)
    tabulation_analysis()
    derivative_single_analysis()
    quadrature_analysis(gauss)
    quadrature_analysis(central)
    quadrature_analysis(left)
    quadrature_analysis(trapezoid)
