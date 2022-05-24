import json
import math
import numpy as np
import plotly.express as px
import pandas as pd
from typing import Iterable

from numerical_methods import taylor, l, central, left, trapezoid, gauss


def draw_graph(x: Iterable = None, y: Iterable = None, yaxis_title: str = None, data_frame: pd.DataFrame = None):
    fig = px.line(data_frame=data_frame) if data_frame is not None else px.line(x=x, y=y)
    if yaxis_title:
        fig.update_layout(yaxis_title=yaxis_title)
    fig.show()


def draw_complex_comparison() -> None:
    chebyshev = pd.read_csv('data/max_eps_chebyshev.csv')
    uniform = pd.read_csv('data/max_eps_uniform.csv')
    union_df = pd.merge(chebyshev, uniform)
    union_df.index = union_df.pop('Узлы')
    draw_graph(data_frame=union_df, yaxis_title='Max_n(eps)')
    

def draw_slow_comparison() -> None:
    comp_df = pd.read_csv('data/max_eps_uniform_slow.csv')
    comp_df.index = comp_df.pop('Узлы')
    draw_graph(data_frame=comp_df, yaxis_title='Max_n(eps)')


def derivative_complex() -> None:
    x = np.linspace(a, b, 2 * n)  # 20 точек
    gt = np.cos(np.pi * x ** 2 / 2)  # подынтегральная функция
    max_dev = []
    max_n = 30
    step = 1
    for i in range(n, max_n, step):
        nodes = np.linspace(a, b, i)  # узлы разбиения равноотдаленные
        # узлы Чебышёва
        # nodes = (a + b) / 2 + (b - a) / 2 * np.cos([np.pi * (2 * k - 1) / 2 / i for k in range(1, i + 1)])
        f = [taylor(x, e) for x in nodes]  # Fn
        x_dev = []
        for point in x:  # расчет для фиксированных точек
            L = sum(f[k] * l(point, k, nodes) for k in range(i))
            x_dev.append(L)
        x_dev = np.array(x_dev)  # значения в фиксированных точках при разбиении i
        max_dev.append(np.max(np.abs(x_dev - gt)))  # Погрешность
    derivative_df = pd.DataFrame(data={'Max(eps)': max_dev}, index=pd.Index(data=range(n, max_n, step), name='Узлы'))
    derivative_df.to_csv('data/max_eps_uniform_slow.csv')
    # draw_graph(data_frame=derivative_df, yaxis_title='Погрешность')
    

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

    draw_graph(x=x, y=gt, yaxis_title="С'(x)")


def tabulation_analysis() -> None:
    x = np.linspace(a, b, n)  # Равноотдаленные узлы
    f = [taylor(i, e, True) for i in x]

    tabulation_df = pd.DataFrame(data={
        'erf(xi)': [erf[0] for erf in f],
        'ni(eps)': [erf[1] for erf in f]
    }, index=pd.Index(data=x, name='xi'))
    tabulation_df.to_csv('data/tabulation.csv')

    draw_graph(x=x, y=[p[0] for p in f], yaxis_title='С(xi)')


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
    e = math.pow(10, -4)
    # tabulation_analyis()
    # derivative_complex()
    draw_complex_comparison()
    draw_slow_comparison()
    # derivative_single_analysis()
    # quadrature_analysis(gauss)
    # quadrature_analysis(central)
    # quadrature_analysis(left)
    # quadrature_analysis(trapezoid)
