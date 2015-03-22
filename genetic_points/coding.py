#coding=utf-8
__author__ = 'Alex Single'

from utils import distance_dict as dd, goodness
import numpy as np
import cProfile

#@profile
def genetic_coding(points_distance, d):
    """
    Собственно генетическое кодирование массива точек.
    На вход подается массив точек, на выходе массив кодовых комбинаций в двоичном и десятичном виде
    """
    n = goodness.n_power(points_distance) # 2^n - количество точек
    flags = dd.flag(n) # кумулятивная сумма сочетаний всех комбинаций из n
    arr_sorted_int, _ = dd.sorted_ones(n) # точки отсортированные по категориям. внутри каждой категории по возрастанию.
    genetic_position = np.zeros(2**n, dtype=np.int32)
    step_zero = np.argmin(sum(points_distance))
    genetic_position[0] = step_zero
    genetic_position[arr_sorted_int[flags[0]:flags[1]]] = np.argsort(points_distance[step_zero])[1:n+1]
    for i in xrange(n-1):
        distance_next = np.array([sum(points_distance[genetic_position[d[key]]]) for key in arr_sorted_int[flags[i+1]:flags[i+2]]])
        new_points = []
        #print [d[key] for key in arr_sorted_int[sum(flags[:i+2]):sum(flags[:i+3])]]
        #print [genetic_position_back[d[key]] for key in arr_sorted_int[sum(flags[:i+2]):sum(flags[:i+3])]]
        distance_next[:, genetic_position[arr_sorted_int[:flags[i+1]]]] = np.inf
        min_row = np.min(distance_next, axis=1)
        min_row_arg = np.argmin(distance_next, axis=1)
        old_point = np.argmin(min_row)
        new_point = min_row_arg[old_point]
        for _ in xrange(distance_next.shape[0]):
            genetic_position[arr_sorted_int[old_point+flags[i+1]]] = new_point
            distance_next[old_point, :] = np.inf
            distance_next[:, new_point] = np.inf
            min_row[old_point] = np.inf
            new_points.append(new_point)
            old_old_point = np.argmin(min_row)
            while (min_row_arg[old_old_point] in new_points) and len(new_points) < (distance_next.shape[0]):
                min_row[old_old_point] = np.min(distance_next[old_old_point, :])
                min_row_arg[old_old_point] = np.argmin(distance_next[old_old_point, :])
                old_old_point = np.argmin(min_row)
            old_point = old_old_point
            new_point = min_row_arg[old_point]
    result = np.argsort(genetic_position)
    goodness_genetic = goodness.goodness(points_distance, d, genetic_position)
    base = '{0:0' + str(n) + 'b}'
    return result, [base.format(item) for item in result], goodness_genetic, genetic_position