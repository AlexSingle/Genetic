#coding=utf-8
__author__ = 'Alex Single'

import time
from plot_result import plot
from utils import distance_dict as dd, goodness_overall as go
from generate_points import points as p
from genetic_points.coding import genetic_coding as g_coding
from random_points.coding import random_coding as r_coding


def main(n, sample, dim=2, test=None, plot_flag=None):
    if test:
        d = dd.dict_with_memory(4)
        filed_of_points = p.Field(4)
        test_points, plot_points = filed_of_points.test()
    else:
        d = dd.dict_with_memory(n)
        filed_of_points = p.Field(n, dim)
        test_points, plot_points = filed_of_points.default()
    result, result_bytes, goodness_genetic, genetic_position = g_coding(test_points, d)
    goodness_random, random_position = r_coding(test_points, d)
    if plot_flag:
        plot.plot_results(genetic_position, random_position, plot_points, test_points, goodness_genetic, sample, d)
    return go.goodness_overall(test_points, goodness_genetic, d, sample)[0]

start = time.time()
#main(3, 10, dim=2)
end = time.time()
print end - start
