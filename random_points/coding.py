#coding=utf-8
__author__ = 'Alex Single'

from utils import goodness as g
import numpy as np
import cProfile

#@profile
def random_coding(points_distance, d):
    n = g.n_power(points_distance)
    random_position = np.random.permutation(2**n)
    goodness_random = g.goodness(points_distance, d, random_position)
    return goodness_random, random_position
