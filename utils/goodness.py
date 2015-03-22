#coding=utf-8
__author__ = 'Alex Single'

import numpy as np


def n_power(points_distance):
    return int(np.log2(points_distance.shape[0]))


def goodness(points_distance, d, posiitons):
    n = n_power(points_distance)
    return sum([sum(points_distance[posiitons[key], posiitons[d[key]]]) for key in d])/(n*2**(n-1))
