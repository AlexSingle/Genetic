#coding=utf-8
__author__ = 'Alex Single'

import numpy as np
from random_points.coding import random_coding as r_coding

def goodness_overall(points, goodness_genetic, d, sample):
    goodnes_list = [r_coding(points, d)[0] for _ in xrange(sample)]
    return (np.mean(goodnes_list) - goodness_genetic)/np.std(goodnes_list), goodnes_list