#coding=utf-8
__author__ = 'Alex Single'

import numpy as np
import matplotlib.pyplot as plt
import time
from utils import distance_dict as dd, goodness as g, goodness_overall as go
from generate_points import points as p
from genetic_points.coding import genetic_coding as g_coding
from random_points.coding import random_coding as r_coding



def main():
    n = 14
    cahced_dict = dd.CachedDict(dd.generate_ones)
    d = cahced_dict(n)
    filed_of_points = p.Field(n)
    test_points, plot_points = filed_of_points.default()

    result, result_bytes, goodness_genetic, genetic_position = g_coding(test_points, d)
    goodness_random, random_position = r_coding(test_points, d)
    #olo = go.goodness_overall(test_points, d, goodness_genetic, 100)
    print goodness_genetic, goodness_random
#plot_lines(genetic_position, plot_points, d)
#plot_results(genetic_position, random_position, plot_points, test_points, d)
#plt.show()
#arr = [2**x for x in xrange(n)]
#arr.insert(0,0)
#prob = [0.1/(len(arr)-1) for x in xrange(n)]
#prob.insert(0,0.9)
#err = np.random.choice(arr, 2**n, p=prob)
#gen_err = []
#rnd_err = []
#for k, element in enumerate(err):
#    if element != 0:
#        gen_err.append(test_points[(genetic_position[k ^ element]),genetic_position[k]])
#        rnd_err.append(test_points[(random_position[k ^ element]),random_position[k]])
#print np.mean(gen_err), np.mean(rnd_err)

start = time.time()
main()
end = time.time()
print end - start
