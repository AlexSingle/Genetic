#coding=utf-8
__author__ = 'Alex Single'

import numpy as np

n = 2

arr = [2**x for x in xrange(n)]
arr.insert(0,0)
prob = [0.1/(len(arr)-1) for x in xrange(n)]
prob.insert(0,0.9)
print prob
err = np.random.choice(arr, 2**n, p=prob)
print err
gen_err = []
rnd_err = []
for k, element in enumerate(err):
    if element != 0:
        gen_err.append(test_points[(genetic_position[k ^ element]),genetic_position[k]])
        rnd_err.append(test_points[(random_position[k ^ element]),random_position[k]])
print np.mean(gen_err), np.mean(rnd_err)