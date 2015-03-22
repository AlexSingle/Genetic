#coding=utf-8
__author__ = 'Alex Single'

import numpy as np
import scipy.spatial.distance as sc_dist

class Field(object):
    def __init__(self, n, dim=2):
        self.size = 2**n
        self.dim = dim
        self.n = n
        #np.random.seed(100)

    def points_distance(self, points):
        return sc_dist.squareform(sc_dist.pdist(points))

    def default(self):
        points = np.random.randn(self.size, self.dim)
        return self.points_distance(points), points

    def test(self):
        points = np.array([[-0.503197961, 0.010541982, 0.308543183, 0.506703083,
                             1.145723424, 0.557661533, -0.618785329, 1.045596663,
                             0.165573193, -0.847130537, 2.253579054, -1.336342554,
                            -0.648908105, -1.980705207, 0.753960391, 1.522945569],
                            [0.53855964, 0.468148519, 0.280514511, 0.448179652,
                             0.146726334, -0.373253851, -0.229981735, 0.568031241,
                            -2.094664344, 0.719403589, 0.393574594, 0.178656057,
                            -0.635632611, 2.170616666, 1.40672544, 0.244241029]]).T
        return self.points_distance(points), points

    def grid_2d(self):
        points = np.array([[x, y] for x in [a for a in xrange(2**(self.n/2))] for y in [b for b in xrange(2**(self.n/2))]]).reshape(self.size,2)
        return self.points_distance(points), points

    def grid_3d(self):
        points = np.array([[x, y, z] for x in [a for a in xrange(2**(self.n/3))] for y in [b for b in xrange(2**(self.n/3))] for z in [c for c in xrange(2**(self.n/3))]]).reshape(self.size,3)
        return self.points_distance(points), points

    def cauchy(self):
        alpha = np.random.uniform(0,np.pi/2,size=self.size)
        ar = np.random.standard_cauchy(size=self.size)
        points = np.dstack((np.multiply(ar,np.cos(alpha)),np.multiply(ar,np.sin(alpha))))[0]
        return self.points_distance(points), points

    def uniform(self):
        points = np.dstack((np.random.uniform(size=self.size),np.random.uniform(size=self.size)))[0]
        return self.points_distance(points), points

    def two_disk(self):
        third = np.random.normal(0, 1, self.size)
        third[0:self.size/2] += 5
        points = np.dstack((np.random.normal(0, 1, size=self.size),np.random.normal(0,1,size=self.size),third))[0]
        return self.points_distance(points), points

    def square_normal(self):
        points = np.random.randn(self.size, 2)
        points[0:self.size/4][:, 0] += 1000
        points[self.size/4:(self.size/4)*2][:, 0] += 1000
        points[self.size/4:(self.size/4)*2][:, 1] += 100
        points[(self.size/4)*2:(self.size/4)*3][:, 1] += 100
        return self.points_distance(points), points
