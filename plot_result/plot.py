#coding=utf-8
__author__ = 'Alex Single'

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import goodness_overall as go
import numpy as np


def plot_lines(positions, points, d, num, fig):
    if fig:
        ax = fig.add_subplot(num, projection='3d')
        [ax.plot([points[positions[key],0],points[positions[one]][0]],[points[positions[key],1],points[positions[one]][1]], [points[positions[key],2],points[positions[one]][2]], '-ko') for key in d for one in d[key]]
    else:
        [plt.plot([points[positions[key],0],points[positions[one]][0]],[points[positions[key],1],points[positions[one]][1]], '-ko') for key in d for one in d[key]]

def plot_results(genetic_position, random_position, plot_points, test_points, goodness_genetic, sample, d):
    if plot_points.shape[1] == 3:
        fig = plt.figure()
    elif plot_points.shape[1] != 2:
        return None
    else:
        fig = None
    plt.subplot(221)
    plt.title('Genetic')
    plt.axis([-1, 4, -1, 4])
    plot_lines(genetic_position, plot_points, d, 221, fig)
    plt.subplot(222)
    plt.title('Random')
    plt.axis([-1, 4, -1, 4])
    plot_lines(random_position, plot_points, d, 222, fig)
    plt.subplot(212)
    overall = go.goodness_overall(test_points, goodness_genetic, d, sample)
    plt.hist(overall[1], bins=20, color='0.75')
    plt.xlabel(r'$\mu_g={0}\ \mu_r={1}\ \sigma_r={2}$'.format(goodness_genetic, np.mean(overall[1]), np.std(overall[1])), fontsize=14)
    plt.show()
    #plt.axvline(goodness_genetic,color='r')
