#coding=utf-8
__author__ = 'Alex Single'

from main import main
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
import scipy.stats

def func(x, a, b, c):
    return a*np.exp(b*x) + c

y = []
x = np.linspace(1, 15, 14)
x_graph = np.linspace(1, 15, 1000)
for n in xrange(1, 15):
    y.append(main(n, sample=1000, plot_flag=(n == 6)))

popt, pcov = curve_fit(func, x, y)
ss_res = np.dot((y - func(x, *popt)),(y - func(x, *popt)))
ss_tot = np.dot((y - np.mean(y)), (y - np.mean(y)))
R_squared = 1-ss_res/ss_tot

plt.plot(x, y, 'ko')
plt.plot(x_graph, func(x_graph, *popt), '-k')
patch = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=r'$R^2={0}$'.format(R_squared))
plt.legend(handles=[patch])
plt.xlabel(r'$N$', fontsize=18)
plt.ylabel(r'$eff$', fontsize=18)
plt.show()

print str(popt[0])+'*log(' + str(popt[1]) + ') + ' + str(popt[2])
print R_squared