#coding=utf-8
__author__ = 'Alex Single'

from main import main
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches

def func(x, a, b, c):
    return a*np.log(b*x) + c

y = []
x = np.linspace(2, 99, 98)
for n in xrange(2, 100):
    if n == 600:
        y.append(main(n, 1000, dim=3, plot_flag=True))
    else:
        y.append(main(8, dim=n, sample=1000))


popt, pcov = curve_fit(func, x, y)
ss_res = np.dot((y - func(x, *popt)),(y - func(x, *popt)))
ss_tot = np.dot((y - np.mean(y)), (y - np.mean(y)))
R_squared = 1-ss_res/ss_tot

plt.plot(x, y, 'ko')
plt.plot(x, func(x, *popt), '-k')
patch = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=r'$R^2={0}$'.format(R_squared))
plt.legend(handles=[patch])
plt.show()