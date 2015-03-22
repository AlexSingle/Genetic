#coding=utf-8
__author__ = 'Alex Single'

import numpy as np
import scipy.spatial.distance as sc_dist
import scipy.misc as sc_misc

class CachedDict:
    """
    CachedDict(fn) - an instance which acts like fn but memorizes its arguments
    Will only work on functions with non-mutable arguments
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}
    def __call__(self, *args):
        if not self.memo.has_key(args):
            self.memo[args] = self.fn(*args)
        return self.memo[args]

def flag(n):
    """
    Кумулятивная сумма числа сочетаний k из n
    """
    return np.cumsum([int(sc_misc.comb(n, k)) for k in xrange(n+1)])


def sorted_ones(n):
    """
    Генерируется массив инт. длины 2**n. После сортирутся по сумме бит (0111 > 1000).
    На выходе два массива с целыми и двоичными числами.
    """
    base = '0{0}b'.format(n)
    arr = np.array([[format(a, base)] for a in xrange(2**n)])
    arr_len = np.array([one[0].count('1') for one in arr])
    sort_ones = arr[np.argsort(arr_len, kind='mergesort')]
    arr_sorted_int = [int(one[0],2) for one in sort_ones]
    arr_sorted = [list(a[0]) for a in sort_ones]
    return arr_sorted_int, arr_sorted

def generate_ones(n):
    """
    Гененрируется словарь, где для каждой точки указаны родители предыдущей категории
    Для 0 точки указаны все потомки. Это необходимо для того, чтобы не терять связь 0 и 1 категории точек
    generate_ones(4)
    {0: [1, 2, 4, 8], 1: [], 2: [], 3: [1, 2], 4: [], 5: [1, 4], 6: [2, 4], 7: [3, 5, 6], 8: [],
    9: [1, 8], 10: [2, 8], 11: [3, 9, 10], 12: [4, 8], 13: [5, 9, 12], 14: [6, 10, 12], 15: [7, 11, 13, 14]}
    """
    flags = flag(n)
    arr_sorted_int,arr_sorted = sorted_ones(n)
    d = {one: [] for one in arr_sorted_int}
    d[0] = [2**x for x in xrange(n)]
    for k in xrange(n-1):
        ham_dist = sc_dist.cdist(arr_sorted[flags[k]:flags[k+1]], arr_sorted[flags[k+1]:flags[k+2]], 'hamming')*n
        near_points = np.argwhere(ham_dist == 1).T
        for r, one in enumerate(near_points[1]):
            d[arr_sorted_int[one+flags[k+1]]].append(arr_sorted_int[near_points[0, r]+flags[k]])
    return d

def dict_with_memory(n):
    cahced_dict = CachedDict(generate_ones)
    return cahced_dict(n)
