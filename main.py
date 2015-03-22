#coding=utf-8
__author__ = 'Alex Single'

import numpy as np
import scipy.spatial.distance as sc_dist
import scipy.misc as sc_misc
import itertools
import matplotlib.pyplot as plt
import cProfile
import time
from mpl_toolkits.mplot3d import Axes3D

start = time.time()


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

def generate_points(n):
    points = np.random.randn(2**n,2)
    #points = np.array([[x, y] for x in [a for a in xrange(2**(n/2))] for y in [b for b in xrange(2**(n/2))]]).reshape(2**n,2)
    #points = np.array([[x, y, z] for x in [a for a in xrange(2**(n/3))] for y in [b for b in xrange(2**(n/3))] for z in [c for c in xrange(2**(n/3))]]).reshape(2**n,3)
    #points = np.array([[x+((y%3)/2*x/2.0), y+(1-y%2/2.0)] for x in [(3**(1/2.0)/2.0*a) for a in xrange(2**(n/2))] for y in [b/2.0+b/3 for b in xrange(2**(n/2))]]).reshape(2**n,2)
    #points =  [key, k for k in [x for x in xrange(4)] for key in [x for x in xrange(4)]]
    #third = np.random.normal(0,1,2**n)
    #third[0:2**n/2] += 5
    #points = np.dstack((np.random.normal(0,1,size=2**n),np.random.normal(0,1,size=2**n),third))[0]
    #points = np.dstack((np.random.uniform(size=2**n),np.random.uniform(size=2**n)))[0]
    #alpha = np.random.uniform(0,np.pi/2,size=2**n)
    #ar = np.random.standard_cauchy(size=2**n)
    #points = np.dstack((np.multiply(ar,np.cos(alpha)),np.multiply(ar,np.sin(alpha))))[0]
    #points[0:2**n/4][:,0] += 1000
    #points[2**n/4:(2**n/4)*2][:,0] += 1000
    #points[2**n/4:(2**n/4)*2][:,1] += 100
    #points[(2**n/4)*2:(2**n/4)*3][:,1] += 100
    test_points = np.array([[-0.503197961, 0.010541982, 0.308543183, 0.506703083, 1.145723424, 0.557661533, -0.618785329, 1.045596663, 0.165573193, -0.847130537, 2.253579054, -1.336342554, -0.648908105, -1.980705207, 0.753960391, 1.522945569], [0.53855964, 0.468148519, 0.280514511, 0.448179652, 0.146726334, -0.373253851, -0.229981735, 0.568031241, -2.094664344, 0.719403589, 0.393574594, 0.178656057, -0.635632611, 2.170616666, 1.40672544, 0.244241029]]).T
    points_distance = sc_dist.squareform(sc_dist.pdist(points)) # матрица расстояний между всеми точками
    return points_distance, points

def n_power(points_distance):
    return int(np.log2(points_distance.shape[0]))

def goodness(points_distance, d, posiitons):
    return sum([sum(points_distance[posiitons[key], posiitons[d[key]]]) for key in d])/(n*2**(n-1))

#@profile
def genetic_coding(points_distance):
    """
    Собственно генетическое кодирование массива точек.
    На вход подается массив точек, на выходе массив кодовых комбинаций в двоичном и десятичном виде
    """
    n = n_power(points_distance) # 2^n - количество точек
    d = generate_ones(n) # словрь с родительскими точками
    flags = flag(n) # кумулятивная сумма сочетаний всех комбинаций из n
    arr_sorted_int, _ = sorted_ones(n) # точки отсортированные по категориям. внутри каждой категории по возрастанию.
    genetic_position = np.zeros(2**n, dtype=np.int32)
    step_zero = np.argmin(sum(points_distance))
    genetic_position[0] = step_zero
    genetic_position[arr_sorted_int[flags[0]:flags[1]]] = np.argsort(points_distance[step_zero])[1:n+1]
    for i in xrange(n-1):
        distance_next = np.array([sum(points_distance[genetic_position[d[key]]]) for key in arr_sorted_int[flags[i+1]:flags[i+2]]])
        new_points = []
        #print [d[key] for key in arr_sorted_int[sum(flags[:i+2]):sum(flags[:i+3])]]
        #print [genetic_position_back[d[key]] for key in arr_sorted_int[sum(flags[:i+2]):sum(flags[:i+3])]]
        distance_next[:, genetic_position[arr_sorted_int[:flags[i+1]]]] = np.inf
        min_row = np.min(distance_next, axis=1)
        min_row_arg = np.argmin(distance_next, axis=1)
        old_point = np.argmin(min_row)
        new_point = min_row_arg[old_point]
        for _ in xrange(distance_next.shape[0]):
            genetic_position[arr_sorted_int[old_point+flags[i+1]]] = new_point
            distance_next[old_point, :] = np.inf
            distance_next[:, new_point] = np.inf
            min_row[old_point] = np.inf
            new_points.append(new_point)
            old_old_point = np.argmin(min_row)
            while (min_row_arg[old_old_point] in new_points) and len(new_points) < (distance_next.shape[0]):
                min_row[old_old_point] = np.min(distance_next[old_old_point, :])
                min_row_arg[old_old_point] = np.argmin(distance_next[old_old_point, :])
                old_old_point = np.argmin(min_row)
            old_point = old_old_point
            new_point = min_row_arg[old_point]
    result = np.argsort(genetic_position)
    goodness_genetic = goodness(points_distance, d, genetic_position)
    base = '{0:0' + str(n) + 'b}'
    return result, [base.format(item) for item in result], goodness_genetic, genetic_position

def random_coding(points_distance):
    n = n_power(points_distance)
    random_position = np.random.permutation(2**n)
    d = generate_ones(n)
    goodness_random = goodness(points_distance, d, random_position)
    return goodness_random, random_position

def plot_lines(positions, points, d, num, fig):
    [plt.plot([points[positions[key],0],points[positions[one]][0]],[points[positions[key],1],points[positions[one]][1]], '-ko') for key in d for one in d[key]]
    #ax = fig.add_subplot(num, projection='3d')
    #[ax.plot([points[positions[key],0],points[positions[one]][0]],[points[positions[key],1],points[positions[one]][1]], [points[positions[key],2],points[positions[one]][2]], '-ko') for key in d for one in d[key]]

def goodness_overall(points, goodness_genetic, sample):
    goodnes_list = [random_coding(points)[0] for _ in xrange(sample)]
    return (np.mean(goodnes_list) - goodness_genetic)/np.std(goodnes_list), goodnes_list

def plot_results(genetic_position, random_position, plot_points, test_points, d):
    plt.subplot(221)
    plt.title('Genetic')
    plt.axis([-5, 5, -5, 5])
    #fig = plt.figure()
    plot_lines(genetic_position, plot_points, d, 221, 'fig')
    plt.subplot(222)
    plt.title('Random')
    plt.axis([-5, 5, -5, 5])
    plot_lines(random_position, plot_points, d, 222, 'fig')
    plt.subplot(212)
    overall = goodness_overall(test_points, goodness_genetic, 1000)
    plt.hist(overall[1], bins=20,color='0.75')
    plt.xlabel(r'$\mu_g={0}\ \mu_r={1}\ \sigma_r={2}$'.format(goodness_genetic, np.mean(overall[1]), np.std(overall[1])),fontsize=14)
    #plt.axvline(goodness_genetic,color='r')

n = 13
d = generate_ones(n)
#np.random.seed(100)

test_points, plot_points = generate_points(n)
result, result_bytes, goodness_genetic, genetic_position = genetic_coding(test_points)
#goodness_random, random_position = random_coding(test_points)


#olo = goodness_overall(test_points, goodness_genetic, 100)
#print goodness_genetic, np.mean(olo[1]), np.std(olo[1]), olo[0]
end = time.time()
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
print end - start