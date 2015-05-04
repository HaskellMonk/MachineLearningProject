from scipy.stats import gaussian_kde
from sklearn.neighbors.kde import KernelDensity
from Moves import *
import numpy as np
import itertools as it
from heapq import *
from collections import deque
import PermutationGeneration as pg
import numpy.linalg as npl
import DistanceMetrics as dm
import munkres
from blist import sortedlist
from random import randint, random
import numpy.random as npr
import csv


def trainKDE(fs):
    return gaussian_kde(fs)


def generateError(f, fs):
    def swap_f(f_i):
        perm = f(range(len(f)))
        return f_i[perm]

    return np.apply_along_axis(swap_f, 1, fs)


def expon_sched(T_0, alpha, i):
    return T_0 * alpha ** i


def boltzmann_sched(T, i):
    return T / np.log(i + 1)


def cauchy_sched(T, i):
    return T / (1 + i)


def linear_sched(T, alpha):
    return T * alpha


def simulated_anneal(f, cost_f, move_f, cooling_sched, T_min=1e-6, epochs=100):
    T_0 = 1.0
    T = T_0
    best_sol = np.arange(len(f))
    best_cost = cost_f(f, best_sol)
    curr_sol = np.arange(len(f))
    curr_cost = best_cost
    j = 0
    while T > T_min:
        for i in xrange(epochs):
            next_sol = move_f(curr_sol)
            next_cost = cost_f(f, next_sol)
            if np.exp((next_cost - curr_cost) / T) > random():
                curr_sol = next_sol
                curr_cost = next_cost
                if next_cost > best_cost:
                    best_cost = next_cost
                    best_sol = next_sol
        T = cooling_sched(T_0, j)
        j += 1
    return best_sol


def BeamSearchGraph(f, beam_width, max_iters, error_prior, kde, move_f):
    pi_0 = np.arange(len(f), dtype=int)
    best_state = pi_0
    best_error = error_prior(pi_0) * kde(f)
    heap = sortedlist([(error_prior(pi_0) * kde(f), pi_0)], key=lambda x: x[0])
    curr_iters = 0
    while len(heap) > 0:
        (prob_f, curr_f) = heap[-1]
        del heap[-1]
        if prob_f > error_prior(curr_f) * kde(f[curr_f]):
            best_state = curr_f
            best_error = error_prior(curr_f) * kde(f[curr_f])
        next_move = [(error_prior(mv) * kde(f[mv]), mv) for mv in move_f(curr_f)]
        # remove all the moves that have error equal to the best element prob
        next_move = filter(lambda (prob, mv): best_error >= error_prior(mv), next_move)
        print next_move
        heap.update(next_move)
        if beam_width < len(heap):
            # delete all the bad nodes
            for i in xrange(len(heap) - beam_width):
                del heap[0]
        curr_iters += 1
        if curr_iters > max_iters:
            return best_state
    return best_state


def assignment(f, distributions):
    """
    The distributions variable is a vector of kdes. One for each
    feature.
    """
    m = munkres.Munkres()
    profit_matrix = np.zeros((len(f), len(f)))
    for i in xrange(len(f)):
        for j in xrange(len(f)):
            profit_matrix[j, i] = distributions[j](f[i])
    cost_matrix = np.max(profit_matrix) - profit_matrix
    permutation = m.compute(cost_matrix)
    return np.array([e2 for (e1, e2) in permutation])


def readBlood():
    with open("../data/Blood/transfusion.data") as f:
        gen = csv.reader(f)
        gen.next()
        for el in gen:
            yield map(float, el)


def readTwitter():
    with open("../data/Twitter/Twitter.data") as f:
        for el in csv.reader(f):
            yield map(float, el)


def test_assignment(training_set, testing_set, error_f, metric):
    (num_examples, num_features) = training_set.shape
    kdes = []
    for i in xrange(num_features):
        kdes.append(gaussian_kde(np.ravel(training_set[:, i])))

    error = 0
    wrong_count = 0
    num_test = testing_set.shape[1]
    for i in xrange(num_test):
        perm = error_f(num_features)
        example = testing_set[i, :]
        assumed_perm = assignment(example[perm], kdes)
        rel_exmp = (example - example[perm[assumed_perm]]) / example
        error += npl.norm(rel_exmp)
        if metric(perm[assumed_perm]) == 0:
            wrong_count += 1
    return (error / num_test, wrong_count / float(num_test))


def test_search(training_set, testing_set, search_f, error_f, metric):
    kernel_density = KernelDensity().fit(training_set)
    (num_examples, num_features) = training_set.shape
    perm_cost = [metric(error_f(num_features)) for i in xrange(num_examples)]
    print perm_cost
    kernel_error = gaussian_kde(perm_cost)

    def density(x):
        print np.exp(kernel_density.score_samples(np.array([x]))[0])
        return np.exp(kernel_density.score_samples(np.array([x]))[0])

    def error_prior(x):
        return kernel_error(metric(x))[0]

    error = 0
    right_count = 0
    num_test = testing_set.shape[1]
    for i in xrange(num_test):
        perm = error_f(num_features)
        print 
        example = testing_set[i, :]
        assumed_perm = search_f(example[perm], error_prior, density)
        error += npl.norm(example - example[perm[assumed_perm]]) / npl.norm(example)
        if metric(perm[assumed_perm]) == 0:
            right_count += 1
    return (error / num_test, right_count / float(num_test))


def k_fold_verification(k, data, test_function):
    #Randomize the data
    permutation = npr.permutation(data.shape[0])
    data = data[permutation, :]
    avg_error_total = 0
    avg_wrong_total = 0
    for i in xrange(k):
        pull_out_start = (data.shape[0] / int(k)) * i
        pull_out_end = (data.shape[0] / int(k)) * (i + 1)
        mask = np.zeros(data.shape[0], dtype=bool)
        mask[pull_out_start:pull_out_end] = True
        test = data[mask, :]
        train = data[True != mask, :]
        (avg_error, percent_wrong) = test_function(train, test)
        avg_error_total += avg_error
        avg_wrong_total += percent_wrong
    return (avg_error_total / k, avg_wrong_total / k)


if __name__ == "__main__":
    data = np.array([exmp for exmp in readBlood()])
    def test1(train, test):
        return test_assignment(train, test, pg.uniform_permutation, dm.min_swap)

    def test2(train, test):
        def search_f(example, error_p, kde):
            return BeamSearchGraph(example, 10, 100, error_p, kde, move_all_swaps)

        return test_search(train, test, search_f, pg.uniform_permutation, dm.min_swap)

    print k_fold_verification(2, data, test2)
