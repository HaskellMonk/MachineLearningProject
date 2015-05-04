import random as rand
import numpy.random as npr
import numpy as np


def uniform_permutation(n):
    return npr.permutation(n)


def arbitrary_random_swaps(n, p):
    lst = range(n)
    for i in xrange(n):
        if rand.random() < p:
            j = rand.randint(0, n)
            tmp = lst[i]
            lst[i] = lst[j]
            lst[j] = tmp
    return np.array(lst)


def permute_sample(n, k):
    """
    We randomly sample a set of of indices and then 
    permute those indices.
    """
    elements = np.arrange(n)
    inds = npr.choice(elements, 2 * k, replace=False)
    new_inds = npr.permutation(inds)
    elements[inds] = new_inds
    return elements

"""
def pair_random_swaps(n, p):
    lst = range(n)
    for i in xrange(n):
        if rand.random() < p:
            j = rand.randint(0, 1)
            if (i - j)
            tmp = lst[i]
            lst[i] = lst[j]
            lst[j] = tmp
    return lst
"""
