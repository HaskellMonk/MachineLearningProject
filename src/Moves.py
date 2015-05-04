import itertools as it
from random import randint


def move_all_swaps(perm):
    perms = []
    for (i, j) in it.combinations(xrange(len(perm)), 2):
        p = perm.copy()
        p[i] = perm[j]
        p[j] = perm[i]
        perms.append(p)
    return perms


def all_swaps(f):
    for (i, j) in it.combinations(xrange(len(f)), 2):
        ftmp = f[:]
        tmp = ftmp[i]
        ftmp[i] = ftmp[j]
        ftmp[j] = tmp
        yield ftmp


def pair_swaps(f):
    for i in xrange(len(f) - 1):
        ftmp = f[:]
        tmp = ftmp[i]
        ftmp[i] = ftmp[i + 1]
        ftmp[i + 1] = tmp
        yield ftmp


def random_swap(f):
    ftmp = f[:]
    i = randint(0, len(f))
    j = randint(0, len(f))
    tmp = ftmp[i]
    ftmp[i] = ftmp[j]
    ftmp[j] = tmp
    yield ftmp


def random_pair_swap(f):
    ftmp = f[:]
    i = randint(0, len(f) - 1)
    tmp = ftmp[i]
    ftmp[i] = ftmp[i + 1]
    ftmp[i + 1] = tmp
    yield ftmp
