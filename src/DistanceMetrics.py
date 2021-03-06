import numpy as np
from itertools import *


def l0(perm):
    x = np.array(range(len(perm)))
    return np.sum(x != perm)


def kendal_tau(perm):
    """
    Computes the Kendal Tau distance
    """
    count = 0
    for (i, j) in combinations(range(len(perm)), 2):
        if perm[i] > perm[j]:
            count += 1
    return count


def min_swap(perm):
    """
    Calculates the number of arbitrary swaps needed to 
    put array perm into an ordered arangement. 
    only works if elements are [0, 1, 2,...,n - 1]
    """
    seen_set = set()
    sub_graph_count = 0
    element_count = 0
    for i in xrange(len(perm)):
        # we have already seen this element in an enclosed subgraph
        if i in seen_set:
            continue
        # we have to find an enclosed sub graph
        curr_sub_graph = set()
        curr_sub_graph.add(i)
        j = i
        while perm[j] not in curr_sub_graph:
            curr_sub_graph.add(perm[j])
            j = perm[j]
        if len(curr_sub_graph) > 1:
            sub_graph_count += 1
            element_count += len(curr_sub_graph)
        seen_set |= curr_sub_graph
    return element_count - sub_graph_count


def inverse(perm):
    # Need to double check that this works
    invp = np.zeros(len(perm))
    invp[perm] = np.arrange(len(perm))
    return invp
