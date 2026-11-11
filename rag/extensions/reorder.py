import numpy as np
from scipy.sparse import csr_matrix

import heapq

def goorder(adj_list: csr_matrix, w: int) -> np.ndarray:
    """GO-PQ (Wei et al. 2016, Alg. 2)"""
    def inc(node):
        if placed[node]:
            return
        keys[node] += 1
        heapq.heappush(heap, (-keys[node], node))

    def dec(node):
        if placed[node]:
            return
        keys[node] -= 1
        heapq.heappush(heap, (-keys[node], node))

    def lazypop(h):
        k,node = heapq.heappop(h)
        # Skip over invalid entries
        while placed[node] or -k != keys[node]:
            k,node = heapq.heappop(h)
        return k,node

    # Pre-processing
    out, inn = _prep_adj_list(adj_list)
    n = adj_list.shape[0]
    keys = [0] * n
    heap = [(0,v) for v in range(n)]
    heapq.heapify(heap)
    placed = [False for _ in range(n)]

    P = []
    # Select start node (one with the most inn degrees)
    v1 = max(range(n), key=lambda i: len(inn[i]))
    P.append(v1)
    placed[v1] = True
    i = 2
    while i <= n:
        v_e = P[i-2]
        for u in out[v_e]:
            inc(u)
        for u in inn[v_e]:
            inc(u)
            for v in out[u]:
                inc(v)
        if i > w + 1:
            v_b = P[i-w-2]
            for u in out[v_b]:
                dec(u)
            for u in inn[v_b]:
                dec(u)
                for v in out[u]:
                    dec(v)
        v_max = lazypop(heap)[1]
        P.append(v_max)
        placed[v_max] = True
        i += 1

    return np.asarray(P, dtype=np.int64)


def _prep_adj_list(adj_list: csr_matrix):
    a = adj_list.copy()
    a.setdiag(0)
    a.eliminate_zeros() # Remove self-edges
    a.sum_duplicates() # Remove duplicate edges
    t = a.T.tocsr() # flip to get incoming edges

    n_rows = a.shape[0]
    out = [a.indices[a.indptr[i]:a.indptr[i + 1]].tolist() for i in range(n_rows)]
    inn = [t.indices[t.indptr[i]:t.indptr[i + 1]].tolist() for i in range(n_rows)]
    return out, inn
