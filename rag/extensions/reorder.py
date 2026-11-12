import numpy as np
from scipy.sparse import csr_matrix

import heapq

def random_order(n: int, seed: int) -> np.ndarray:
    """Uniform random permutation: the negative-control layout (no locality at all)."""
    return np.random.default_rng(seed).permutation(n).astype(np.int64)

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

def inverse(perm: np.ndarray) -> np.ndarray:
    # old_id -> new_pos
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm), dtype=perm.dtype)
    return inv


def apply_order(index, perm: np.ndarray) -> np.ndarray:
    # Apply with permutation with FAISS API
    perm = np.ascontiguousarray(perm, dtype=np.int64)
    if not np.array_equal(np.sort(perm), np.arange(index.ntotal)):
        raise ValueError("perm is not a permutation of 0..ntotal-1")
    index.permute_entries(perm)
    return perm


def to_original_ids(I: np.ndarray, perm: np.ndarray) -> np.ndarray:
    # Map ids from reordered index to original
    return np.where(I >= 0, perm[np.maximum(I, 0)], -1)


def apply_order_arrays(perm: np.ndarray, adj: csr_matrix, vecs: np.ndarray, entry: int):
    # Numpy based reordering for sense check
    adj_new = adj[perm][:, perm].tocsr()
    vecs_new = vecs[perm]
    return adj_new, vecs_new, int(inverse(perm)[entry])


def level0_csr(index) -> csr_matrix:
    # Bottom-layer graph of a FAISS IndexHNSW as CSR (row i = out-links of node i)
    import faiss
    h = index.hnsw
    n = index.ntotal
    offsets = faiss.vector_to_array(h.offsets).astype(np.int64)[:-1]
    neighbors = faiss.vector_to_array(h.neighbors)
    cum = faiss.vector_to_array(h.cum_nneighbor_per_level)
    width = int(cum[1] - cum[0])  # 2*M slots on level 0, padded with -1
    nbrs = neighbors[offsets[:, None] + cum[0] + np.arange(width)]
    rows = np.repeat(np.arange(n), width)
    cols = nbrs.ravel()
    keep = cols >= 0
    return csr_matrix((np.ones(keep.sum(), dtype=np.int8), (rows[keep], cols[keep])), shape=(n, n))


def bfs_order(adj: csr_matrix, start: int) -> np.ndarray:
    # Breadth-first walk from `start`, so a node's neighbours get nearby ids.
    # Nodes the walk can't reach are appended in their original order.
    from scipy.sparse.csgraph import breadth_first_order
    seen = breadth_first_order(adj, start, directed=True, return_predecessors=False)
    rest = np.setdiff1d(np.arange(adj.shape[0]), seen, assume_unique=True)
    return np.concatenate([seen, rest]).astype(np.int64)
