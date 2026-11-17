"""Offline locality statistics of each cached ordering, at the granularities the hardware uses.

Run from rag/ with mls-rag-env (needs faiss + scipy):
    ~/mls-rag-env/bin/python results/ann-exp-data/offline_locality.py
Writes results/ann-exp-data/offline/locality.csv and offline/page_backing.csv (how much of each
retriever's vector array the kernel backs with 2 MiB transparent huge pages on *this* machine).

Byte layout: vector at position p starts at BASE + p*1536 (FAISS IndexFlat codes buffer).
BASE mod 4096 = 16 was measured (glibc mmap header) and is unchanged by permute_entries;
BASE mod 2 MiB is not known, so the 2 MiB column assumes the same 16-byte offset.
"""
import csv
import sys
from pathlib import Path

import faiss
import numpy as np

RAG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAG))
from extensions.reorder import inverse, level0_csr  # noqa: E402

CACHE = Path.home() / "mls-rag-assets" / "ann-cache"
OUT = Path(__file__).resolve().parent / "offline" / "locality.csv"
VEC_BYTES, LINE, PAGE, HUGE, BASE = 1536, 64, 4096, 2 << 20, 16


def span(pos, unit):
    """First and last `unit`-sized block touched by the vector at each position."""
    start = BASE + pos.astype(np.int64) * VEC_BYTES
    return start // unit, (start + VEC_BYTES - 1) // unit


def overlap(pos_a, pos_b, unit):
    a0, a1 = span(pos_a, unit)
    b0, b1 = span(pos_b, unit)
    return (a0 <= b1) & (b0 <= a1)


def distinct_pages_per_row(indptr, pages):
    """Mean over nodes of (# distinct 4 KiB start pages among its neighbours) / degree."""
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    key = np.unique(rows.astype(np.int64) * (1 << 32) + pages)
    distinct = np.bincount((key >> 32).astype(np.int64), minlength=len(indptr) - 1)
    deg = np.diff(indptr)
    ok = deg > 0
    return float((distinct[ok] / deg[ok]).mean())


def sibling_adjacent_per_neighbour(indptr, sib_pos):
    """Mean over nodes of (# pairs of entries in its neighbour list at adjacent positions) / degree.

    Siblings are evaluated back to back in one expansion, so each adjacent pair shares one boundary
    cache line (vectors are 1536 B at a 16-byte offset, i.e. 25 lines each, one shared with p+1).
    This is the saving per distance computation a cache without prefetch or TLB *can* see."""
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr)).astype(np.int64)
    key = np.sort(rows * (1 << 32) + sib_pos)
    same_row = (key[1:] >> 32) == (key[:-1] >> 32)
    adj = same_row & (np.diff(key & 0xFFFFFFFF) == 1)
    pairs = np.bincount((key[1:][adj] >> 32).astype(np.int64), minlength=len(indptr) - 1)
    deg = np.diff(indptr)
    ok = deg > 0
    return float((pairs[ok] / deg[ok]).mean())


def backing_kb(addr, nbytes):
    """Rss and AnonHugePages (kB) summed over every mapping that overlaps [addr, addr + nbytes)."""
    import re
    tot, cur = {"Rss": 0, "AnonHugePages": 0}, False
    for line in open("/proc/self/smaps"):
        m = re.match(r"^([0-9a-f]+)-([0-9a-f]+) ", line)
        if m:
            lo, hi = int(m[1], 16), int(m[2], 16)
            cur = lo < addr + nbytes and addr < hi
        elif cur and line.split(":")[0] in tot:
            tot[line.split(":")[0]] += int(line.split()[1])
    return tot


def page_backing():
    """numpy (as ExactDenseRetriever loads it) vs FAISS IndexFlatIP vs FAISS HNSW storage."""
    thp = Path("/sys/kernel/mm/transparent_hugepage/enabled").read_text().strip()
    npy = {"24k": RAG / "data/rag_corpus/embeddings/corpus_enc_a.npy",
           "58k": RAG / "data/fiqa_full/embeddings/corpus_enc_a.npy"}
    rows, keep = [], []
    for corpus in ["24k", "58k"]:
        X = np.asarray(np.load(npy[corpus]), dtype=np.float32)  # exactly ExactDenseRetriever.__init__
        keep.append(X)
        bufs = [("exact_dense (numpy)", X.ctypes.data, X.nbytes)]
        for label, f in [("faiss_exact (IndexFlatIP)", "flat.faiss"), ("hnsw storage", "hnsw-corpus_enc_a-16-200.faiss")]:
            raw = faiss.read_index(str(CACHE / corpus / f))
            keep.append(raw)  # keep the owner alive while smaps is read
            flat = faiss.downcast_index(raw.storage if hasattr(raw, "storage") else raw)
            bufs.append((label, int(flat.codes.data()), flat.ntotal * flat.d * 4))
        for label, addr, nbytes in bufs:
            t = backing_kb(addr, nbytes)
            rows.append(dict(corpus=corpus, buffer=label, thp_setting=thp, array_kb=nbytes >> 10,
                             rss_kb=t["Rss"], huge_kb=t["AnonHugePages"],
                             frac_on_2m_pages=t["AnonHugePages"] / max(1, nbytes >> 10)))
    with open(OUT.parent / "page_backing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"[pages] {r['corpus']} {r['buffer']}: {100 * r['frac_on_2m_pages']:.0f}% on 2 MiB pages ({thp})")


def main():
    rows = []
    for corpus in ["24k", "58k"]:
        path = CACHE / corpus / "hnsw-corpus_enc_a-16-200.faiss"
        index = faiss.read_index(str(path))
        A = level0_csr(index)
        coo = A.tocoo()
        n = index.ntotal
        layouts = [("identity", None)] + [
            (p.name[len(path.stem) + 1:-len(".perm.npz")].replace("-", ":", 1), p)
            for p in sorted(path.parent.glob(f"{path.stem}.*.perm.npz"))]
        for order, p in layouts:
            pos = np.arange(n) if p is None else inverse(np.load(p)["perm"])
            pr, pc = pos[coo.row], pos[coo.col]
            gap = np.abs(pr - pc)
            # neighbour pages in the node's *new* row order (CSR rows are original ids; fine for a per-row stat)
            start_page = (BASE + pos[A.indices].astype(np.int64) * VEC_BYTES) // PAGE
            rows.append(dict(
                corpus=corpus, order=order, n_links=len(gap),
                mean_link_id_gap=gap.mean(),
                frac_links_within_64=float((gap <= 64).mean()),
                frac_links_adjacent=float((gap == 1).mean()),
                frac_links_share_line=float(overlap(pr, pc, LINE).mean()),
                frac_links_share_4k_page=float(overlap(pr, pc, PAGE).mean()),
                frac_links_share_2m_page=float(overlap(pr, pc, HUGE).mean()),
                distinct_4k_pages_per_neighbour=distinct_pages_per_row(A.indptr, start_page),
                sibling_adjacent_pairs_per_neighbour=sibling_adjacent_per_neighbour(
                    A.indptr, pos[A.indices].astype(np.int64)),
            ))
            print(f"[locality] {corpus} {order}: within64={rows[-1]['frac_links_within_64']:.4f} "
                  f"page={rows[-1]['frac_links_share_4k_page']:.4f}")
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}")
    page_backing()


if __name__ == "__main__":
    main()
