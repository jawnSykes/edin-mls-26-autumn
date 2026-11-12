"""Search benchmark: one CSV row per cell, each cell measured in its own child process.

A cell is one (index kind, M, efConstruction, order, efSearch, SIMD level, simulated LL size).
Every combination of the list-valued flags is a cell. Index kinds: exact_dense (the course's
ExactDenseRetriever, called as the pipeline calls it), faiss_exact (FAISS IndexFlatIP) and hnsw.

  --mode time        pinned, single-threaded wall clock: p50/p95/p99 per query, batched QPS,
                     FAISS work counters. Cells are interleaved over shuffled --rounds.
  --mode cachegrind  the child runs under valgrind --tool=cachegrind with a simulated D1/LL.
                     Each cell also runs a 'nosearch' twin (load + reorder, no search) that is
                     subtracted, so only the search's misses count. Deterministic: one round.

Before measuring, every reordered graph is checked against the original order (same top-k,
same distance computations). A mismatch aborts the run: the permutation is wrong.

Indexes, permutations and ground truth live in $PERSIST_INDEX_PATH/<--name>/ (ground truth
comes from `exp.py prep`). Normally run via run_experiments.sbatch, e.g.:
    bench.py --name 24k --embeddings .../corpus_enc_a.npy --queries $PERSIST_INDEX_PATH/queries/course.npy \\
        --index hnsw --M 16 --efc 40 --ef 32 128 --order identity random:0 random:1 bfs gorder:5 \\
        --out rows.csv --log-dir $DISK_SCRATCH/logs
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import resource
import socket
import subprocess
import time

import numpy as np

GT_K = 100  # exp.py prep stores exact top-100


# ----------------------------------------------------------------------------- orderings
# Each returns perm with perm[new_pos] = old_id (FAISS permute_entries convention).

def order_random(index, arg):
    from extensions.reorder import random_order
    return random_order(index.ntotal, int(arg))


def order_bfs(index, arg):
    from extensions.reorder import bfs_order, level0_csr
    return bfs_order(level0_csr(index), int(index.hnsw.entry_point))


def order_gorder(index, arg):
    from extensions.reorder import goorder, level0_csr
    return goorder(level0_csr(index), int(arg))


ORDERS = {"identity": None, "random": order_random, "bfs": order_bfs, "gorder": order_gorder}
NEEDS_ARG = {"random", "gorder"}


def parse_order(spec):
    """'gorder:5' -> ('gorder', '5')."""
    name, _, arg = spec.partition(":")
    if name not in ORDERS:
        raise SystemExit(f"unknown order {spec!r}; known: {sorted(ORDERS)}")
    if name in NEEDS_ARG and not arg:
        raise SystemExit(f"order {name!r} needs an argument, e.g. {name}:5")
    return name, arg or None


def perm_hash(perm):
    """Short fingerprint logged with every row, so a result can be tied to its exact layout."""
    return hashlib.sha1(np.ascontiguousarray(perm, dtype=np.int64).tobytes()).hexdigest()[:12]


def ensure_perm(index_path, spec):
    """Compute once and cache next to the index. Returns (path or None, compute seconds, hash)."""
    name, arg = parse_order(spec)
    if name == "identity":
        return None, 0.0, "identity"
    p = index_path.with_name(f"{index_path.stem}.{spec.replace(':', '-')}.perm.npz")
    if not p.exists():
        import faiss
        import scipy.sparse.csgraph  # noqa: F401  (imported before timing: time the ordering, not imports)
        import extensions.reorder  # noqa: F401
        index = faiss.read_index(str(index_path))
        t = time.perf_counter()
        perm = ORDERS[name](index, arg)
        secs = time.perf_counter() - t
        assert np.array_equal(np.sort(perm), np.arange(index.ntotal)), f"{spec} is not a permutation"
        tmp = p.with_name(p.name + f".tmp{os.getpid()}.npz")  # array tasks may race; rename is atomic
        np.savez(tmp, perm=perm, seconds=secs)
        os.replace(tmp, p)
        print(f"[prepare] {p.name}: computed in {secs:.1f}s", flush=True)
    z = np.load(p)
    return p, float(z["seconds"]), perm_hash(z["perm"])


# ----------------------------------------------------------------------------- index kinds
# prepare -> dict(path, build_s, index_bytes). Exact kinds have no M/efc/order/ef.

def _meta(path):
    m = path.with_name(path.name + ".meta.json")
    return json.loads(m.read_text()) if m.exists() else {}


def _save_meta(path, **kw):
    path.with_name(path.name + ".meta.json").write_text(json.dumps(kw))


def prepare_exact_dense(args, cache, M, efc):
    """The course's ExactDenseRetriever (src/retriever.py): no build, just the vector matrix."""
    X = np.load(args.embeddings, mmap_mode="r")
    return dict(path=args.embeddings, build_s=0.0, index_bytes=X.size * 4, n=X.shape[0], d=X.shape[1])


def prepare_faiss_exact(args, cache, M, efc):
    import faiss
    path = cache / "flat.faiss"  # file name kept, so existing caches are reused
    if not path.exists():
        X = np.ascontiguousarray(np.load(args.embeddings), dtype=np.float32)
        t = time.perf_counter()
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        faiss.write_index(index, str(path))
        _save_meta(path, build_s=time.perf_counter() - t)
    index = faiss.read_index(str(path))
    return dict(path=path, build_s=_meta(path).get("build_s", float("nan")), index_bytes=path.stat().st_size,
                n=index.ntotal, d=index.d)


def prepare_hnsw(args, cache, M, efc):
    import faiss
    from extensions.hnsw import HNSWRetriever
    os.environ["PERSIST_INDEX_PATH"] = str(cache)  # HNSWRetriever names by stem; corpora share a stem
    t = time.perf_counter()
    r = HNSWRetriever(args.embeddings, args.ids, M=M, efConstruction=efc, efSearch=16)
    path = Path(r.index_path)
    if not r.loaded:
        # build_s: graph construction only (index.add). build_total_s adds loading the .npy,
        # the norm check and writing the index. Peak RSS is this whole process's, so it
        # includes anything loaded before the build: an upper bound.
        _save_meta(path, build_s=r.build_s, build_total_s=time.perf_counter() - t,
                   build_threads=faiss.omp_get_max_threads(),
                   build_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    print(f"[prepare] {path.name}: {'loaded' if r.loaded else 'built'}", flush=True)
    m = _meta(path)
    return dict(path=path, build_s=m.get("build_s", float("nan")), build_total_s=m.get("build_total_s", float("nan")),
                build_threads=m.get("build_threads", ""),
                build_peak_rss_mb=m.get("build_peak_rss_mb", float("nan")), index_bytes=path.stat().st_size,
                n=r.index.ntotal, d=r.index.d)


# exact_dense: the current method, the course's ExactDenseRetriever itself (numpy matmul + top-k)
# faiss_exact: the same exact search in FAISS (IndexFlatIP), to separate library from algorithm
# hnsw:        the graph index
INDEXES = {"exact_dense": prepare_exact_dense, "faiss_exact": prepare_faiss_exact, "hnsw": prepare_hnsw}


# ----------------------------------------------------------------------------- child process

def child(cfg):
    """Modes: 'check' (reorder correctness), 'time', 'cg' (under valgrind) and 'perf' (under
    perf stat); cg and perf run twice, cfg['search'] on/off, and the parent subtracts."""
    if cfg["cpu"] is not None:
        os.sched_setaffinity(0, {cfg["cpu"]})
    import faiss
    from extensions.reorder import apply_order, to_original_ids
    faiss.omp_set_num_threads(1)
    k, kind = cfg["k"], cfg["kind"]
    Q = np.ascontiguousarray(np.load(cfg["queries"])[:cfg["nq"]], dtype=np.float32)
    stats = faiss.cvar.hnsw_stats

    if kind == "exact_dense":
        from src.retriever import ExactDenseRetriever
        r = ExactDenseRetriever(cfg["path"], cfg["ids"])
        row_of = {rid: i for i, rid in enumerate(r.record_ids)}

        def search_one(q):  # what is timed and profiled: the retriever's own call, as the pipeline makes it
            return r.retrieve(q[0], k)

        def search(q):  # row indices, for recall (outside the timed region)
            return np.array([[row_of[x] for x in search_one(q)["record_ids"]]])
    else:
        index = faiss.read_index(cfg["path"])
        if kind == "hnsw":
            index.hnsw.efSearch = cfg["ef"]

        def search(q):
            return index.search(q, k)[1]

        def search_d(q):
            return index.search(q, k)

        search_one = search

    def one_pass():
        stats.reset()
        I = np.concatenate([search(Q[i:i + 1]) for i in range(len(Q))])
        return I, int(stats.ndis), int(stats.nhops)

    out = {}
    if cfg["mode"] == "check":
        def check_pass():
            stats.reset()
            D, I = zip(*(search_d(Q[i:i + 1]) for i in range(len(Q))))
            return np.concatenate(D), np.concatenate(I), int(stats.ndis)
        D_ref, I_ref, ndis_ref = check_pass()
        perm = apply_order(index, np.load(cfg["perm"])["perm"])
        D, I, ndis = check_pass()
        # Same id at every rank, or the same distance there (duplicate vectors tie, and FAISS
        # may break the tie by id, which the reordering changes).
        same = (to_original_ids(I, perm) == I_ref) | (D == D_ref)
        out = dict(topk_match=float(same.all(axis=1).mean()),
                   ndis_rel_diff=abs(ndis - ndis_ref) / max(ndis_ref, 1))
    else:
        perm = apply_order(index, np.load(cfg["perm"])["perm"]) if cfg["perm"] else None

        def plain_pass():
            for i in range(len(Q)):
                search_one(Q[i:i + 1])

        if cfg["mode"] == "cg":
            if cfg["search"]:
                stats.reset()
                plain_pass()
                out["ndis"], out["nhops"] = int(stats.ndis), int(stats.nhops)
        elif cfg["mode"] == "perf":
            # Under `perf stat -D -1 --control`: counters start disabled and are switched on only
            # around the timed passes, so start-up, loading and the warm-up pass never count.
            plain_pass()  # warm-up
            ctl, ack = open(cfg["perf_ctl"], "w"), open(cfg["perf_ack"])

            def perf_cmd(c):
                ctl.write(c + "\n")
                ctl.flush()
                ack.readline()

            stats.reset()
            perf_cmd("enable")
            for _ in range(cfg["reps"]):
                plain_pass()
            perf_cmd("disable")
            out["ndis"], out["nhops"] = int(stats.ndis), int(stats.nhops)
        else:
            I, ndis, nhops = one_pass()  # also the warm-up pass
            if perm is not None:
                I = to_original_ids(I, perm)
            gt = np.load(cfg["gt"])[:cfg["nq"], :k]
            lat = np.empty(cfg["reps"] * len(Q), dtype=np.int64)
            j = 0
            for _ in range(cfg["reps"]):
                for i in range(len(Q)):
                    t = time.perf_counter_ns()
                    search_one(Q[i:i + 1])
                    lat[j] = time.perf_counter_ns() - t
                    j += 1
            if kind == "exact_dense":
                batch_qps = float("nan")  # ExactDenseRetriever has no batched call
            else:
                t = time.perf_counter()
                search(Q)  # batched: all queries in one call, still 1 thread
                batch_qps = len(Q) / (time.perf_counter() - t)
            us = lat / 1e3
            out = dict(p50_us=np.percentile(us, 50), p95_us=np.percentile(us, 95), p99_us=np.percentile(us, 99),
                       mean_us=us.mean(), batch_qps=batch_qps,
                       ndis_per_q=ndis / len(Q), nhops_per_q=nhops / len(Q),
                       recall_at_k=float(np.mean([len(set(I[i]) & set(gt[i])) / k for i in range(len(Q))])),
                       # recall@r vs exact top-r, from the same top-k results
                       **{f"recall_at_{r_}": float(np.mean([len(set(I[i, :r_]) & set(gt[i, :r_])) / r_ for i in range(len(Q))]))
                          for r_ in (1, 5) if r_ < k},
                       # peak RSS of this search process (index + Python + numpy/FAISS), in MB
                       search_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    out.update(simd_level=faiss.SIMDConfig.get_level_name(), faiss_version=faiss.__version__)
    print("RESULT " + json.dumps({k_: (v if isinstance(v, str) else float(v)) for k_, v in out.items()}), flush=True)


# ----------------------------------------------------------------------------- parent

CG_RE = re.compile(r"==\d+==\s+(D\s+refs|D1\s+misses|LLd\s+misses):\s+([\d,]+)")


PERF_EVENTS = "cycles:u,instructions:u,LLC-loads:u,LLC-load-misses:u,L1-dcache-load-misses:u,dTLB-load-misses:u"


def parse_perf(path):
    """perf stat -x, output -> {event: count}. On hybrid Intel each event is reported per core
    type (cpu_core/..., cpu_atom/...); the child is pinned to one core, so the counts are summed."""
    counts = {}
    for line in Path(path).read_text().splitlines():
        f = line.split(",")
        if len(f) < 3 or not f[0].replace(".", "", 1).isdigit():
            continue  # comments, blank lines, <not counted>, <not supported>
        ev = f[2].split("/")[-2] if "/" in f[2] else f[2]
        ev = ev.split(":")[0].replace("-", "_")
        counts[ev] = counts.get(ev, 0) + float(f[0])
    return counts


def run_child(cfg, log, simd=None, valgrind=None, perf_out=None):
    """Run one child, log its output, return (RESULT dict, valgrind or perf counters).
    `valgrind` is a command prefix (valgrind or perf stat); `perf_out` the perf stat output file."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "_child", json.dumps(cfg)]
    if valgrind:
        cmd = valgrind + cmd
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if simd:
        env["FAISS_SIMD_LEVEL"] = simd
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"$ {' '.join(cmd)}\n--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr}")
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    if proc.returncode != 0 or not lines:
        raise RuntimeError(f"child failed (rc={proc.returncode}), see {log}\n{proc.stderr[-2000:]}")
    counters = {re.sub(r"\s+", "_", m.group(1)): int(m.group(2).replace(",", ""))
                for m in CG_RE.finditer(proc.stderr)}
    if perf_out:
        counters = parse_perf(perf_out)
        if "cycles" not in counters:
            raise RuntimeError(f"no perf counts in {perf_out}")
    elif valgrind and "LLd_misses" not in counters:
        raise RuntimeError(f"no cachegrind summary in {log}")
    return json.loads(lines[-1][len("RESULT "):]), counters


def pick_cpu(requested):
    if requested is not None:
        return requested
    if os.environ.get("BENCH_CPU"):  # set by with_pinned_freq.sh, so both pin the same core
        return int(os.environ["BENCH_CPU"])
    pcores = Path("/sys/devices/cpu_core/cpus")  # hybrid Intel: prefer a P-core
    if pcores.exists():
        return int(re.split(r"[-,]", pcores.read_text().strip())[0])
    return min(os.sched_getaffinity(0))


def freq_state(cpu):
    """Clock settings for the pinned core, so pinned and unpinned runs can be told apart."""
    def rd(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return ""
    c = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq"
    return {"governor": rd(f"{c}/scaling_governor"), "no_turbo": rd("/sys/devices/system/cpu/intel_pstate/no_turbo"),
            "max_freq_khz": rd(f"{c}/scaling_max_freq")}


def machine_info():
    info = {"host": socket.gethostname().split(".")[0]}
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for key, col in [("Model name", "cpu_model"), ("L1d cache", "l1d"), ("L2 cache", "l2"), ("L3 cache", "l3")]:
            m = re.search(rf"^\s*{key}:\s*(.+)$", lscpu, re.M)
            if m:
                info[col] = m.group(1).strip()
    except FileNotFoundError:
        pass
    return info


def append_row(path, row):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="corpus label, e.g. 24k; cache subdir and 'corpus' column")
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--ids", type=Path, help="default: <embeddings stem>_ids.json next to it")
    ap.add_argument("--queries", type=Path, required=True, help="(nq, d) float32 .npy, L2-normalised")
    ap.add_argument("--nq", type=int, default=None, help="use only the first N queries")
    ap.add_argument("--index", nargs="+", default=["hnsw"], choices=sorted(INDEXES))
    ap.add_argument("--M", nargs="+", type=int, default=[16])
    ap.add_argument("--efc", nargs="+", type=int, default=[40])
    ap.add_argument("--ef", nargs="+", type=int, default=[32, 128])
    ap.add_argument("--order", nargs="+", default=["identity"], help="identity, random:<seed>, bfs, gorder:<w>")
    ap.add_argument("--simd", nargs="+", default=[None], help="FAISS_SIMD_LEVEL per cell (default: inherit env)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mode", choices=["time", "cachegrind", "perf"], default="time")
    ap.add_argument("--reps", type=int, default=5, help="timed passes over all queries per cell per round")
    ap.add_argument("--rounds", type=int, default=3, help="interleaved rounds (time mode)")
    ap.add_argument("--seed", type=int, default=0, help="shuffles the interleaving only")
    ap.add_argument("--cpu", type=int, default=None, help="core to pin to (default: BENCH_CPU, else first P-core)")
    ap.add_argument("--LL", nargs="+", type=int, default=[8 << 20], help="cachegrind: simulated LL bytes (16-way, 64B)")
    ap.add_argument("--D1", default=os.environ.get("D1", ""), help="cachegrind: size,ways,line (default: $D1)")
    ap.add_argument("--valgrind", default=os.environ.get("VALGRIND", "valgrind"))
    ap.add_argument("--perf", default=os.environ.get("PERF", "perf"), help="perf mode: the perf binary")
    ap.add_argument("--perf-events", default=PERF_EVENTS, help="perf mode: events for perf stat -e")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True, help="per-cell logs and cachegrind files (scratch)")
    args = ap.parse_args()

    args.embeddings, args.queries = args.embeddings.expanduser(), args.queries.expanduser()
    args.ids = (args.ids or args.embeddings.with_name(args.embeddings.stem + "_ids.json")).expanduser()
    for key in ("index", "M", "efc", "ef", "order", "simd", "LL"):  # e.g. ef95 == ef99 -> one cell
        setattr(args, key, list(dict.fromkeys(getattr(args, key))))
    for spec in args.order:
        parse_order(spec)
    if "PERSIST_INDEX_PATH" not in os.environ:
        raise SystemExit("set PERSIST_INDEX_PATH (indexes, permutations and ground truth are cached there)")
    cache = Path(os.environ["PERSIST_INDEX_PATH"]).expanduser() / args.name
    cache.mkdir(parents=True, exist_ok=True)
    cg = args.mode == "cachegrind"
    pf = args.mode == "perf"
    if cg and not args.D1:
        raise SystemExit("cachegrind mode needs --D1 size,ways,line (or $D1)")
    if cg and any((lv or os.environ.get("FAISS_SIMD_LEVEL")) not in (None, "", "AVX2", "NONE") for lv in args.simd):
        raise SystemExit("cachegrind mode: valgrind cannot run AVX-512, use --simd AVX2")
    args.log_dir.mkdir(parents=True, exist_ok=True)  # cachegrind writes its out-files here
    gt = cache / f"gt-{args.queries.stem}-k{GT_K}.npy"
    if not gt.exists():
        raise SystemExit(f"missing ground truth {gt}; run `exp.py prep` first")
    nq_total = len(np.load(args.queries, mmap_mode="r"))
    nq = min(args.nq or nq_total, nq_total)
    cpu = pick_cpu(args.cpu)
    rounds = 1 if cg else args.rounds

    # ---- prepare: graphs, permutations, correctness check
    graphs = []  # (kind, M, efc, info, {order: (perm path, secs, hash)})
    for kind in args.index:
        grid = itertools.product(args.M, args.efc) if kind == "hnsw" else [(None, None)]
        for M, efc in grid:
            info = INDEXES[kind](args, cache, M, efc)
            orders = args.order if kind == "hnsw" else ["identity"]
            perms = {o: ensure_perm(info["path"], o) for o in orders}
            graphs.append((kind, M, efc, info, perms))

    for kind, M, efc, info, perms in graphs:
        for order, (p, _, _) in perms.items():
            if p is None:
                continue
            for ef in args.ef:
                cfg = dict(mode="check", kind=kind, path=str(info["path"]), perm=str(p), queries=str(args.queries),
                           nq=nq, k=args.k, ef=ef, cpu=cpu)
                res, _ = run_child(cfg, args.log_dir / f"check-{kind}-M{M}-efc{efc}-{order.replace(':', '-')}-ef{ef}.log")
                if res["topk_match"] != 1.0 or res["ndis_rel_diff"] >= 1e-3:
                    raise SystemExit(f"reordering changed results: {kind} M={M} efc={efc} {order} ef={ef}: {res}")
    print(f"[prepare] reorder correctness check passed", flush=True)

    # ---- measure
    cells = []
    for kind, M, efc, info, perms in graphs:
        efs = args.ef if kind == "hnsw" else [None]
        for order, ef, simd, LL in itertools.product(perms, efs, args.simd, args.LL if cg else [None]):
            cells.append((kind, M, efc, info, order, perms[order], ef, simd, LL))

    base = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **machine_info(), "pinned_cpu": cpu, **freq_state(cpu),
            "mode": args.mode, "tag": args.tag, "corpus": args.name, "queries": args.queries.stem,
            "nq": nq, "k": args.k, "reps": 0 if cg else args.reps}
    if pf:
        base["perf_events"] = args.perf_events
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary, total, done = {}, len(cells) * rounds, 0
    for rnd in range(rounds):
        for ci in rng.permutation(len(cells)):
            kind, M, efc, info, order, (p, reorder_s, phash), ef, simd, LL = cells[ci]
            done += 1
            cid = f"{kind}-M{M}-efc{efc}-{order.replace(':', '-')}-ef{ef}-{simd}-LL{LL}-r{rnd}"
            cfg = dict(kind=kind, path=str(info["path"]), ids=str(args.ids), perm=str(p) if p else None,
                       queries=str(args.queries), gt=str(gt), nq=nq, k=args.k, ef=ef, reps=args.reps, cpu=cpu)
            row = {**base, "round": rnd, "index": kind, "M": M, "efc": efc, "order": order, "ef": ef,
                   "simd_requested": simd or os.environ.get("FAISS_SIMD_LEVEL", ""), "perm_sha1": phash,
                   "reorder_s": round(reorder_s, 4), "build_s": info.get("build_s"),
                   "build_total_s": info.get("build_total_s", ""), "build_threads": info.get("build_threads", ""),
                   "build_peak_rss_mb": info.get("build_peak_rss_mb", ""), "index_bytes": info["index_bytes"],
                   "n": info["n"], "d": info["d"]}
            if cg:
                row.update(D1=args.D1, LL_bytes=LL)
                vg = lambda twin: [args.valgrind, "--tool=cachegrind", "--cache-sim=yes", f"--D1={args.D1}",
                                   f"--LL={LL},16,64", f"--cachegrind-out-file={args.log_dir / (cid + f'-{twin}.cg.out')}"]
                res, c_s = run_child({**cfg, "mode": "cg", "search": True}, args.log_dir / f"{cid}-search.log", simd, vg("search"))
                _, c_b = run_child({**cfg, "mode": "cg", "search": False}, args.log_dir / f"{cid}-nosearch.log", simd, vg("nosearch"))
                net = {key: c_s[key] - c_b.get(key, 0) for key in c_s}
                row.update(simd_level=res["simd_level"], faiss_version=res["faiss_version"],
                           ndis_per_q=res.get("ndis", 0) / nq, nhops_per_q=res.get("nhops", 0) / nq,
                           **{f"{key}_raw": c_s[key] for key in c_s}, **{f"{key}_base": c_b.get(key, 0) for key in c_s},
                           D_refs_per_q=net.get("D_refs", 0) / nq, D1_miss_per_q=net["D1_misses"] / nq,
                           LL_miss_per_q=net["LLd_misses"] / nq,
                           LL_miss_per_ndis=net["LLd_misses"] / res["ndis"] if res.get("ndis") else "")
                msg = f"D1/q={row['D1_miss_per_q']:.0f} LL/q={row['LL_miss_per_q']:.0f} LL={LL >> 20}MiB"
                key_cols = ["LL_miss_per_q"]
            elif pf:
                # Real hardware counters, warm cache, per query. The child switches the counters on
                # only around its timed passes (perf's control FIFOs), so nothing else counts.
                fifo_dir = args.log_dir / f"{cid}-fifo"
                fifo_dir.mkdir(parents=True, exist_ok=True)
                ctl, ack, pout = fifo_dir / "ctl", fifo_dir / "ack", args.log_dir / f"{cid}.perf"
                for f_ in (ctl, ack):
                    if not f_.exists():
                        os.mkfifo(f_)
                pcmd = [args.perf, "stat", "-D", "-1", "--control", f"fifo:{ctl},{ack}", "-x,", "-o", str(pout),
                        "-e", args.perf_events, "--"]
                res, net = run_child({**cfg, "mode": "perf", "perf_ctl": str(ctl), "perf_ack": str(ack)},
                                     args.log_dir / f"{cid}.log", simd, pcmd, pout)
                nqs = nq * args.reps
                row.update(simd_level=res["simd_level"], faiss_version=res["faiss_version"],
                           ndis_per_q=res.get("ndis", 0) / nqs, nhops_per_q=res.get("nhops", 0) / nqs,
                           **{f"{key}_total": v for key, v in net.items()},
                           **{f"{key}_per_q": v / nqs for key, v in net.items()},
                           ipc=net["instructions"] / net["cycles"] if net.get("cycles") else "",
                           LLC_miss_per_ndis=net.get("LLC_load_misses", 0) / res["ndis"] if res.get("ndis") else "")
                msg = (f"cycles/q={row['cycles_per_q']:.0f} ipc={row['ipc']:.2f} "
                       f"LLC-miss/q={row.get('LLC_load_misses_per_q', float('nan')):.0f}")
                key_cols = ["cycles_per_q", "LLC_load_misses_per_q"]
            else:
                res, _ = run_child({**cfg, "mode": "time"}, args.log_dir / f"{cid}.log", simd)
                row.update(res)
                msg = (f"p50={res['p50_us']:.1f}us p95={res['p95_us']:.1f}us recall@{args.k}={res['recall_at_k']:.3f} "
                       f"ndis/q={res['ndis_per_q']:.0f} simd={res['simd_level']}")
                key_cols = ["p50_us", "p95_us"]
            append_row(args.out, row)
            summary.setdefault((kind, M, efc, order, ef, simd, LL), []).append(row)
            print(f"[{done}/{total}] r{rnd} {args.name} {kind} M={M} efc={efc} {order:<10} ef={ef} {msg}", flush=True)
    print(f"appended {total} rows -> {args.out}")
    if not cg and freq_state(cpu) != {k: base[k] for k in ("governor", "no_turbo", "max_freq_khz")}:
        print("!! clock settings changed during the run; rows record the state at the start")

    # Median over rounds with the min-max spread, to see whether a difference beats the noise.
    print(f"\nmedian [min-max] over {rounds} round(s)")
    for key, rs in summary.items():
        txt = []
        for c in key_cols:
            v = np.array([float(r[c]) for r in rs])
            txt.append(f"{c}={np.median(v):.1f} [{v.min():.1f}-{v.max():.1f}]")
        print(f"  {' '.join(str(x) for x in key if x is not None):<40} " + "  ".join(txt))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "_child":
        child(json.loads(sys.argv[2]))
    else:
        main()
