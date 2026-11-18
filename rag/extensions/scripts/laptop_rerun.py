"""Re-run, on the Core Ultra laptop, every non-pipeline step that was previously produced on the cluster.

The laptop timing and perf data (3-stage1, 4-reorder, perf, build, retune) already came from this
machine, but they loaded a graph, ground truth and permutations that were produced on the cluster
($ASSETS/ann-cache). This script regenerates those inputs here, in a separate cache
($ASSETS/ann-cache-laptop), and checks each against the file the timings used. It also times the
orderings on the pinned P-core, so reorder cost and break-even are same-machine figures.

Stages (default: all, in this order):
  graph    build the HNSW graph (M, efC from settings.json) on 1 thread, pinned, save it to the
           laptop cache; compare its link hash and file hash with the cached graph
  prep     exact top-100 ground truth + FAISS-vs-numpy exact check (exp.py prep) into the laptop
           cache; compare the ground truth with the cached one
  quality  qrels recall/MRR/nDCG@10 on the dev queries, exact vs HNSW (exp.py quality), on the
           laptop-built graph
  reorder  time each ordering in a fresh process pinned to --cpu, interleaved over shuffled
           rounds, with the same timing boundary as bench.py (loaded index -> permutation, which
           includes the level-0 CSR extraction); compare each permutation with the cached one

Further stages (run by name; operating points ef 32 / 64 from the practice-query rule, retune.py):
  reorder-ef  bench.py time: identity, 5 random seeds, BFS, Gorder w=10 at --efs, 5 rounds (4-reorder)
  perf        bench.py perf: exact_dense + the same orderings at --efs, 3 rounds
  wcheck      bench.py time on the 350 practice queries: Gorder w = 3, 5, 10 at --efs, 5 rounds (retune)
  robust      M x efC grid (--Ms, --efcs): ef 16-512 + exact_dense on all course queries (3 rounds), then
              recall on the practice queries (1 round) to pick each configuration's recall >= 0.95 point

Each run writes results/ann-exp-data/laptop/ann/<step>/<host>-<time>/; fold new runs into
results/ann-exp-data/consolidated/ with results/ann-exp-data/consolidate.py.

Run from rag/ on AC power, with --cpu on the performance governor and turbo off
(with_pinned_freq.sh sets both, or set them beforehand):
    ~/mls-rag-env/bin/python extensions/scripts/laptop_rerun.py [stage ...]
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RAG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAG))
ASSETS = Path.home() / "mls-rag-assets"
CACHED = ASSETS / "ann-cache"            # what every laptop timing step loaded
LAPTOP = ASSETS / "ann-cache-laptop"     # regenerated here
OUT = RAG / "results/ann-exp-data/laptop/ann"
CORPUS = "24k"
EMB = RAG / "data/rag_corpus/embeddings/corpus_enc_a.npy"
COURSE_QV = ASSETS / "query_vectors/queries_enc_a.npy"
QRELS = RAG / "data/rag_corpus/qrels_dev.tsv"
SETTINGS = RAG / "results/ann-exp-data/consolidated/settings.json"   # fixed graph config (retune.py rule)
PIN_ENV = {"FAISS_SIMD_LEVEL": "AVX2", "OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
EXP = RAG / "extensions/scripts/exp.py"


def sha1_file(p):
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()[:12]


def sysfs(path, default=""):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def on_ac():
    return ",".join(sysfs(str(p)) for p in Path("/sys/class/power_supply").glob("*/online"))


def run_dir(step):
    d = OUT / step / f"{platform.node()}-{datetime.now():%Y%m%dT%H%M%S}"
    d.mkdir(parents=True)
    return d


def write_env(d, step, cpu, extra=""):
    import faiss
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAG, capture_output=True, text=True).stdout.strip()
    lines = [f"step: {step}", f"date: {datetime.now():%Y%m%dT%H%M%S}", f"host: {platform.node()}", f"git: {git}",
             f"python {sys.executable} | faiss {faiss.__version__} | numpy {np.__version__}",
             f"pinned_cpu={cpu} governor={sysfs(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor')} "
             f"no_turbo={sysfs('/sys/devices/system/cpu/intel_pstate/no_turbo', 'n/a')} on_ac={on_ac()} "
             f"thp={sysfs('/sys/kernel/mm/transparent_hugepage/enabled')}",
             " ".join(f"{k}={v}" for k, v in PIN_ENV.items()),
             f"cached={CACHED} laptop_cache={LAPTOP}", extra,
             "--- lscpu", subprocess.run(["lscpu"], capture_output=True, text=True).stdout]
    (d / "env.txt").write_text("\n".join(lines))


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}", flush=True)


def child_env():
    return {**os.environ, **PIN_ENV}


def pinned(cpu, *args):
    return ["taskset", "-c", str(cpu), sys.executable, *map(str, args)]


def run_child(cmd, what):
    p = subprocess.run(cmd, env=child_env(), capture_output=True, text=True, cwd=RAG)
    res = next((json.loads(l[7:]) for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if p.returncode or res is None:
        sys.exit(f"[{what}] child failed:\n{p.stderr[-3000:]}")
    return res


# ----------------------------------------------------------------------------- graph

def graph_child(cfg):
    import faiss
    faiss.omp_set_num_threads(1)
    X = np.ascontiguousarray(np.load(cfg["emb"]), dtype=np.float32)
    index = faiss.IndexHNSWFlat(X.shape[1], cfg["M"], faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = cfg["efc"]
    t = time.perf_counter()
    index.add(X)
    build_s = time.perf_counter() - t
    faiss.write_index(index, cfg["path"])
    links = hashlib.sha1(faiss.vector_to_array(index.hnsw.neighbors).tobytes()).hexdigest()[:12]
    print("RESULT " + json.dumps(dict(build_s=build_s, links_sha1=links)), flush=True)


def stage_graph(a, s):
    import faiss
    name = f"hnsw-corpus_enc_a-{s['M']}-{s['efc']}.faiss"
    cached, new = CACHED / CORPUS / name, LAPTOP / CORPUS / name
    new.parent.mkdir(parents=True, exist_ok=True)
    d = run_dir("graph")
    write_env(d, "graph", a.cpu)
    res = run_child(pinned(a.cpu, __file__, "--child-graph",
                           json.dumps(dict(emb=str(EMB), M=s["M"], efc=s["efc"], path=str(new)))), "graph")
    ci = faiss.read_index(str(cached))
    cached_links = hashlib.sha1(faiss.vector_to_array(ci.hnsw.neighbors).tobytes()).hexdigest()[:12]
    row = dict(corpus=CORPUS, M=s["M"], efc=s["efc"], threads=1, cpu=a.cpu, build_s=res["build_s"],
               links_sha1=res["links_sha1"], cached_links_sha1=cached_links,
               same_links=res["links_sha1"] == cached_links,
               file_sha1=sha1_file(new), cached_file_sha1=sha1_file(cached))
    row["same_file"] = row["file_sha1"] == row["cached_file_sha1"]
    write_rows(d / "rows.csv", [row])
    print(f"[graph] built in {res['build_s']:.2f}s; links identical: {row['same_links']}; "
          f"file identical: {row['same_file']}", flush=True)
    if not row["same_links"]:
        sys.exit("[graph] the laptop graph differs from the cached one: stop, the later stages would use a different graph")
    new.with_name(new.name + ".meta.json").write_text(json.dumps(dict(build_s=res["build_s"], build_threads=1)))


# ----------------------------------------------------------------------------- prep

def stage_prep(a, s):
    d = run_dir("1-setup")
    write_env(d, "1-setup", a.cpu)
    subprocess.run(pinned(a.cpu, EXP, "prep", "--cache", LAPTOP, "--corpus", f"{CORPUS}={EMB}",
                          "--course-queries", COURSE_QV, "--out", d),
                   env=child_env(), cwd=RAG, check=True)
    rows = []
    for q in ("course",):
        f = f"gt-{q}-k100.npy"
        new, old = np.load(LAPTOP / CORPUS / f), np.load(CACHED / CORPUS / f)
        rows.append(dict(item=f, nq=len(new), identical=bool(np.array_equal(new, old)),
                         rows_identical=float((new == old).all(axis=1).mean()),
                         top10_sets_identical=float(np.mean([set(x[:10]) == set(y[:10]) for x, y in zip(new, old)]))))
    for f in ("course.npy", "course_ids.json"):
        rows.append(dict(item=f"queries/{f}", identical=sha1_file(LAPTOP / "queries" / f) == sha1_file(CACHED / "queries" / f)))
    write_rows(d / "rows-vs-cached.csv", rows)
    for r in rows:
        print(f"[prep] {r['item']}: identical to cached = {r['identical']}", flush=True)


# ----------------------------------------------------------------------------- quality

def stage_quality(a, s):
    d = run_dir("9-quality")
    write_env(d, "9-quality", a.cpu)
    subprocess.run(pinned(a.cpu, EXP, "quality", "--cache", LAPTOP, "--corpus", f"{CORPUS}={EMB}",
                          "--queries", COURSE_QV, "--qrels", QRELS, "--settings", SETTINGS, "--out", d),
                   env=child_env(), cwd=RAG, check=True)


# ----------------------------------------------------------------------------- reorder

def reorder_child(cfg):
    import faiss
    import scipy.sparse.csgraph  # noqa: F401  (imported before timing, as bench.py does)
    import extensions.reorder  # noqa: F401
    from extensions.scripts.bench import ORDERS, parse_order, perm_hash
    name, arg = parse_order(cfg["order"])
    index = faiss.read_index(cfg["index"])
    t = time.perf_counter()
    perm = ORDERS[name](index, arg)
    secs = time.perf_counter() - t
    assert np.array_equal(np.sort(perm), np.arange(index.ntotal)), "not a permutation"
    print("RESULT " + json.dumps(dict(reorder_s=secs, perm_sha1=perm_hash(perm))), flush=True)


def stage_reorder(a, s):
    from extensions.scripts.bench import perm_hash
    stem = f"hnsw-corpus_enc_a-{s['M']}-{s['efc']}"
    index = LAPTOP / CORPUS / f"{stem}.faiss"
    if not index.exists():
        sys.exit(f"[reorder] no laptop graph at {index}; run the graph stage first")
    d = run_dir("reorder-time")
    write_env(d, "reorder-time", a.cpu, f"orders={' '.join(a.orders)} rounds={a.rounds} seed={a.seed}")
    cached_hash = {}
    for o in a.orders:
        p = CACHED / CORPUS / f"{stem}.{o.replace(':', '-')}.perm.npz"
        cached_hash[o] = perm_hash(np.load(p)["perm"]) if p.exists() else ""
    rng, rows = random.Random(a.seed), []
    for rnd in range(a.rounds):
        order = a.orders[:]
        rng.shuffle(order)
        for o in order:
            res = run_child(pinned(a.cpu, __file__, "--child-reorder", json.dumps(dict(index=str(index), order=o))), "reorder")
            rows.append(dict(timestamp=datetime.now().isoformat(timespec="seconds"), host=platform.node(),
                             corpus=CORPUS, M=s["M"], efc=s["efc"], order=o, round=rnd, cpu=a.cpu, **res,
                             cached_perm_sha1=cached_hash[o], same_as_cached=res["perm_sha1"] == cached_hash[o],
                             on_ac=on_ac(), loadavg_1m=sysfs("/proc/loadavg").split()[0]))
            print(f"[reorder] round {rnd} {o}: {res['reorder_s']:.2f}s perm {res['perm_sha1']} "
                  f"(same as cached: {rows[-1]['same_as_cached']})", flush=True)
            write_rows(d / "rows.csv", rows)  # rewrite each time: partial runs survive


# ----------------------------------------------------------------------------- bench.py stages
# Operating points come from the practice-query rule (retune.py): ef95 = 32, ef99 = 64.

BENCH = RAG / "extensions/scripts/bench.py"
RETUNE = ASSETS / "ann-cache-retune"     # practice-query vectors + their ground truth (retune.py prep)
ROBUST = ASSETS / "ann-cache-robust"     # graphs for the M x efC robustness sweep
SCRATCH = Path("/tmp") / os.environ.get("USER", "user") / "ann"
ORDERS_TIME = ["identity", "random:0", "random:1", "random:2", "random:3", "random:4", "bfs", "gorder:10"]


def bench(a, step, cache, queries, extra, out_name="rows.csv"):
    """Run bench.py pinned to a.cpu with cache as PERSIST_INDEX_PATH; log to job.log.gz in the run dir."""
    import gzip
    d = run_dir(step)
    cmd = [sys.executable, BENCH, "--name", CORPUS, "--embeddings", EMB, "--queries", queries, "--cpu", a.cpu,
           "--out", d / out_name, "--log-dir", SCRATCH / step / d.name, *extra]
    write_env(d, step, a.cpu, "cmd: " + " ".join(map(str, cmd)))
    env = {**child_env(), "PERSIST_INDEX_PATH": str(cache), "BENCH_CPU": str(a.cpu)}
    log = []
    with subprocess.Popen(list(map(str, cmd)), env=env, cwd=RAG, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True) as p:
        for line in p.stdout:
            print(line, end="", flush=True)
            log.append(line)
    with gzip.open(d / "job.log.gz", "wt") as f:
        f.writelines(log)
    if p.returncode:
        sys.exit(f"[{step}] bench.py failed (rc={p.returncode}); log in {d / 'job.log.gz'}")
    return d


def efs(a):
    return [str(e) for e in a.efs]


def stage_reorder_ef(a, s):
    bench(a, "4-reorder", LAPTOP, LAPTOP / "queries/course.npy",
          ["--index", "hnsw", "--M", s["M"], "--efc", s["efc"], "--ef", *efs(a), "--order", *ORDERS_TIME,
           "--mode", "time", "--rounds", 5, "--reps", 5, "--tag", "4-reorder"])


def stage_perf(a, s):
    bench(a, "perf", LAPTOP, LAPTOP / "queries/course.npy",
          ["--index", "exact_dense", "hnsw", "--M", s["M"], "--efc", s["efc"], "--ef", *efs(a),
           "--order", *ORDERS_TIME, "--mode", "perf", "--rounds", 3, "--reps", 5, "--tag", "perf"])


def stage_wcheck(a, s):
    q = RETUNE / "queries/course_practice.npy"
    if not q.exists():
        sys.exit(f"[wcheck] no practice queries at {q}; run retune.py prep first")
    bench(a, "retune", RETUNE, q,
          ["--index", "hnsw", "--M", s["M"], "--efc", s["efc"], "--ef", *efs(a),
           "--order", "identity", "gorder:3", "gorder:5", "gorder:10", "--mode", "time", "--rounds", 5, "--reps", 3,
           "--tag", "retune-gorder"], out_name="rows-gorder.csv")


def stage_robust(a, s):
    """M x efC grid: timing + recall on all course queries, then recall on practice queries (selection)."""
    cdir = ROBUST / CORPUS
    cdir.mkdir(parents=True, exist_ok=True)
    (ROBUST / "queries").mkdir(exist_ok=True)
    links = {cdir / "gt-course-k100.npy": LAPTOP / CORPUS / "gt-course-k100.npy",
             cdir / "gt-course_practice-k100.npy": RETUNE / CORPUS / "gt-course_practice-k100.npy",
             ROBUST / "queries/course.npy": LAPTOP / "queries/course.npy",
             ROBUST / "queries/course_ids.json": LAPTOP / "queries/course_ids.json",
             ROBUST / "queries/course_practice.npy": RETUNE / "queries/course_practice.npy",
             ROBUST / "queries/course_practice_ids.json": RETUNE / "queries/course_practice_ids.json"}
    stem = f"hnsw-corpus_enc_a-{s['M']}-{s['efc']}.faiss"   # the measured graph: reuse, don't rebuild
    links[cdir / stem] = LAPTOP / CORPUS / stem
    links[cdir / (stem + ".meta.json")] = LAPTOP / CORPUS / (stem + ".meta.json")
    for link, target in links.items():
        if not link.exists():
            link.symlink_to(target.resolve())
    grid = ["--M", *map(str, a.Ms), "--efc", *map(str, a.efcs)]
    sweep = [str(e) for e in (16, 32, 64, 128, 256, 512)]
    bench(a, "robust", ROBUST, ROBUST / "queries/course.npy",
          ["--index", "exact_dense", "hnsw", *grid, "--ef", *sweep, "--mode", "time", "--rounds", 3, "--reps", 5,
           "--tag", "robust"])
    bench(a, "robust-practice", ROBUST, ROBUST / "queries/course_practice.npy",
          ["--index", "hnsw", *grid, "--ef", *sweep, "--mode", "time", "--rounds", 1, "--reps", 1,
           "--tag", "robust-practice"])


STAGES = {"graph": stage_graph, "prep": stage_prep, "quality": stage_quality, "reorder": stage_reorder,
          "reorder-ef": stage_reorder_ef, "perf": stage_perf, "wcheck": stage_wcheck, "robust": stage_robust}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", help="graph prep quality reorder (default: all)")
    ap.add_argument("--cpu", type=int, default=int(os.environ.get("BENCH_CPU", 2)))
    ap.add_argument("--orders", nargs="+", default=["bfs", "gorder:10", "random:0"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="interleaving order only")
    ap.add_argument("--efs", nargs="+", type=int, default=[32, 64], help="operating points (practice-query rule)")
    ap.add_argument("--Ms", nargs="+", type=int, default=[16, 32, 48], help="robust: M grid")
    ap.add_argument("--efcs", nargs="+", type=int, default=[40, 200], help="robust: efConstruction grid")
    ap.add_argument("--child-graph", help=argparse.SUPPRESS)
    ap.add_argument("--child-reorder", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child_graph:
        return graph_child(json.loads(a.child_graph))
    if a.child_reorder:
        return reorder_child(json.loads(a.child_reorder))
    stages = STAGES
    a.stages = a.stages or ["graph", "prep", "quality", "reorder"]
    if set(a.stages) - set(stages):
        ap.error(f"unknown stage(s) {sorted(set(a.stages) - set(stages))}; known: {list(stages)}")
    if "1" not in on_ac().split(","):
        print("[laptop_rerun] WARNING: not on AC power; the earlier laptop timings were all taken on AC",
              file=sys.stderr, flush=True)
    s = json.loads(SETTINGS.read_text())[CORPUS]
    for st in a.stages:
        print(f"=== {st}", flush=True)
        stages[st](a, s)


if __name__ == "__main__":
    main()
