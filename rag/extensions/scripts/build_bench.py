"""Time HNSW graph construction on this machine (the laptop), without touching the cached graphs.

Every other measurement uses ONE saved graph per corpus (built once on the cluster with 8 threads),
so this script never writes an index: each build is timed in a fresh child process and thrown away.

Per build it records:
  build_s       graph construction only (index.add), the same boundary as HNSWRetriever / bench.py
  peak_rss_mb   the child's peak RSS (vectors + graph + Python/FAISS), an upper bound on build memory
  recall@10     of the *new* graph vs exact search at ef 32 and 128 (course queries), to show what
                build nondeterminism does to quality
  graph_sha1    hash of the graph's links: identical across reps <=> the build is deterministic
  same_as_cached  whether that hash equals the cached graph's (the one every other step measured)

Thread configs are "THREADS:CPUS" (taskset list). Defaults, for the Core Ultra 5 235U
(cpus 0-3 = 2 P-cores x 2 hyper-threads, 4-11 = E-cores, 12-13 = LP E-cores):
  1:2     one thread on one P-core (the core every laptop search measurement used)
  2:0,2   two threads, one per P-core (no hyper-thread sharing, no E-cores)
Configs are interleaved across rounds in a seeded shuffled order, so drift lands on every config.

Run from rag/ under the frequency pin, with every CPU you use listed in BENCH_CPUS:
  BENCH_CPUS="0 2" extensions/scripts/with_pinned_freq.sh \
      ~/mls-rag-env/bin/python extensions/scripts/build_bench.py
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RAG = Path(__file__).resolve().parents[2]
ASSETS = Path.home() / "mls-rag-assets" / "ann-cache"
CORPORA = {"24k": RAG / "data/rag_corpus/embeddings/corpus_enc_a.npy"}
PIN_ENV = {"FAISS_SIMD_LEVEL": "AVX2", "OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OMP_PROC_BIND": "true"}


def graph_hash(index):
    import faiss
    return hashlib.sha1(faiss.vector_to_array(index.hnsw.neighbors).tobytes()).hexdigest()[:12]


def cached_hash(corpus, M, efc):
    import faiss
    path = ASSETS / corpus / f"hnsw-corpus_enc_a-{M}-{efc}.faiss"
    return graph_hash(faiss.read_index(str(path))) if path.exists() else ""


# ----------------------------------------------------------------------------- child: one build
def child(cfg):
    import faiss
    faiss.omp_set_num_threads(cfg["threads"])
    X = np.ascontiguousarray(np.load(cfg["embeddings"]), dtype=np.float32)
    Q = np.ascontiguousarray(np.load(cfg["queries"]), dtype=np.float32)
    gt = np.load(cfg["gt"])[:, :10]
    index = faiss.IndexHNSWFlat(X.shape[1], cfg["M"], faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = cfg["efc"]
    t = time.perf_counter()
    index.add(X)
    build_s = time.perf_counter() - t
    out = dict(build_s=build_s, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               omp_threads=faiss.omp_get_max_threads(), simd_level=faiss.SIMDConfig.get_level_name(),
               faiss_version=faiss.__version__, max_level=index.hnsw.max_level,
               graph_sha1=graph_hash(index))
    faiss.omp_set_num_threads(1)
    for ef in (32, 128):
        index.hnsw.efSearch = ef
        _, I = index.search(Q, 10)
        out[f"recall@10_ef{ef}"] = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(I, gt)]))
    print("RESULT " + json.dumps(out), flush=True)


# ----------------------------------------------------------------------------- parent
def sysfs(path, default=""):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def cpu_state(cpus):
    return {f"cpu{c}": sysfs(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor") for c in cpus}


def write_env(out_dir, args, cpus_used):
    lines = [f"step: build", f"date: {datetime.now():%Y%m%dT%H%M%S}", f"host: {platform.node()}",
             f"python {sys.executable}",
             f"git: {subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=RAG, capture_output=True, text=True).stdout.strip()}",
             f"configs={' '.join(args.configs)} rounds={args.rounds} corpora={' '.join(args.corpora)} seed={args.seed}",
             f"governors {cpu_state(cpus_used)}",
             f"no_turbo={sysfs('/sys/devices/system/cpu/intel_pstate/no_turbo', 'n/a')}",
             f"power_profile={subprocess.run(['powerprofilesctl', 'get'], capture_output=True, text=True).stdout.strip() if Path('/usr/bin/powerprofilesctl').exists() else 'n/a'}",
             f"on_ac={','.join(sysfs(p) for p in map(str, Path('/sys/class/power_supply').glob('*/online')))}",
             f"thp={sysfs('/sys/kernel/mm/transparent_hugepage/enabled')}",
             f"loadavg_start={sysfs('/proc/loadavg')}",
             " ".join(f"{k}={v}" for k, v in PIN_ENV.items()),
             "--- lscpu", subprocess.run(["lscpu"], capture_output=True, text=True).stdout]
    (out_dir / "env.txt").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["1:2", "2:0,2"], help="THREADS:CPUS (taskset list)")
    ap.add_argument("--corpora", nargs="+", default=["24k"], choices=list(CORPORA))
    ap.add_argument("--rounds", type=int, default=5, help="builds per (corpus, config)")
    ap.add_argument("--settings", type=Path, default=RAG / "results/ann-exp-data/consolidated/settings.json")
    ap.add_argument("--out", type=Path, default=RAG / "results/ann-exp-data/laptop/ann/build")
    ap.add_argument("--seed", type=int, default=0, help="seed for the interleaving order only")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        return child(json.loads(a.child))

    settings = json.loads(a.settings.read_text())
    configs = []
    for spec in a.configs:
        t, cpus = spec.split(":")
        configs.append((int(t), cpus))
    cpus_used = sorted({int(c) for _, cpus in configs for c in cpus.replace("-", ",").split(",")})
    pinned = set(os.environ.get("BENCH_CPUS", os.environ.get("BENCH_CPU", "")).split())
    if not {str(c) for c in cpus_used} <= pinned:
        print(f"[build] WARNING: cpus {cpus_used} not all frequency-pinned (BENCH_CPUS={sorted(pinned)}); "
              "run under with_pinned_freq.sh with BENCH_CPUS listing every cpu", file=sys.stderr)

    out_dir = a.out / f"{platform.node()}-{datetime.now():%Y%m%dT%H%M%S}"
    out_dir.mkdir(parents=True)
    write_env(out_dir, a, cpus_used)
    print(f"[build] writing {out_dir}", flush=True)

    cached = {c: cached_hash(c, settings[c]["M"], settings[c]["efc"]) for c in a.corpora}
    rng = random.Random(a.seed)
    cells = [(c, t, cpus) for c in a.corpora for t, cpus in configs]
    rows = []
    for rnd in range(a.rounds):
        order = cells[:]
        rng.shuffle(order)
        for corpus, threads, cpus in order:
            cfg = dict(embeddings=str(CORPORA[corpus]), queries=str(ASSETS / "queries/course.npy"),
                       gt=str(ASSETS / corpus / "gt-course-k100.npy"), threads=threads,
                       M=settings[corpus]["M"], efc=settings[corpus]["efc"])
            env = {**os.environ, **PIN_ENV, "OMP_NUM_THREADS": str(threads)}
            cmd = ["taskset", "-c", cpus, sys.executable, __file__, "--child", json.dumps(cfg)]
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=RAG)
            res = next((json.loads(l[7:]) for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
            if p.returncode or res is None:
                sys.exit(f"[build] child failed ({corpus}, {threads}t on {cpus}):\n{p.stderr[-3000:]}")
            row = dict(timestamp=datetime.now().isoformat(timespec="seconds"), host=platform.node(),
                       corpus=corpus, M=cfg["M"], efc=cfg["efc"], threads=threads, cpus=cpus, round=rnd,
                       loadavg_1m=sysfs("/proc/loadavg").split()[0], **res,
                       same_as_cached=res["graph_sha1"] == cached[corpus])
            rows.append(row)
            print(f"[build] round {rnd} {corpus} {threads}t@{cpus}: {res['build_s']:.2f}s "
                  f"rss {res['peak_rss_mb']:.0f} MB recall@10 ef32 {res['recall@10_ef32']:.4f} "
                  f"graph {res['graph_sha1']} (cached: {row['same_as_cached']}) load {row['loadavg_1m']}", flush=True)
            with open(out_dir / "rows.csv", "w", newline="") as f:  # rewrite each time: partial runs survive
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    with open(out_dir / "env.txt", "a") as f:
        f.write(f"\nloadavg_end={sysfs('/proc/loadavg')}\n")
    print(f"[build] done: {len(rows)} builds -> {out_dir / 'rows.csv'}")


if __name__ == "__main__":
    main()
