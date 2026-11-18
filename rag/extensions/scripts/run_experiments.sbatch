#!/bin/bash
# The bench.py / exp.py steps of the ANN experiments that laptop_rerun.py doesn't run.
#
#   sbatch --gres=gpu:1 rag/extensions/scripts/run_experiments.sbatch    # from the repo root
#
# Steps (each writes $RESULTS_DIR/<step>/<host>-<time>/: rows*.csv, env.txt, job.log.gz, and
# FAILED if it failed; resume a failed job with STEPS="<step> ..."):
#   1-setup      course query vectors + exact top-100 ground truth into the cache (exp.py prep);
#                every later step reads them
#   3-stage1     recall/latency curve: exact_dense, faiss_exact and HNSW at every ef
#   4-reorder    HNSW at ef95/ef99 under identity, 5 random seeds, BFS and Gorder (wall clock)
#   perf         the 4-reorder cells plus both exact baselines under `perf stat` (laptop only:
#                the cluster blocks perf); not in the default STEPS
#   11-pipeline  end-to-end RAG pipeline, exact_dense vs HNSW (needs a GPU)
# Environment check only:  STEPS=env bash rag/extensions/scripts/run_experiments.sbatch
#
# Graph settings (M, efC, ef95, ef99, Gorder w) are read from results/ann-exp-data/consolidated/
# settings.json (retune.py). On the laptop, after copying $PERSIST_INDEX_PATH from the cluster so
# both machines time the same saved graphs:
#   STEPS="3-stage1 4-reorder perf" RESULTS_DIR=$PWD/rag/results/ann-exp-data/laptop/ann \
#       rag/extensions/scripts/with_pinned_freq.sh bash rag/extensions/scripts/run_experiments.sbatch
# Then fold the new runs into consolidated/ with results/ann-exp-data/consolidate.py.
#
# Where things go:
#   RESULTS_DIR         rag/results/ann-exp-data/ann/ by default (the cluster layout consolidate.py reads)
#   PERSIST_INDEX_PATH  ~/mls-rag-assets/ann-cache/: saved indexes, permutations, ground truth,
#                       query sets; kept between runs so every step reuses the same graphs
#   SCRATCH_DIR         /disk/scratch/$USER/ann on cluster nodes (else /tmp/$USER/ann): per-cell
#                       logs, deleted when the job ends
# Data paths can be overridden: C24 (corpus_enc_a.npy, ids json next to it), COURSE_QV (query
# vectors, ids json next to it), QRELS (course qrels_dev.tsv).
#
# Timed steps run one cell at a time on one pinned core.
#SBATCH -p Teaching
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=ann-experiments-%j.out

set -euo pipefail
# Jobs inherit the login node's soft virtual-memory cap (512000 KB on the Teaching cluster), and
# loading faiss's native library alone needs more than that: it segfaults on import. The hard
# limit is unlimited, so lift the soft one for this job and everything it starts.
ulimit -v "$(ulimit -Hv)"

STEPS="${STEPS:-1-setup 3-stage1 4-reorder 11-pipeline}"
# ---------------------------------------------------------------- environment
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../.."
fi
RESULTS_DIR="${RESULTS_DIR:-$PWD/rag/results/ann-exp-data/ann}"
PERSIST_INDEX_PATH="$HOME/mls-rag-assets/ann-cache"
if [[ -d /disk/scratch ]]; then SCRATCH_DIR="/disk/scratch/$USER/ann"; else SCRATCH_DIR="/tmp/$USER/ann"; fi
# shellcheck disable=SC1091
[[ -n "${VIRTUAL_ENV:-}" ]] || source "${RAG_VENV:-$HOME/mls-rag-env}/bin/activate"
export PYTHONUNBUFFERED=1 PERSIST_INDEX_PATH
export FAISS_SIMD_LEVEL="${FAISS_SIMD_LEVEL:-AVX2}"   # one FAISS code path on every machine
# Same for numpy's OpenBLAS, which otherwise picks its kernels from the CPU (AVX-512 on Sapphire
# Rapids nodes): pin Haswell (AVX2) so the numpy exact baseline runs the same code everywhere.
export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}"
PY="$(command -v python)"
BENCH="$PY rag/extensions/scripts/bench.py"
EXP="$PY rag/extensions/scripts/exp.py"

C24="${C24:-rag/data/rag_corpus/embeddings/corpus_enc_a.npy}"
COURSE_QV="${COURSE_QV:-$HOME/mls-rag-assets/query_vectors/queries_enc_a.npy}"
QRELS="${QRELS:-rag/data/rag_corpus/qrels_dev.tsv}"
CORPUS=24k
QCOURSE="$PERSIST_INDEX_PATH/queries/course.npy"
SETTINGS="$PWD/rag/results/ann-exp-data/consolidated/settings.json"

EFS="${EFS:-16 32 64 128 256 512}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2 3 4}"
ROUNDS="${ROUNDS:-3}"; ROUNDS_REORDER="${ROUNDS_REORDER:-5}"; REPS="${REPS:-5}"; NQ="${NQ:-}"
# The two exact baselines HNSW is compared with: the course's ExactDenseRetriever itself (the
# current method) and the same search in FAISS (IndexFlatIP), which separates library from algorithm.
EXACT="exact_dense faiss_exact"

TS="$(date +%Y%m%dT%H%M%S)"
WORK="$SCRATCH_DIR/$TS-$$"
mkdir -p "$WORK" "$RESULTS_DIR" "$PERSIST_INDEX_PATH"
trap 'rm -rf "$WORK"' EXIT

has_step() { [[ "$STEPS" == env || " $STEPS " == *" $1 "* ]]; }
fail() { echo "!! $*" >&2; exit 1; }

# ---------------------------------------------------------------- env check: fail fast
check_env() {
    echo "== environment check on $(hostname -s)"
    echo "  results $RESULTS_DIR | cache $PERSIST_INDEX_PATH | scratch $SCRATCH_DIR"
    echo "  cpu: $(lscpu | sed -n 's/^Model name: *//p') | cpus allowed $(nproc) | vmem limit $(ulimit -v)"
    "$PY" -X faulthandler - <<'EOF' || fail "python environment check failed (a crash prints the failing line above)"
import os, sys
import numpy, scipy, faiss
assert tuple(map(int, faiss.__version__.split(".")[:2])) >= (1, 15), f"faiss {faiss.__version__} < 1.15"
lvl = faiss.SIMDConfig.get_level_name()
assert lvl == os.environ["FAISS_SIMD_LEVEL"], f"FAISS_SIMD_LEVEL={os.environ['FAISS_SIMD_LEVEL']} but faiss runs {lvl}"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
print(f"  python {sys.version.split()[0]}, numpy {numpy.__version__}, scipy {scipy.__version__}, faiss {faiss.__version__} ({lvl}), core pinning ok")
EOF
    local f
    for f in "$C24" "$COURSE_QV"; do [[ -f "$f" && -f "${f%.npy}_ids.json" ]] || fail "vectors or ids json missing: $f"; done
    [[ -f "$QRELS" ]] || fail "qrels missing: $QRELS (set QRELS)"
    [[ -f "$SETTINGS" ]] || fail "$SETTINGS missing"
    echo "  data files ok"

    if has_step 11-pipeline; then
        if nvidia-smi -L >/dev/null 2>&1; then
            export PATH=/opt/cuda-12.8.0/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
            export RAG_ASR_PYTHON="${RAG_ASR_PYTHON:-${CONDA_BASE:-$HOME/miniconda3}/envs/mls-asr/bin/python}"
            [[ -x "$RAG_ASR_PYTHON" ]] || fail "ASR python not found: $RAG_ASR_PYTHON (set RAG_ASR_PYTHON)"
            if [[ "${ASR_CUDA:-all}" != none ]]; then
                command -v nvcc >/dev/null || fail "nvcc not found (needed to build the hw1 ASR kernels; or set ASR_CUDA=none)"
            fi
            grep -q '^ELM_API_KEY=.' rag/.env 2>/dev/null || [[ -n "${ELM_API_KEY:-}" ]] || fail "ELM_API_KEY not set (rag/.env)"
            echo "  $(nvidia-smi -L | head -1); ASR $RAG_ASR_PYTHON; ELM key set"
        else
            echo "  no GPU visible: skipping 11-pipeline (submit with --gres=gpu:1 to include it)"; STEPS="${STEPS/11-pipeline/}"
        fi
    fi
    if [[ " $STEPS " == *" perf "* ]]; then  # only when asked for: the cluster blocks perf
        PERF="${PERF:-$(command -v perf || true)}"
        [[ -n "$PERF" ]] || fail "perf not found (install linux-tools, or set PERF=/path/to/perf)"
        local paranoid; paranoid="$(cat /proc/sys/kernel/perf_event_paranoid)"
        (( paranoid <= 2 )) || fail "perf_event_paranoid=$paranoid: user-space counters blocked (needs <= 2)"
        "$PERF" stat -x, -e cycles:u,instructions:u "$PY" -c 'sum(range(10**6))' 2>&1 | grep -q "cycles" \
            || fail "perf stat could not count cycles"
        export PERF
        echo "  perf ($PERF), perf_event_paranoid=$paranoid: counters ok"
    fi
    if [[ -z "${SLURM_JOB_ID:-}" && -z "${BENCH_CPU:-}" ]]; then
        echo "  !! not under Slurm and not via with_pinned_freq.sh: laptop clock not pinned"
    fi
    echo "== steps: $(echo $STEPS)"
}

# ---------------------------------------------------------------- helpers
setting() {  # setting <key>
    "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])' "$SETTINGS" "$CORPUS" "$1" \
        || fail "no '$1' for $CORPUS in $SETTINGS"
}
orders() { local s o="identity"; for s in $RANDOM_SEEDS; do o="$o random:$s"; done; echo "$o bfs gorder:$(setting gorder_w)"; }
bench() {  # bench <out csv> [bench args...]
    local csv="$1"; shift
    $BENCH --name "$CORPUS" --embeddings "$C24" --out "$OUT/$csv" --log-dir "$WORK/logs-$STEP" \
        ${NQ:+--nq "$NQ"} --tag "$STEP" "$@"
}

write_env() {
    {
        echo "step: $STEP"; echo "date: $TS"; echo "host: $(hostname)"
        echo "slurm: job=${SLURM_JOB_ID:-} partition=${SLURM_JOB_PARTITION:-} cpus=${SLURM_CPUS_PER_TASK:-}"
        echo "git: $(git rev-parse HEAD 2>/dev/null) on $(git branch --show-current 2>/dev/null)"
        echo "git status:"; git status --short 2>/dev/null | sed 's/^/  /'
        "$PY" -c 'import sys, faiss, numpy, scipy; print("python", sys.executable, "| faiss", faiss.__version__, faiss.SIMDConfig.get_level_name(), "| numpy", numpy.__version__, "| scipy", scipy.__version__)'
        echo "RESULTS_DIR=$RESULTS_DIR PERSIST_INDEX_PATH=$PERSIST_INDEX_PATH SCRATCH_DIR=$SCRATCH_DIR"
        echo "FAISS_SIMD_LEVEL=$FAISS_SIMD_LEVEL OPENBLAS_CORETYPE=$OPENBLAS_CORETYPE BENCH_CPU=${BENCH_CPU:-}"
        echo "C24=$C24 COURSE_QV=$COURSE_QV QRELS=$QRELS"
        echo "EFS=$EFS RANDOM_SEEDS=$RANDOM_SEEDS ROUNDS=$ROUNDS ROUNDS_REORDER=$ROUNDS_REORDER REPS=$REPS NQ=$NQ"
        echo "settings.json:"; cat "$SETTINGS"
        echo "--- lscpu"; lscpu
    } > "$OUT/env.txt"
}

# ---------------------------------------------------------------- steps
step_1_setup() {
    $EXP prep --cache "$PERSIST_INDEX_PATH" --corpus "$CORPUS=$C24" --course-queries "$COURSE_QV" --out "$OUT"
}

step_3_stage1() {
    bench rows.csv --queries "$QCOURSE" --index $EXACT hnsw --M "$(setting M)" --efc "$(setting efc)" \
        --ef $EFS --rounds "$ROUNDS" --reps "$REPS"
}

step_4_reorder() {
    bench rows.csv --queries "$QCOURSE" --index hnsw --M "$(setting M)" --efc "$(setting efc)" \
        --ef "$(setting ef95)" "$(setting ef99)" --order $(orders) --rounds "$ROUNDS_REORDER" --reps "$REPS"
}

# Real hardware counters (laptop): the same cells as 4-reorder plus both exact baselines, under
# perf stat, warm cache, net of a load + warm-up twin. Counters vary run to run, so several rounds.
step_perf() {
    bench rows.csv --mode perf --queries "$QCOURSE" --index $EXACT hnsw --M "$(setting M)" --efc "$(setting efc)" \
        --ef "$(setting ef95)" "$(setting ef99)" --order $(orders) --rounds "$ROUNDS" --reps "$REPS"
}

step_11_pipeline() {
    # ASR runs on the hw1 CUDA kernels (ASR_CUDA, default all), compiled per GPU type into
    # hw1-asr/build/sm_XX/; build them for this node's GPU (a no-op when already built).
    # ASR_CUDA=none uses the PyTorch reference instead and needs no build.
    if [[ "${ASR_CUDA:-all}" != none ]]; then
        make -s -C "${RAG_ASR_ROOT:-hw1-asr}" lib
    fi
    # Every dev query (PIPE_N limits it), exact_dense vs HNSW at the ef95 setting.
    $EXP pipeline --cache "$PERSIST_INDEX_PATH" --settings "$SETTINGS" --device "${DEVICE:-cpu}" --qrels "$QRELS" \
        ${PIPE_N:+--n "$PIPE_N"} --out "$OUT"
}

# One step: its own results folder and log; stops the job on failure.
run_step() {
    STEP="$1"
    OUT="$RESULTS_DIR/$STEP/$(hostname -s)-$TS"
    mkdir -p "$OUT" "$WORK/logs-$STEP"
    echo; echo "== $STEP -> $OUT"
    write_env
    set +e   # the step runs in its own subshell with -e on; collect its exit code here
    ( set -e; "step_${STEP//-/_}" ) 2>&1 | tee "$WORK/$STEP.log"
    local rc="${PIPESTATUS[0]}"
    set -e
    gzip -c "$WORK/$STEP.log" > "$OUT/job.log.gz"
    rm -rf "$WORK/logs-$STEP"
    if (( rc != 0 )); then
        echo "exit code $rc" > "$OUT/FAILED"
        fail "$STEP failed (exit $rc); fix, then resume with STEPS=\"$STEP ...\""
    fi
}

check_env
[[ "$STEPS" == env ]] && exit 0
for s in $STEPS; do run_step "$s"; done
echo; echo "== all done: $RESULTS_DIR"
