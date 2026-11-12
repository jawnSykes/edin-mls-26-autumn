#!/bin/bash
# The whole ANN experiment sequence (rag/docs/ann-index-brief.md) in one job.
#
#   sbatch rag/extensions/scripts/run_experiments.sbatch                 # from the repo root
#   sbatch --gres=gpu:1 rag/extensions/scripts/run_experiments.sbatch    # also runs 11-pipeline
#
# Steps run in order; each writes $RESULTS_DIR/<step>/<host>-<time>/ (rows*.csv, env.txt,
# job.log.gz, and FAILED if it failed). Step 2 writes $RESULTS_DIR/settings.json, which later
# steps read, so a failed or timed-out job resumes from any step:
#   sbatch --export=ALL,STEPS="4-reorder 5-cachesweep diag report" rag/extensions/scripts/run_experiments.sbatch
# Environment check only:  STEPS=env bash rag/extensions/scripts/run_experiments.sbatch
#
# Laptop (wall-clock steps), after copying $PERSIST_INDEX_PATH and settings.json from the
# cluster so both machines time the same saved graphs:
#   STEPS="3-stage1 4-reorder 10-queryset" rag/extensions/scripts/with_pinned_freq.sh \
#       bash rag/extensions/scripts/run_experiments.sbatch
#
# Where things go (set below, same on the cluster and the laptop):
#   RESULTS_DIR         rag/results/ann/ in this checkout (gitignored): rows, env, logs, report
#   PERSIST_INDEX_PATH  ~/mls-rag-assets/ann-cache/: saved indexes, permutations, ground truth,
#                       query sets; kept between runs so every step reuses the same graphs
#   SCRATCH_DIR         /disk/scratch/$USER/ann on cluster nodes (else /tmp/$USER/ann):
#                       per-cell logs and Cachegrind files, deleted when the job ends
# Data paths can be overridden: C24, C58 (corpus_enc_a.npy, ids json next to each), COURSE_QV,
# FIQA_QV (query vectors, ids json next to each), QRELS (course qrels_dev.tsv).
#
# Timed steps run one cell at a time on one pinned core. Only the Cachegrind sweep (step 5)
# runs JOBS sweeps in parallel: simulated miss counts don't depend on contention.
#SBATCH -p Teaching
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=ann-experiments-%j.out

set -euo pipefail

STEPS="${STEPS:-1-setup 2-tune 3-stage1 4-reorder 5-cachesweep 8-simd 9-quality 10-queryset 11-pipeline diag report}"
# ---------------------------------------------------------------- environment
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../.."
fi
RESULTS_DIR="$PWD/rag/results/ann"
PERSIST_INDEX_PATH="$HOME/mls-rag-assets/ann-cache"
if [[ -d /disk/scratch ]]; then SCRATCH_DIR="/disk/scratch/$USER/ann"; else SCRATCH_DIR="/tmp/$USER/ann"; fi
# Smoke test (smoke_test.sbatch): every step on tiny grids, in separate folders so the real
# results and cache stay clean. A smoke run that includes 1-setup starts from empty results.
if [[ -n "${SMOKE:-}" ]]; then
    RESULTS_DIR="$PWD/rag/results/ann-smoke"
    PERSIST_INDEX_PATH="$HOME/mls-rag-assets/ann-cache-smoke"
    if [[ " $STEPS " == *" 1-setup "* ]]; then rm -rf "$RESULTS_DIR"; fi
    MS=16 EFCS=40 EFS="16 64 256" GORDER_WS=3 LLS="1 8" RANDOM_SEEDS_TIME="0 1" RANDOM_SEEDS_CG=0
    ROUNDS=1 ROUNDS_REORDER=1 REPS=1 TUNE_REPS=1 NQ=20 GW_NQ=20 SMOKE_NQ=5 PIPE_N=2
fi
# shellcheck disable=SC1091
[[ -n "${VIRTUAL_ENV:-}" ]] || source "${RAG_VENV:-$HOME/mls-rag-env}/bin/activate"
export PYTHONUNBUFFERED=1 PERSIST_INDEX_PATH
export FAISS_SIMD_LEVEL="${FAISS_SIMD_LEVEL:-AVX2}"   # one code path everywhere (see the brief)
PY="$(command -v python)"
BENCH="$PY rag/extensions/scripts/bench.py"
EXP="$PY rag/extensions/scripts/exp.py"

C24="${C24:-rag/data/rag_corpus/embeddings/corpus_enc_a.npy}"
C58="${C58:-rag/data/fiqa_full/embeddings/corpus_enc_a.npy}"
COURSE_QV="${COURSE_QV:-$HOME/mls-rag-assets/query_vectors/queries_enc_a.npy}"
FIQA_QV="${FIQA_QV:-rag/data/fiqa_full/embeddings/queries_enc_a.npy}"
QRELS="${QRELS:-rag/data/rag_corpus/qrels_dev.tsv}"
CORPORA="${CORPORA:-24k=$C24 58k=$C58}"
QCOURSE="$PERSIST_INDEX_PATH/queries/course.npy"
QFIQA="$PERSIST_INDEX_PATH/queries/fiqa_extra.npy"
SETTINGS="$RESULTS_DIR/settings.json"

# Grids (brief defaults; override from the environment, e.g. for a smoke test)
MS="${MS:-16 32 48}"
EFCS="${EFCS:-40 200}"
EFS="${EFS:-16 32 64 128 256 512}"
RANDOM_SEEDS_TIME="${RANDOM_SEEDS_TIME:-0 1 2 3 4}"
RANDOM_SEEDS_CG="${RANDOM_SEEDS_CG:-0 1 2}"
GORDER_WS="${GORDER_WS:-3 5 10}"
LLS="${LLS:-1 2 4 8 16 32 64 128}"          # MiB
SIMDS="${SIMDS:-AVX2 AVX512 AVX512_SPR}"     # step 8
ROUNDS="${ROUNDS:-3}"; ROUNDS_REORDER="${ROUNDS_REORDER:-5}"; REPS="${REPS:-5}"
TUNE_REPS="${TUNE_REPS:-1}"; GW_NQ="${GW_NQ:-500}"; SMOKE_NQ="${SMOKE_NQ:-20}"; NQ="${NQ:-}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-4}}"

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
    "$PY" - <<'EOF' || fail "python environment check failed"
import os, sys
import numpy, scipy, faiss
assert tuple(map(int, faiss.__version__.split(".")[:2])) >= (1, 15), f"faiss {faiss.__version__} < 1.15"
lvl = faiss.SIMDConfig.get_level_name()
assert lvl == os.environ["FAISS_SIMD_LEVEL"], f"FAISS_SIMD_LEVEL={os.environ['FAISS_SIMD_LEVEL']} but faiss runs {lvl}"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
print(f"  python {sys.version.split()[0]}, numpy {numpy.__version__}, scipy {scipy.__version__}, faiss {faiss.__version__} ({lvl}), core pinning ok")
EOF
    local f
    for f in $CORPORA; do f="${f#*=}"; [[ -f "$f" && -f "${f%.npy}_ids.json" ]] || fail "corpus or its ids json missing: $f (set C24/C58)"; done
    for f in "$COURSE_QV" "$FIQA_QV"; do [[ -f "$f" && -f "${f%.npy}_ids.json" ]] || fail "query vectors or ids json missing: $f"; done
    [[ -f "$QRELS" ]] || fail "qrels missing: $QRELS (set QRELS)"
    if [[ "$STEPS" != env ]] && ! has_step 2-tune; then
        [[ -f "$SETTINGS" ]] || fail "$SETTINGS missing: run 2-tune first"
    fi
    echo "  data files ok"

    if has_step 8-simd; then
        if grep -qw avx512f /proc/cpuinfo; then
            local lv got
            for lv in $SIMDS; do
                got="$(FAISS_SIMD_LEVEL=$lv "$PY" -c 'import faiss; print(faiss.SIMDConfig.get_level_name())')"
                [[ "$got" == "$lv" ]] || fail "FAISS_SIMD_LEVEL=$lv gives $got (fix SIMDS)"
            done
            echo "  SIMD levels ok: $SIMDS"
        else
            echo "  no AVX-512 on this CPU: skipping 8-simd"; STEPS="${STEPS/8-simd/}"
        fi
    fi

    if has_step 1-setup || has_step 2-tune || has_step 5-cachesweep; then
        VALGRIND="${VALGRIND:-$(command -v valgrind || true)}"
        [[ -n "$VALGRIND" && -x "$VALGRIND" ]] || fail "valgrind not found (set VALGRIND=/path/to/valgrind)"
        D1="${D1:-$(lscpu -C=NAME,ONE-SIZE,WAYS -B | awk '$1=="L1d"{print $2","$3",64"}')}"
        export VALGRIND D1
        "$PY" -c 'import sys; s, w, l = map(int, sys.argv[1].split(",")); n = s // (w * l); assert n & (n - 1) == 0' "$D1" \
            || fail "L1d $D1 has a non-power-of-two set count; set D1=size,ways,64 by hand"
        "$VALGRIND" --tool=cachegrind --cache-sim=yes --D1="$D1" --LL=4194304,16,64 --cachegrind-out-file=/dev/null \
            "$PY" -c 'import numpy as np, faiss
i = faiss.IndexHNSWFlat(16, 8); i.add(np.random.rand(500, 16).astype("float32"))
i.search(np.random.rand(5, 16).astype("float32"), 5)' > "$WORK/preflight.log" 2>&1 || true
        grep -q "LLd misses" "$WORK/preflight.log" || { tail -20 "$WORK/preflight.log" >&2; fail "cachegrind could not run FAISS"; }
        echo "  $("$VALGRIND" --version) ($VALGRIND), D1=$D1: cachegrind runs FAISS ok"
    fi

    if has_step 11-pipeline; then
        if nvidia-smi -L >/dev/null 2>&1; then
            export PATH=/opt/cuda-12.8.0/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
            export RAG_ASR_PYTHON="${RAG_ASR_PYTHON:-${CONDA_BASE:-$HOME/miniconda3}/envs/mls-asr/bin/python}"
            [[ -x "$RAG_ASR_PYTHON" ]] || fail "ASR python not found: $RAG_ASR_PYTHON (set RAG_ASR_PYTHON)"
            grep -q '^ELM_API_KEY=.' rag/.env 2>/dev/null || [[ -n "${ELM_API_KEY:-}" ]] || fail "ELM_API_KEY not set (rag/.env)"
            echo "  $(nvidia-smi -L | head -1); ASR $RAG_ASR_PYTHON; ELM key set"
        else
            echo "  no GPU visible: skipping 11-pipeline (submit with --gres=gpu:1 to include it)"; STEPS="${STEPS/11-pipeline/}"
        fi
    fi
    if [[ -z "${SLURM_JOB_ID:-}" && -z "${BENCH_CPU:-}" ]]; then
        echo "  !! not under Slurm and not via with_pinned_freq.sh: laptop clock not pinned"
    fi
    echo "== steps: $(echo $STEPS)"
}

# ---------------------------------------------------------------- helpers
setting() {  # setting <corpus> <key>
    "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])' "$SETTINGS" "$1" "$2" \
        || fail "no '$2' for $1 in $SETTINGS"
}
randoms() { for s in $1; do printf 'random:%s ' "$s"; done; }
each_corpus() { for spec in $CORPORA; do "$1" "${spec%%=*}" "${spec#*=}"; done; }
bench() {  # bench <name> <embeddings> <out csv> [bench args...]
    local name="$1" emb="$2" csv="$3"; shift 3
    $BENCH --name "$name" --embeddings "$emb" --out "$OUT/$csv" --log-dir "$WORK/logs-$STEP" \
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
        echo "FAISS_SIMD_LEVEL=$FAISS_SIMD_LEVEL BENCH_CPU=${BENCH_CPU:-} JOBS=$JOBS"
        [[ -n "${VALGRIND:-}" ]] && echo "valgrind: $VALGRIND ($("$VALGRIND" --version)) D1=$D1"
        echo "CORPORA=$CORPORA"; echo "COURSE_QV=$COURSE_QV FIQA_QV=$FIQA_QV QRELS=$QRELS"
        echo "MS=$MS EFCS=$EFCS EFS=$EFS LLS=$LLS GORDER_WS=$GORDER_WS SIMDS=$SIMDS"
        echo "RANDOM_SEEDS_TIME=$RANDOM_SEEDS_TIME RANDOM_SEEDS_CG=$RANDOM_SEEDS_CG"
        echo "ROUNDS=$ROUNDS ROUNDS_REORDER=$ROUNDS_REORDER REPS=$REPS TUNE_REPS=$TUNE_REPS GW_NQ=$GW_NQ NQ=$NQ"
        if [[ -f "$SETTINGS" ]]; then echo "settings.json:"; cat "$SETTINGS"; fi
        echo "--- lscpu"; lscpu
    } > "$OUT/env.txt"
}

# ---------------------------------------------------------------- steps
step_1_setup() {
    $EXP prep --cache "$PERSIST_INDEX_PATH" --corpus $CORPORA --course-queries "$COURSE_QV" \
        --fiqa-queries "$FIQA_QV" --out "$OUT"
    # Cachegrind smoke test: the two LL sizes must give different misses. Also times Gorder once.
    local spec="${CORPORA%% *}"
    NQ="$SMOKE_NQ" bench "${spec%%=*}" "${spec#*=}" rows-smoke.csv --mode cachegrind --queries "$QCOURSE" \
        --index hnsw --M 16 --efc 40 --ef 32 --order identity gorder:5 --LL $((1 << 20)) $((8 << 20))
}

tune_grid() { bench "$1" "$2" rows.csv --queries "$QFIQA" --index hnsw --M $MS --efc $EFCS --ef $EFS --rounds 1 --reps "$TUNE_REPS"; }
tune_gorder() {
    local orders="identity" w; for w in $GORDER_WS; do orders="$orders gorder:$w"; done
    NQ="$GW_NQ" bench "$1" "$2" rows-cg.csv --mode cachegrind --queries "$QFIQA" --index hnsw \
        --M "$(setting "$1" M)" --efc "$(setting "$1" efc)" --ef "$(setting "$1" ef95)" --order $orders --LL $((4 << 20))
}
step_2_tune() {
    each_corpus tune_grid
    $EXP pick --rows "$OUT/rows.csv" --settings "$SETTINGS"
    each_corpus tune_gorder
    $EXP pick --rows "$OUT/rows.csv" "$OUT/rows-cg.csv" --settings "$SETTINGS"
    cp "$SETTINGS" "$OUT/settings.json"
}

stage1() {
    bench "$1" "$2" rows.csv --queries "$QCOURSE" --index numpy flat hnsw --M "$(setting "$1" M)" \
        --efc "$(setting "$1" efc)" --ef $EFS --rounds "$ROUNDS" --reps "$REPS"
}
step_3_stage1() { each_corpus stage1; }

orders() {  # orders <corpus> <random seeds>
    echo "identity $(randoms "$2") bfs gorder:$(setting "$1" gorder_w)"
}
reorder() {
    bench "$1" "$2" rows.csv --queries "$QCOURSE" --index hnsw --M "$(setting "$1" M)" --efc "$(setting "$1" efc)" \
        --ef "$(setting "$1" ef95)" "$(setting "$1" ef99)" --order $(orders "$1" "$RANDOM_SEEDS_TIME") \
        --rounds "$ROUNDS_REORDER" --reps "$REPS"
}
step_4_reorder() { each_corpus reorder; }

sweep_one() {  # sweep_one <name> <emb> <LL MiB>: own CSV, so parallel sweeps never share a file
    bench "$1" "$2" "rows-$1-LL$3.csv" --mode cachegrind --queries "$QCOURSE" --index flat hnsw \
        --M "$(setting "$1" M)" --efc "$(setting "$1" efc)" --ef "$(setting "$1" ef95)" "$(setting "$1" ef99)" \
        --order $(orders "$1" "$RANDOM_SEEDS_CG") --LL $(($3 << 20))
}
perms_only() {  # computes and checks any missing permutations once, before the sweeps go parallel
    NQ=5 bench "$1" "$2" ../perms.csv --queries "$QCOURSE" --index hnsw --M "$(setting "$1" M)" --efc "$(setting "$1" efc)" \
        --ef "$(setting "$1" ef95)" --order $(orders "$1" "$RANDOM_SEEDS_CG") --rounds 1 --reps 1 > /dev/null
    rm -f "$OUT/../perms.csv"
}
step_5_cachesweep() {
    each_corpus perms_only
    local spec ll t i=0 bad=0 tasks=()
    for spec in $CORPORA; do for ll in $LLS; do tasks+=("${spec%%=*} ${spec#*=} $ll"); done; done
    echo "${#tasks[@]} sweeps (corpus x LL size), $JOBS at a time"
    for t in "${tasks[@]}"; do
        while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || true; done
        # shellcheck disable=SC2086
        ( sweep_one $t && touch "$WORK/ok-$i" ) > "$WORK/sweep-$i.log" 2>&1 &
        i=$((i + 1))
    done
    wait || true
    for ((i = 0; i < ${#tasks[@]}; i++)); do
        echo "--- sweep ${tasks[$i]}"; cat "$WORK/sweep-$i.log"
        [[ -f "$WORK/ok-$i" ]] || { echo "!! failed: ${tasks[$i]}"; bad=1; }
    done
    (( bad == 0 )) || fail "some cache-sweep runs failed (see above)"
}

simd() {
    bench "$1" "$2" rows.csv --queries "$QCOURSE" --index flat hnsw --M "$(setting "$1" M)" --efc "$(setting "$1" efc)" \
        --ef "$(setting "$1" ef95)" --simd $SIMDS --rounds "$ROUNDS_REORDER" --reps "$REPS"
}
step_8_simd() { each_corpus simd; }

step_9_quality() {
    $EXP quality --cache "$PERSIST_INDEX_PATH" --corpus $CORPORA --queries "$QCOURSE" --qrels "$QRELS" \
        --settings "$SETTINGS" --out "$OUT"
}

queryset() {
    bench "$1" "$2" rows.csv --queries "$QFIQA" --index numpy flat hnsw --M "$(setting "$1" M)" --efc "$(setting "$1" efc)" \
        --ef "$(setting "$1" ef95)" "$(setting "$1" ef99)" --order identity bfs "gorder:$(setting "$1" gorder_w)" \
        --rounds "$ROUNDS" --reps "$TUNE_REPS"
}
step_10_queryset() { each_corpus queryset; }

step_11_pipeline() { $EXP pipeline --cache "$PERSIST_INDEX_PATH" --settings "$SETTINGS" --device "${DEVICE:-cpu}" --n "${PIPE_N:-20}" --out "$OUT"; }

step_diag() {
    local names="" spec; for spec in $CORPORA; do names="$names ${spec%%=*}"; done
    $EXP diag --cache "$PERSIST_INDEX_PATH" --names $names --settings "$SETTINGS" --out "$OUT"
}

step_report() { $EXP report --results "$RESULTS_DIR"; }

# One step: its own results folder and log; stops the job on failure.
run_step() {
    STEP="$1"
    OUT="$RESULTS_DIR/$STEP/$(hostname -s)-$TS"
    [[ "$STEP" == report ]] && OUT="$RESULTS_DIR/report"
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
