#!/bin/bash
# Collect everything the hand-in needs into one Markdown file, in one job on one GPU:
# the GPU name, device facts, `make test`, `./run.sh --profile`, and one "Try this"
# experiment per task. From hw1-asr, on the head node:
#
#     sbatch --time=00:15:00 handin.sbatch smoke   # quick sanity check -> REPORT-smoke.md
#     sbatch handin.sbatch                         # full run           -> REPORT.md
#
# Experiments are built from sed-edited copies of csrc/ in build/experiments/, so the
# real csrc/ and build/sm_XX are never touched.
#SBATCH -p Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=handin-%j.out

set -u
MODE="${1:-full}"
if [[ "$MODE" != full && "$MODE" != smoke ]]; then
    echo "Usage: sbatch handin.sbatch [smoke]" >&2
    exit 2
fi

export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
EXP=build/experiments
mkdir -p "$EXP"

if [[ "$MODE" == smoke ]]; then
    OUT=REPORT-smoke.md
    TAG=" (smoke)"
else
    OUT=REPORT.md
    TAG=""
fi
TMP=$(mktemp "$EXP/report.XXXXXX")

# --- helpers -----------------------------------------------------------------

note() { printf '%s\n\n' "$*" >> "$TMP"; }

# section "Title" cmd args...  -> runs cmd, appends its output and exit status
section() {
    local title="$1"; shift
    echo ">>> $title$TAG"                      # progress in handin-JOBID.out
    {
        printf '## %s%s\n\n' "$title" "$TAG"
        printf '`$ %s`\n\n```text\n' "$*"
    } >> "$TMP"
    "$@" >> "$TMP" 2>&1
    local status=$?
    printf '```\n\nexit status: %d\n\n' "$status" >> "$TMP"
    return 0
}

# make_variant NAME FILE SED_EXPR EXPECTED_LINE_REGEX -> copy of csrc/ with one edit
make_variant() {
    local name="$1" file="$2" expr="$3" expect="$4"
    rm -rf "$EXP/$name"
    mkdir -p "$EXP/$name"
    cp -r csrc "$EXP/$name/csrc"
    sed -i -E "$expr" "$EXP/$name/csrc/$file"
    printf '```diff\n' >> "$TMP"
    diff -u "csrc/$file" "$EXP/$name/csrc/$file" >> "$TMP"
    printf '```\n\n' >> "$TMP"
    if ! grep -Eq "$expect" "$EXP/$name/csrc/$file" || diff -q "csrc/$file" "$EXP/$name/csrc/$file" >/dev/null; then
        note "**EXPERIMENT SETUP FAILED**: the sed edit to \`$file\` did not apply."
        return 1
    fi
}

# run_variant NAME TESTNAME   (TESTNAME e.g. test_task1_rope)
run_variant() {
    local name="$1" test="$2" n
    n=$(sed -E 's/test_task([0-9]).*/\1/' <<< "$test")
    if [[ "$MODE" == smoke ]]; then
        section "Build $name" make -s "$EXP/$name/build/$test" SRC="$EXP/$name/csrc" BUILD="$EXP/$name/build" SM="$SM"
        section "$name: $test small" "$EXP/$name/build/$test" small
    else
        section "$name: make test$n" make -s "test$n" SRC="$EXP/$name/csrc" BUILD="$EXP/$name/build" SM="$SM"
    fi
}

# --- header ------------------------------------------------------------------

{
    printf '# hw1-asr hand-in results%s\n\n' "$TAG"
    printf -- '- Date: %s\n' "$(date -Is)"
    printf -- '- Job: %s on %s\n' "${SLURM_JOB_ID:-none}" "$(hostname)"
    printf -- '- Commit: %s (branch %s)\n' "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)"
    printf -- '- Mode: %s\n\n' "$MODE"
    printf 'Raw outputs for reference while writing the report. Before/after pairs: each\n'
    printf 'experiment below is compared with the Baseline sections, from the same node.\n\n'
} >> "$TMP"

section "GPU" nvidia-smi -L
section "Uncommitted changes in csrc/ (empty = results match the commit)" git status --short csrc/

# --- device facts ------------------------------------------------------------

cat > "$EXP/devinfo.cu" <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
  int dev = 0, v = 0;
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, dev);
  auto attr = [&](cudaDeviceAttr a) { cudaDeviceGetAttribute(&v, a, dev); return v; };
  int sms = attr(cudaDevAttrMultiProcessorCount);
  int clk_khz = attr(cudaDevAttrMemoryClockRate);
  int bus = attr(cudaDevAttrGlobalMemoryBusWidth);
  int smem_optin = attr(cudaDevAttrMaxSharedMemoryPerBlockOptin);
  int l2 = attr(cudaDevAttrL2CacheSize);
  double peak = 2.0 * clk_khz * 1e3 * (bus / 8.0) / 1e9;  // DDR: 2 transfers per clock
  std::printf("name                         %s\n", p.name);
  std::printf("compute capability           %d.%d\n", p.major, p.minor);
  std::printf("SMs                          %d\n", sms);
  std::printf("global memory                %.1f GiB\n", p.totalGlobalMem / 1073741824.0);
  std::printf("memory clock                 %d MHz\n", clk_khz / 1000);
  std::printf("memory bus width             %d bit\n", bus);
  std::printf("peak memory bandwidth        %.0f GB/s (2 x clock x bus width)\n", peak);
  std::printf("L2 cache                     %.1f MiB\n", l2 / 1048576.0);
  std::printf("shared memory per block      %zu B (default), %d B (opt-in)\n", p.sharedMemPerBlock, smem_optin);
  std::printf("shared memory per SM         %zu B\n", p.sharedMemPerMultiprocessor);
  std::printf("registers per block          %d\n", p.regsPerBlock);
  std::printf("max threads per block        %d\n", p.maxThreadsPerBlock);
  return 0;
}
EOF
nvcc -o "$EXP/devinfo" "$EXP/devinfo.cu" >/dev/null 2>&1
section "Device facts" "$EXP/devinfo"
note "On a MIG slice the bandwidth above is the whole H200's; the slice gets a share of it."
section "nvidia-smi query" nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.memory --format=csv

# --- baseline ----------------------------------------------------------------

if [[ "$MODE" == smoke ]]; then
    section "Baseline: build" make -s
    for t in test_task1_rope test_task2_reduce test_task3_gemm test_task4_attention; do
        section "Baseline: $t small" "build/sm_$SM/$t" small
    done
    section "Baseline: one transcription" ./run.sh --warmup 0 --runs 1
else
    section "Baseline: make test" make test
    section "Baseline: ./run.sh --profile" ./run.sh --profile
fi

# --- experiments -------------------------------------------------------------

note "# Try this experiments"

# sweep TAG FILE CONSTANT BASELINE TEST VALUES...
# One variant per value. Every sed edit is applied and checked, but smoke mode only
# builds and runs the first value (the edits are identical apart from the number).
sweep() {
    local tag="$1" file="$2" const="$3" base="$4" test="$5"; shift 5
    local first=1 v
    note "Baseline value $const = $base: see the Baseline sections."
    for v in "$@"; do
        note "### $const = $v"
        if make_variant "${tag}_$v" "$file" "s/^(constexpr int $const = )$base;/\\1$v;/" \
                "^constexpr int $const = $v;"; then
            if [[ "$MODE" == full || $first == 1 ]]; then
                run_variant "${tag}_$v" "$test"
            else
                note "Smoke mode: edit checked, not built."
            fi
        fi
        first=0
    done
}

note "## Task 1, Try this 1: ROPE_THREADS sweep (watch the audio encoder GB/s)"
sweep t1_rope task1_rope.cu ROPE_THREADS 256 test_task1_rope 32 64 128 512 1024

note "## Task 2, Try this 1: REDUCE_THREADS sweep (watch layernorm audio encoder and softmax vocabulary row)"
sweep t2_reduce task2_reduce.cu REDUCE_THREADS 256 test_task2_reduce 32 64 128 512 1024

note "## Task 3, Try this 1: TILE sweep (encoder cases, and lm_head at TILE = 32)"
sweep t3_tile task3_gemm.cu TILE 16 test_task3_gemm 8 32

note "## Task 4, Try this 3: attention on PyTorch, everything else on your kernels"
note "No code change. Compare the attention line with the Baseline profile."
if [[ "$MODE" == smoke ]]; then
    note "Skipped in smoke mode."
else
    section "./run.sh --cuda task1,task2,task3 --profile" ./run.sh --cuda task1,task2,task3 --profile
fi

# --- finish ------------------------------------------------------------------

[[ -f "$OUT" ]] && mv "$OUT" "$OUT.bak"
mv "$TMP" "$OUT"
echo ">>> wrote $PWD/$OUT"
