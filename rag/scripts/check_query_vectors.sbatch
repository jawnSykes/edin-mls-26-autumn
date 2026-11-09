#!/bin/bash
# Check the saved query vectors (from build_query_vectors.sbatch) match evaluate_rag.py:
# run evaluate_rag.py on the same transcript cache, recompute the metrics from the saved
# vectors, and diff the two "Aggregate:" blocks for each split.
#
# CPU only: every transcript is cached, so evaluate_rag.py never starts ASR.
#
# Submit from the repo root, after build_query_vectors.sbatch has finished:
#     sbatch rag/scripts/check_query_vectors.sbatch
#     cat check-query-vectors-JOB_ID.out      # ends with PASS or FAIL
#SBATCH -p Teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=check-query-vectors-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source "${RAG_VENV:-$HOME/mls-rag-env}/bin/activate"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"

VEC="$HOME/mls-rag-assets/query_vectors"
DEVICE="${DEVICE:-cpu}"   # must match the device used in build_query_vectors.sbatch
[[ -f "$VEC/queries_enc_a.npy" ]] || { echo "No saved vectors in $VEC" >&2; exit 1; }

# Per-job temp files on node-local scratch, removed at exit
SCRATCH="/disk/scratch/$USER/check-query-vectors/${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"; rmdir "/disk/scratch/$USER/check-query-vectors" 2>/dev/null || true' EXIT

hostname
status=0
for split in practice dev; do
    echo "===== $split ====="
    python rag/scripts/evaluate_rag.py --split "$split" --device "$DEVICE" \
        --transcript-cache "$VEC/transcripts.jsonl" > "$SCRATCH/eval.txt"
    python rag/scripts/check_query_vectors.py --split "$split" --vectors-dir "$VEC" > "$SCRATCH/saved.txt"

    # Keep only the Aggregate block from each
    sed -n '/^Aggregate:/,$p' "$SCRATCH/eval.txt" > "$SCRATCH/eval_agg.txt"
    sed -n '/^Aggregate:/,$p' "$SCRATCH/saved.txt" > "$SCRATCH/saved_agg.txt"
    echo "--- evaluate_rag.py ---"; cat "$SCRATCH/eval_agg.txt"
    echo "--- saved vectors ---";   cat "$SCRATCH/saved_agg.txt"
    if diff "$SCRATCH/eval_agg.txt" "$SCRATCH/saved_agg.txt"; then
        echo "$split: MATCH"
    else
        echo "$split: MISMATCH"; status=1
    fi
done

[[ $status -eq 0 ]] && echo "PASS" || echo "FAIL"
exit $status
