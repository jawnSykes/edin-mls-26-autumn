#!/bin/bash
# Transcribe + encode all 475 audio queries once, save them to ~/mls-rag-assets/query_vectors/
# and zip that to ~/mls-rag-assets/query_vectors.zip. Benchmarks then just load the .npy.
#
# Needs the mls-asr conda env (for ASR) and mls-rag-env (for the encoder).
# Transcripts are cached in the output dir, so a rerun only redoes ASR for missing queries.
# Existing rag/.cache/*transcripts.jsonl (from evaluate_rag.py) are used to seed the cache.
#
# Submit from the repo root:
#     sbatch rag/scripts/build_query_vectors.sbatch
#     cat query-vectors-JOB_ID.out
# Then, on your local machine:
#     scp <head-node>:mls-rag-assets/query_vectors.zip .
#SBATCH -p Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=query-vectors-%j.out

set -euo pipefail

export PATH=/opt/cuda-12.8.0/bin:$PATH
# shellcheck disable=SC1091
source "${RAG_VENV:-$HOME/mls-rag-env}/bin/activate"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"

# ASR runs in its own interpreter (see rag/README.md)
export RAG_ASR_PYTHON="${RAG_ASR_PYTHON:-${CONDA_BASE:-$HOME/miniconda3}/envs/mls-asr/bin/python}"
[[ -x "$RAG_ASR_PYTHON" ]] || { echo "ASR python not found: $RAG_ASR_PYTHON (set RAG_ASR_PYTHON)" >&2; exit 1; }

# cpu matches the README's evaluate_rag.py command; ASR still uses the GPU
DEVICE="${DEVICE:-cpu}"
OUT="$HOME/mls-rag-assets/query_vectors"
CACHE="$OUT/transcripts.jsonl"
mkdir -p "$OUT"

hostname
nvidia-smi -L
echo "ASR python: $RAG_ASR_PYTHON | encoder device: $DEVICE | out: $OUT"

if [[ ! -s "$CACHE" ]] && compgen -G "rag/.cache/*transcripts.jsonl" >/dev/null; then
    echo "Seeding transcript cache from rag/.cache/"
    cat rag/.cache/*transcripts.jsonl > "$CACHE"
fi

python rag/scripts/build_query_vectors.py --transcript-cache "$CACHE" --out-dir "$OUT" --device "$DEVICE"

# Zip next to the folder (python's zipfile, in case `zip` isn't installed)
cd "$HOME/mls-rag-assets"
rm -f query_vectors.zip
python -m zipfile -c query_vectors.zip query_vectors/
ls -la query_vectors query_vectors.zip
