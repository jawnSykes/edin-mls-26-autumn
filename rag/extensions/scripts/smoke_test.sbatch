#!/bin/bash
# Smoke test: every step of run_experiments.sbatch on tiny grids (two corpora, M=16, three
# efSearch values, 20 queries per cell, two simulated LL sizes), to check the setup end to end
# before the real run. Writes rag/results/ann-smoke/ and ~/mls-rag-assets/ann-cache-smoke/,
# never the real results or cache. Numbers from it mean nothing; it only has to finish.
#
#   sbatch --gres=gpu:1 rag/extensions/scripts/smoke_test.sbatch    # from the repo root
#   cat ann-smoke-JOB_ID.out                                        # ends with "== all done"
#   sacct -j JOB_ID -o JobID,MaxRSS,Elapsed                         # memory and time actually used
# Without --gres=gpu:1 it skips step 11 (pipeline). Extra settings pass through, e.g.
#   sbatch --export=ALL,STEPS=env rag/extensions/scripts/smoke_test.sbatch
#SBATCH -p Teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=ann-smoke-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/../../..}"
SMOKE=1 exec bash rag/extensions/scripts/run_experiments.sbatch
