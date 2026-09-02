#!/usr/bin/env bash
#SBATCH --job-name=tcga-classical
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tcga-classical-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MATRIX="${MATRIX:-$REPO_ROOT/data/cache/expression.float32.npy}"
SPLIT="${SPLIT:-$REPO_ROOT/data/processed/split_manifest.tsv}"
OUTDIR="${OUTDIR:-$REPO_ROOT/results/benchmarks/classical}"
NEGATIVE_POLICY="${NEGATIVE_POLICY:-clip}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"

module --force purge
module load StdEnv/2023
module load python/3.11

cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"
mkdir -p "$REPO_ROOT/logs" "$OUTDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ -s "$OUTDIR/benchmark.json" ]]; then
  echo "[classical] complete output already exists: $OUTDIR/benchmark.json"
  exit 0
fi

python -u -m tcga_ml.benchmark_cli \
  --matrix "$MATRIX" \
  --split "$SPLIT" \
  --outdir "$OUTDIR" \
  --negative-policy "$NEGATIVE_POLICY" \
  --n-jobs "$N_JOBS"

test -s "$OUTDIR/benchmark.json"
echo "[classical] complete: $OUTDIR/benchmark.json"
