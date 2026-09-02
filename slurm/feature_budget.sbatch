#!/usr/bin/env bash
#SBATCH --job-name=tcga-feature-budget
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tcga-feature-budget-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MATRIX="${MATRIX:-$REPO_ROOT/data/cache/expression.float32.npy}"
SPLIT="${SPLIT:-$REPO_ROOT/data/processed/split_manifest.tsv}"
GENES="${GENES:-$REPO_ROOT/data/cache/genes.tsv}"
OUTDIR="${OUTDIR:-$REPO_ROOT/results/feature_budget/elastic_net}"
MODEL="${MODEL:-elastic_net}"
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

python -u -m tcga_ml.feature_budget_cli \
  --matrix "$MATRIX" \
  --split "$SPLIT" \
  --genes "$GENES" \
  --outdir "$OUTDIR" \
  --model "$MODEL" \
  --negative-policy "$NEGATIVE_POLICY" \
  --n-jobs "$N_JOBS"
