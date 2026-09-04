#!/usr/bin/env bash
#SBATCH --job-name=tcga-xgb-cv
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tcga-xgb-cv-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MATRIX="${MATRIX:-$REPO_ROOT/data/cache/expression.float32.npy}"
SPLIT="${SPLIT:-$REPO_ROOT/data/processed/split_manifest.tsv}"
GENES="${GENES:-$REPO_ROOT/data/cache/genes.tsv}"
OUTDIR="${OUTDIR:-$REPO_ROOT/results/benchmarks/xgboost-cpu}"
GENE_BUDGET="${GENE_BUDGET:-1000}"
NEGATIVE_POLICY="${NEGATIVE_POLICY:-clip}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

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

if [[ -s "$OUTDIR/xgboost_benchmark.json" ]]; then
  echo "[xgboost] complete output already exists: $OUTDIR/xgboost_benchmark.json"
  exit 0
fi

python -u -m tcga_ml.xgboost_cli cv \
  --matrix "$MATRIX" \
  --split "$SPLIT" \
  --genes "$GENES" \
  --outdir "$OUTDIR" \
  --device cpu \
  --threads "$THREADS" \
  --fold-jobs 1 \
  --gene-budget "$GENE_BUDGET" \
  --negative-policy "$NEGATIVE_POLICY"

test -s "$OUTDIR/xgboost_benchmark.json"
test -s "$OUTDIR/xgboost_feature_importance.tsv"
echo "[xgboost] complete: $OUTDIR/xgboost_benchmark.json"
