#!/usr/bin/env bash
#SBATCH --job-name=tcga-xgb-gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tcga-xgb-gpu-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MATRIX="${MATRIX:-$REPO_ROOT/data/cache/expression.float32.npy}"
SPLIT="${SPLIT:-$REPO_ROOT/data/processed/split_manifest.tsv}"
GENES="${GENES:-$REPO_ROOT/data/cache/genes.tsv}"
OUTDIR="${OUTDIR:-$REPO_ROOT/results/benchmarks/xgboost-gpu}"
GENE_BUDGET="${GENE_BUDGET:-1000}"
NEGATIVE_POLICY="${NEGATIVE_POLICY:-clip}"
THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$REPO_ROOT/logs" "$OUTDIR"
cd "$REPO_ROOT"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

python -m tcga_ml.xgboost_cli probe
python -m tcga_ml.xgboost_cli cv \
  --matrix "$MATRIX" \
  --split "$SPLIT" \
  --genes "$GENES" \
  --outdir "$OUTDIR" \
  --device cuda \
  --threads "$THREADS" \
  --fold-jobs 1 \
  --gene-budget "$GENE_BUDGET" \
  --negative-policy "$NEGATIVE_POLICY"
